//! Chunk planning: how gdalwarp splits a target into chunks, which source
//! window each chunk reads, and the kernel scale each chunk is warped at,
//! returned as values.
//!
//! GDAL's warp operation plans before it reads (`CollectChunkList`,
//! gdalwarpoperation.cpp ~L977-1670 at `739fcf03bf`, GDAL 3.14.0dev): it
//! computes the source window of the whole target (`ComputeSourceWindow`,
//! ~L2640-3390), and while that window would take more working memory than
//! the warp memory limit (gdalwarp's `-wm`), or reads mostly outside the
//! source (a fill ratio below 0.5), halves the target along its longer side
//! and plans each half again. Each chunk is then warped on its own, with
//! the kernel scale estimated for it (`GDALWarpKernel::PerformWarp`).
//!
//! The chunk boundaries matter to the output, not only to memory: they set
//! the kernel scale, the window edges where kernels are clipped, and the
//! scanline extents the approximate transformer interpolates over. Warping
//! a whole raster cell for cell as gdalwarp does needs gdalwarp's chunks.
//!
//! [`plan_chunks`] returns a [`WarpPlan`]: target windows, source windows,
//! fill ratios and kernel scales, with serde derives behind the `serde`
//! feature. It does no I/O; read each chunk's source window however suits
//! (a COG, Zarr, an array in memory) and warp it with
//! [`Chunk::warp_options`], or hand a source buffer covering the plan to
//! [`warp_chunks`].
//!
//! ```
//! use rwarp::plan::{plan_chunks, ChunkOptions};
//! use rwarp::warp::ResampleAlg;
//! # use rwarp::Transformer;
//! # struct Half;
//! # impl Transformer for Half {
//! #     fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
//! #         let f = if dst_to_src { 2.0 } else { 0.5 };
//! #         x.iter_mut().chain(y.iter_mut()).for_each(|v| *v *= f);
//! #         vec![true; x.len()]
//! #     }
//! # }
//! // a 1000 x 1000 Float32 source to 500 x 500, in 1 MB of working memory
//! let opts = ChunkOptions::new(ResampleAlg::Bilinear)
//!     .with_memory_limit(1024.0 * 1024.0)
//!     .with_pixel_cost(rwarp::warp::DataType::Float32, 1, false, false);
//! let plan = plan_chunks(&Half, [1000, 1000], [500, 500], &opts);
//! assert!(plan.chunks.len() > 1);
//! for c in plan.live() {
//!     // read c.src_off .. c.src_off + c.src_size, then warp with c.warp_options(..)
//!     assert_eq!(c.scale, [0.5, 0.5]);
//! }
//! ```
//!
//! ## Where this differs from gdalwarp
//!
//! By default the plan follows gdalwarp except in two places, each an
//! option ([`ChunkOptions::gdalwarp`] turns both off):
//!
//! - [`ScaleOrigin::Chunk`]: GDAL estimates every chunk's kernel scale at
//!   the output's top-left, because its sample points omit the chunk
//!   offset; rwarp estimates it over the chunk ([`ScaleOrigin::Output`] is
//!   GDAL's).
//! - [`ChunkOptions::pad_for_fixed_scale`]: under `XSCALE` / `YSCALE`
//!   GDAL pads source windows for the scale it estimates, not the one
//!   given, so a kernel stretched further is clipped at the window edge.
//!
//! [`ChunkOptions::split_antimeridian`] (off by default) adds the split at
//! a discontinuity of 0.2's `collect_chunk_list`, which this replaces.
//!
//! Not modelled: `CHECK_WITH_INVERT_PROJ` (a PROJ switch, retried by GDAL
//! when a window comes back with wild values), `STREAMABLE_OUTPUT`, and
//! GDAL's shortcut for a translation by whole pixels (no kernel padding,
//! nearest neighbour).

use std::cell::OnceCell;

use crate::source_window::{find_discontinuity_range, SourceWindow};
use crate::transform::Transformer;
use crate::warp::{estimate_scale, warp_to, DataType, KernelScaling, ResampleAlg, Sample, SourceBuffer, WarpOptions};

/// GDAL's default warp memory limit, 64 MiB (`dfWarpMemoryLimit`).
pub const DEFAULT_MEMORY_LIMIT: f64 = 64.0 * 1024.0 * 1024.0;

/// GDAL's `DEFAULT_STEP_COUNT`.
const DEFAULT_STEP_COUNT: usize = 21;

/// Where each chunk's kernel scale is estimated (see [`estimate_scale`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ScaleOrigin {
    /// Over the chunk itself (rwarp's default).
    #[default]
    Chunk,
    /// Over a window of the chunk's size at the output's top-left, as
    /// GDAL does: its sample points omit the chunk offset, so equal-sized
    /// chunks get the same scale wherever they are.
    Output,
}

/// How GDAL samples the target window to find the source window (the
/// `SAMPLE_STEPS` warp option).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SampleSteps {
    /// 21 points per edge, or every edge pixel if a corner of the window
    /// fails to transform (GDAL's default).
    #[default]
    Auto,
    /// This many points per edge (at least 2), without the corner check.
    Count(usize),
    /// Every pixel edge (`SAMPLE_STEPS=ALL`).
    All,
}

/// A check on the forward probe (the source grid transformed to the
/// target, GDAL's `ComputeSourceWindowStartingFromSource`): a point that
/// does not transform back to within one source pixel of where it started
/// is dropped. GDAL has no such check. It is for backends that return a
/// point instead of failing, such as proj4rs at the antipode of a polar
/// stereographic projection, which would put the opposite pole of the
/// source inside the target.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct RoundTrip {
    /// For a source that wraps in x (a global lon/lat grid): its width in
    /// pixels, so a point that comes back a whole turn away still matches.
    pub wrap_width: Option<f64>,
    /// For a source that runs pole to pole: its height in pixels. On the
    /// pole rows (y at 0 or at this height) any x goes back to the same
    /// point, so only y is compared there.
    pub pole_rows: Option<f64>,
}

impl RoundTrip {
    /// Compare x modulo `width` source pixels (a source that wraps).
    pub fn with_wrap_width(mut self, width: f64) -> Self {
        self.wrap_width = Some(width);
        self
    }

    /// Compare only y on the pole rows of a source `height` pixels tall.
    pub fn with_pole_rows(mut self, height: f64) -> Self {
        self.pole_rows = Some(height);
        self
    }
}

/// Options for [`plan_chunks`] and [`ChunkPlanner`]. Build with
/// [`ChunkOptions::new`] or [`ChunkOptions::gdalwarp`] and the `with_`
/// methods.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ChunkOptions {
    /// The resampling method: its kernel radius pads source windows.
    pub alg: ResampleAlg,
    /// How each chunk's kernel scale is chosen: estimated per chunk
    /// ([`KernelScaling::WidenWhenDownsampling`]), given
    /// ([`KernelScaling::Fixed`], GDAL's `XSCALE`/`YSCALE`) or 1
    /// ([`KernelScaling::Point`]).
    pub scaling: KernelScaling,
    /// Bytes of working memory a chunk may take before it is split
    /// (gdalwarp's `-wm`). Default [`DEFAULT_MEMORY_LIMIT`].
    pub memory_limit: f64,
    /// Bits of working memory per source pixel, masks included (GDAL's
    /// `GetWorkingMemoryForWindow`). See [`ChunkOptions::set_pixel_cost`].
    pub src_pixel_bits: f64,
    /// Bits of working memory per target pixel, masks included.
    pub dst_pixel_bits: f64,
    /// A chunk whose source fill ratio is above 0 and below this is split
    /// (GDAL: 0.5; 0 turns the heuristic off, as
    /// `SRC_FILL_RATIO_HEURISTICS=NO`).
    pub min_fill_ratio: f64,
    /// ... if either side of it is longer than this (GDAL: 100).
    pub min_split_size: usize,
    /// Block size of the target dataset, for GDAL's `OPTIMIZE_SIZE=AUTO`,
    /// which splits on block boundaries when the target is at least two
    /// blocks each way. `None` (GDAL with no target dataset, as with an
    /// in-memory or striped target) never applies it.
    pub dst_block_size: Option<[usize; 2]>,
    /// Target pixel positions that force grid sampling of every chunk when
    /// one falls in the target (`0 <= x <= width`, `0 <= y <= height`).
    /// GDAL uses longitude 0 at latitudes -89.9999 and 89.9999 of the
    /// source's geographic CRS, transformed to target pixels; the caller
    /// computes them, as they need the CRS.
    pub special_points: Vec<[f64; 2]>,
    /// Sample a grid over each chunk from the start (`SAMPLE_GRID=YES`).
    pub sample_grid: bool,
    /// Points per edge (`SAMPLE_STEPS`).
    pub sample_steps: SampleSteps,
    /// Pixels added to every side of each source window (`SOURCE_EXTRA`).
    /// `None`: 10 when any sample point failed to transform, as GDAL does.
    pub source_extra: Option<usize>,
    /// Where each chunk's kernel scale is estimated. Default
    /// [`ScaleOrigin::Chunk`]; GDAL's is [`ScaleOrigin::Output`].
    pub scale_origin: ScaleOrigin,
    /// With [`KernelScaling::Fixed`], pad source windows for the smaller
    /// of the given and the estimated scale, so a kernel stretched by the
    /// given scale is not clipped at the window edge. GDAL pads for the
    /// estimate only. Default `true`.
    pub pad_for_fixed_scale: bool,
    /// When a chunk's source window spans the whole source width (GDAL's
    /// 90% rule, as at the antimeridian), find the target columns where
    /// source x jumps and split the chunk there, into compact windows
    /// either side. GDAL does not (its fill ratio stays high); off by
    /// default.
    pub split_antimeridian: bool,
    /// A round-trip check on the forward probe; `None` (GDAL) checks
    /// nothing.
    pub round_trip: Option<RoundTrip>,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        ChunkOptions::new(ResampleAlg::NearestNeighbour)
    }
}

impl ChunkOptions {
    /// rwarp's defaults for `alg`: gdalwarp's planning with a Byte,
    /// one-band, nodata-free working memory cost, except for
    /// [`ScaleOrigin::Chunk`] and [`ChunkOptions::pad_for_fixed_scale`].
    pub fn new(alg: ResampleAlg) -> Self {
        ChunkOptions {
            alg,
            scaling: KernelScaling::WidenWhenDownsampling,
            memory_limit: DEFAULT_MEMORY_LIMIT,
            src_pixel_bits: 8.0,
            dst_pixel_bits: 8.0,
            min_fill_ratio: 0.5,
            min_split_size: 100,
            dst_block_size: None,
            special_points: Vec::new(),
            sample_grid: false,
            sample_steps: SampleSteps::Auto,
            source_extra: None,
            scale_origin: ScaleOrigin::Chunk,
            pad_for_fixed_scale: true,
            split_antimeridian: false,
            round_trip: None,
        }
    }

    /// gdalwarp's planning exactly: [`ScaleOrigin::Output`] and no padding
    /// for a fixed scale beyond GDAL's.
    pub fn gdalwarp(alg: ResampleAlg) -> Self {
        ChunkOptions { scale_origin: ScaleOrigin::Output, pad_for_fixed_scale: false, ..ChunkOptions::new(alg) }
    }

    /// Set [`ChunkOptions::alg`].
    pub fn with_alg(mut self, alg: ResampleAlg) -> Self {
        self.alg = alg;
        self
    }

    /// Set [`ChunkOptions::scaling`].
    pub fn with_scaling(mut self, scaling: KernelScaling) -> Self {
        self.scaling = scaling;
        self
    }

    /// Set [`ChunkOptions::memory_limit`], in bytes.
    pub fn with_memory_limit(mut self, bytes: f64) -> Self {
        self.memory_limit = bytes;
        self
    }

    /// [`ChunkOptions::set_pixel_cost`], as a builder.
    pub fn with_pixel_cost(mut self, working: DataType, nbands: usize, src_nodata: bool, dst_nodata: bool) -> Self {
        self.set_pixel_cost(working, nbands, src_nodata, dst_nodata);
        self
    }

    /// Set [`ChunkOptions::min_fill_ratio`] and
    /// [`ChunkOptions::min_split_size`].
    pub fn with_fill_ratio_split(mut self, min_fill_ratio: f64, min_split_size: usize) -> Self {
        self.min_fill_ratio = min_fill_ratio;
        self.min_split_size = min_split_size;
        self
    }

    /// Set [`ChunkOptions::dst_block_size`].
    pub fn with_dst_block_size(mut self, block: Option<[usize; 2]>) -> Self {
        self.dst_block_size = block;
        self
    }

    /// Set [`ChunkOptions::special_points`].
    pub fn with_special_points(mut self, points: Vec<[f64; 2]>) -> Self {
        self.special_points = points;
        self
    }

    /// Set [`ChunkOptions::sample_grid`].
    pub fn with_sample_grid(mut self, grid: bool) -> Self {
        self.sample_grid = grid;
        self
    }

    /// Set [`ChunkOptions::sample_steps`].
    pub fn with_sample_steps(mut self, steps: SampleSteps) -> Self {
        self.sample_steps = steps;
        self
    }

    /// Set [`ChunkOptions::source_extra`].
    pub fn with_source_extra(mut self, extra: Option<usize>) -> Self {
        self.source_extra = extra;
        self
    }

    /// Set [`ChunkOptions::scale_origin`].
    pub fn with_scale_origin(mut self, origin: ScaleOrigin) -> Self {
        self.scale_origin = origin;
        self
    }

    /// Set [`ChunkOptions::pad_for_fixed_scale`].
    pub fn with_pad_for_fixed_scale(mut self, pad: bool) -> Self {
        self.pad_for_fixed_scale = pad;
        self
    }

    /// Set [`ChunkOptions::split_antimeridian`].
    pub fn with_split_antimeridian(mut self, split: bool) -> Self {
        self.split_antimeridian = split;
        self
    }

    /// Set [`ChunkOptions::round_trip`].
    pub fn with_round_trip(mut self, round_trip: Option<RoundTrip>) -> Self {
        self.round_trip = round_trip;
        self
    }

    /// Set the working memory cost per pixel as GDAL counts it for
    /// `nbands` bands in the `working` type, with a per-band validity mask
    /// when the source or the target has nodata. gdalwarp gives the target
    /// the source's nodata unless told otherwise. Add 32 bits to
    /// [`ChunkOptions::src_pixel_bits`] for a source alpha band or a
    /// cutline, 1 for a per-dataset mask; 32 to
    /// [`ChunkOptions::dst_pixel_bits`] for a target alpha band.
    pub fn set_pixel_cost(&mut self, working: DataType, nbands: usize, src_nodata: bool, dst_nodata: bool) {
        let bits = (type_bits(working) * nbands) as f64;
        let mask = nbands as f64;
        self.src_pixel_bits = bits + if src_nodata { mask } else { 0.0 };
        self.dst_pixel_bits = bits + if dst_nodata { mask } else { 0.0 };
    }
}

fn type_bits(t: DataType) -> usize {
    use DataType::*;
    match t {
        UInt8 | Int8 => 8,
        UInt16 | Int16 | Float16 => 16,
        UInt32 | Int32 | Float32 => 32,
        UInt64 | Int64 | Float64 => 64,
    }
}

/// One chunk of a [`WarpPlan`]: a target window and the source window it
/// reads, in pixels of the full target and source.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Chunk {
    /// Target window offset `[col, row]`.
    pub dst_off: [usize; 2],
    /// Target window size `[ncol, nrow]`.
    pub dst_size: [usize; 2],
    /// Source window offset `[col, row]`.
    pub src_off: [usize; 2],
    /// Source window size `[ncol, nrow]`; zero when the chunk sees no
    /// source (its pixels keep the fill value).
    pub src_size: [usize; 2],
    /// GDAL's fill ratio of the source window: its area over the area of
    /// the unclamped window. 0 for an empty window.
    pub fill_ratio: f64,
    /// Kernel scale, target pixels per source pixel along x and y (below 1
    /// is downsampling); `[1, 1]` for an empty chunk.
    pub scale: [f64; 2],
}

impl Chunk {
    /// Whether the chunk reads nothing.
    pub fn is_empty(&self) -> bool {
        self.src_size[0] == 0 || self.src_size[1] == 0
    }

    /// `base` with this chunk's target offset and kernel scale, for
    /// [`crate::warp::warp`] on this chunk's source window into a
    /// `dst_size` target.
    pub fn warp_options(&self, base: &WarpOptions) -> WarpOptions {
        WarpOptions {
            scaling: KernelScaling::Fixed { x: self.scale[0], y: self.scale[1] },
            dst_col_off: self.dst_off[0],
            dst_row_off: self.dst_off[1],
            ..base.clone()
        }
    }
}

/// A warp's chunks, sorted top to bottom, then left to right, as GDAL
/// warps them.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WarpPlan {
    /// Size of the full source `[ncol, nrow]`.
    pub src_size: [usize; 2],
    /// Offset of the planned target window (zero for a whole target).
    pub dst_off: [usize; 2],
    /// Size of the planned target window.
    pub dst_size: [usize; 2],
    /// The chunks, covering the target window without overlap.
    pub chunks: Vec<Chunk>,
}

impl WarpPlan {
    /// The chunks that read something.
    pub fn live(&self) -> impl Iterator<Item = &Chunk> {
        self.chunks.iter().filter(|c| !c.is_empty())
    }

    /// The smallest source window covering every chunk's, as
    /// `(offset, size)`, or `None` if no chunk reads anything.
    pub fn src_region(&self) -> Option<([usize; 2], [usize; 2])> {
        let mut it = self.live().peekable();
        it.peek()?;
        let (mut lo, mut hi) = ([usize::MAX; 2], [0; 2]);
        for c in it {
            for i in 0..2 {
                lo[i] = lo[i].min(c.src_off[i]);
                hi[i] = hi[i].max(c.src_off[i] + c.src_size[i]);
            }
        }
        Some((lo, [hi[0] - lo[0], hi[1] - lo[1]]))
    }

    /// Pixels read in all, summed over chunks (windows can overlap).
    pub fn src_pixels(&self) -> usize {
        self.live().map(|c| c.src_size[0] * c.src_size[1]).sum()
    }
}

/// Plan a warp of a `src_size` source to a whole `dst_size` target
/// through `transformer` (target pixels to source pixels). Use the
/// transformer the warp will use: gdalwarp plans with its approximate
/// transformer. See the [module documentation](self).
pub fn plan_chunks(
    transformer: &impl Transformer,
    src_size: [usize; 2],
    dst_size: [usize; 2],
    opts: &ChunkOptions,
) -> WarpPlan {
    ChunkPlanner::new(transformer, src_size, dst_size, opts.clone()).plan()
}

/// A source grid point forward transformed: (src x, src y, dst x, dst y,
/// ok).
type GridPoint = (f64, f64, f64, f64, bool);

/// A computed source window, before it becomes a [`SourceWindow`].
struct Win {
    off: [i32; 2],
    size: [i32; 2],
    fill: f64,
    failed: i32,
    samples: i32,
    /// The x extent took the whole source width (GDAL's 90% rule).
    full_width: bool,
}

/// GDAL's warp planning for one source and target, holding what it
/// computes once per warp (the forward-transformed source grid). Use it
/// for several windows of one target, or for single source windows
/// exactly as GDAL computes them.
pub struct ChunkPlanner<'a, T: Transformer> {
    t: &'a T,
    src: [i32; 2],
    dst: [i32; 2],
    opts: ChunkOptions,
    source_grid: OnceCell<Vec<GridPoint>>,
}

impl<'a, T: Transformer> ChunkPlanner<'a, T> {
    /// A planner for a `src_size` source and a `dst_size` target.
    pub fn new(transformer: &'a T, src_size: [usize; 2], dst_size: [usize; 2], opts: ChunkOptions) -> Self {
        let i = |v: usize| i32::try_from(v).expect("raster size beyond i32");
        ChunkPlanner {
            t: transformer,
            src: src_size.map(i),
            dst: dst_size.map(i),
            opts,
            source_grid: OnceCell::new(),
        }
    }

    /// The options in use.
    pub fn options(&self) -> &ChunkOptions {
        &self.opts
    }

    /// Chunks for the whole target.
    pub fn plan(&self) -> WarpPlan {
        self.plan_window([0, 0], self.dst.map(|v| v as usize))
    }

    /// Chunks for one window of the target (GDAL's `ChunkAndWarpImage` on
    /// a window).
    pub fn plan_window(&self, dst_off: [usize; 2], dst_size: [usize; 2]) -> WarpPlan {
        let mut chunks = Vec::new();
        let i = |v: usize| v as i32;
        self.collect(dst_off.map(i), dst_size.map(i), &mut chunks);
        chunks.sort_by_key(|c| (c.dst_off[1], c.dst_off[0]));
        WarpPlan { src_size: self.src.map(|v| v as usize), dst_off, dst_size, chunks }
    }

    /// The source window of one target window, as GDAL's
    /// `ComputeSourceWindow` gives it: `None` if too few sample points
    /// transform (GDAL then skips the window), a zero-sized window if the
    /// target misses the source.
    ///
    /// Unlike [`crate::source_window::compute_source_window`], this keeps
    /// GDAL's arithmetic throughout: the forward probe and special points,
    /// bounds carried between sampling passes, no extra pixel beyond the
    /// kernel radius, and GDAL's fill ratio for a window spanning the
    /// source width.
    pub fn source_window(&self, dst_off: [usize; 2], dst_size: [usize; 2]) -> Option<SourceWindow> {
        let i = |v: usize| v as i32;
        self.window(dst_off.map(i), dst_size.map(i)).map(|w| SourceWindow {
            xoff: w.off[0],
            yoff: w.off[1],
            xsize: w.size[0],
            ysize: w.size[1],
            fill_ratio: w.fill,
            n_failed: w.failed,
            n_samples: w.samples,
        })
    }

    /// GDAL's `CollectChunkListInternal`.
    fn collect(&self, off: [i32; 2], size: [i32; 2], out: &mut Vec<Chunk>) {
        let w = self.window(off, size);
        let (src_size, mut fill, full_width) = match &w {
            Some(w) => (w.size, w.fill, w.full_width),
            None => ([0, 0], 0.0, false),
        };
        let o = &self.opts;
        let [dw, dh] = size;
        let mut antimeridian = false;
        if o.split_antimeridian && full_width && src_size[0] > 0 && src_size[1] > 0 {
            // GDAL's fill ratio stays high when the 90% rule takes the
            // whole width; mark the window as wasteful instead.
            fill = 0.01;
            antimeridian = true;
        }
        let memory = (o.src_pixel_bits * src_size[0] as f64 * src_size[1] as f64
            + o.dst_pixel_bits * dw as f64 * dh as f64)
            / 8.0;
        let big = o.min_split_size as i32;
        let split = (memory > o.memory_limit && (dw > 2 || dh > 2))
            || (fill > 0.0 && fill < o.min_fill_ratio && (dw > big || dh > big));
        if split {
            if antimeridian {
                if let Some((c0, c1)) = find_discontinuity_range(self.t, off, size, self.src) {
                    let left = c0 - off[0];
                    let right = off[0] + dw - c1;
                    if left > 0 {
                        self.collect(off, [left, dh], out);
                    }
                    if c1 > c0 {
                        // The gap itself is not split again.
                        self.push(out, [c0, off[1]], [c1 - c0, dh], self.window([c0, off[1]], [c1 - c0, dh]));
                    }
                    if right > 0 {
                        self.collect([c1, off[1]], [right, dh], out);
                    }
                    return;
                }
            }
            let [bx, by] = o.dst_block_size.map_or([1, 1], |b| b.map(|v| v as i32));
            let optimize = o.dst_block_size.is_some()
                && src_size[0] > 0
                && dh > 0
                && {
                    let (sx, sy) = (src_size[0] as f64, src_size[1] as f64);
                    let (fx, fy) = (dw as f64, dh as f64);
                    if dw > dh {
                        (fx / fy - sx / sy).abs() < 5.0 * fx / fy
                    } else {
                        (fy / fx - sy / sx).abs() < 5.0 * fy / fx
                    }
                }
                && dw / 2 >= bx
                && dh / 2 >= by;
            if dw > dh && (!optimize || dw / 2 >= bx || dh == 1) {
                let mut a = dw / 2;
                if optimize && a > bx {
                    a = (a / bx) * bx;
                }
                self.collect(off, [a, dh], out);
                self.collect([off[0] + a, off[1]], [dw - a, dh], out);
            } else {
                let mut a = dh / 2;
                if optimize && a > by {
                    a = (a / by) * by;
                }
                self.collect(off, [dw, a], out);
                self.collect([off[0], off[1] + a], [dw, dh - a], out);
            }
            return;
        }
        self.push(out, off, size, w);
    }

    fn push(&self, out: &mut Vec<Chunk>, off: [i32; 2], size: [i32; 2], w: Option<Win>) {
        let u = |v: i32| v.max(0) as usize;
        let mut c = Chunk {
            dst_off: off.map(u),
            dst_size: size.map(u),
            src_off: [0, 0],
            src_size: [0, 0],
            fill_ratio: 0.0,
            scale: [1.0, 1.0],
        };
        if let Some(w) = w {
            c.src_off = w.off.map(u);
            c.src_size = w.size.map(u);
            c.fill_ratio = w.fill;
        }
        if !c.is_empty() {
            c.scale = self.chunk_scale(&c);
        }
        out.push(c);
    }

    /// The kernel scale of a chunk (`GDALWarpKernel::PerformWarp`).
    fn chunk_scale(&self, c: &Chunk) -> [f64; 2] {
        match self.opts.scaling {
            KernelScaling::Point => [1.0, 1.0],
            KernelScaling::Fixed { x, y } => [x, y],
            KernelScaling::WidenWhenDownsampling => {
                let off = match self.opts.scale_origin {
                    ScaleOrigin::Chunk => c.dst_off,
                    ScaleOrigin::Output => [0, 0],
                };
                let (x, y) = estimate_scale(self.t, c.dst_size[0], c.dst_size[1], off[0], off[1]);
                [x, y]
            }
        }
    }

    /// GDAL's `ComputeSourceWindowTransformPoints`: transform sample
    /// points of the target window to the source, widening `b`
    /// (min x, min y, max x, max y), which carries over between calls as
    /// in GDAL. Returns (samples, failed).
    fn sample(&self, off: [i32; 2], size: [i32; 2], grid: bool, all: bool, steps: usize, b: &mut [f64; 4]) -> (i32, i32) {
        let (ox, oy) = (off[0] as f64, off[1] as f64);
        let (sx, sy) = (size[0] as f64, size[1] as f64);
        let step = if all { 0.0 } else { 1.0 / (steps - 1) as f64 };
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        let mut push = |x: f64, y: f64| {
            xs.push(x);
            ys.push(y);
        };
        if grid {
            if all {
                for iy in 0..=size[1] {
                    for ix in 0..=size[0] {
                        push(ox + ix as f64, oy + iy as f64);
                    }
                }
            } else {
                // GDAL uses the x size for both near-edge ratios.
                let ratio = |i: usize| {
                    if i == 0 {
                        0.5 / sx
                    } else if i <= steps {
                        (i - 1) as f64 * step
                    } else {
                        1.0 - 0.5 / sx
                    }
                };
                for iy in 0..steps + 2 {
                    for ix in 0..steps + 2 {
                        push(ratio(ix) * sx + ox, ratio(iy) * sy + oy);
                    }
                }
            }
        } else if all {
            for ix in 0..=size[0] {
                push(ox + ix as f64, oy);
                push(ox + ix as f64, oy + sy);
            }
            for iy in 1..size[1] {
                push(ox, oy + iy as f64);
                push(ox + sx, oy + iy as f64);
            }
        } else {
            let mut r = 0.0;
            while r <= 1.0 + step * 0.5 {
                push(r * sx + ox, oy);
                push(r * sx + ox, oy + sy);
                push(ox, r * sy + oy);
                push(sx + ox, r * sy + oy);
                r += step;
            }
        }
        let ok = self.t.transform(true, &mut xs, &mut ys);
        let mut failed = 0;
        for i in 0..xs.len() {
            if !ok[i] || xs[i].is_nan() || ys[i].is_nan() {
                failed += 1;
                continue;
            }
            b[0] = b[0].min(xs[i]);
            b[1] = b[1].min(ys[i]);
            b[2] = b[2].max(xs[i]);
            b[3] = b[3].max(ys[i]);
        }
        (xs.len() as i32, failed)
    }

    fn steps(&self) -> usize {
        match self.opts.sample_steps {
            SampleSteps::Count(n) => n.max(2),
            _ => DEFAULT_STEP_COUNT,
        }
    }

    /// GDAL's `ComputeSourceWindowStartingFromSource`: source grid points
    /// that land in the target window widen `b`.
    fn probe_source(&self, off: [i32; 2], size: [i32; 2], b: &mut [f64; 4]) {
        let grid = self.source_grid.get_or_init(|| {
            let [w, h] = self.src.map(|n| n as f64);
            let steps = self.steps();
            let step = 1.0 / (steps - 1) as f64;
            let ratio = |i: usize, n: f64| {
                if i == 0 {
                    0.5 / n
                } else if i <= steps {
                    (i - 1) as f64 * step
                } else {
                    1.0 - 0.5 / n
                }
            };
            let (mut xs, mut ys) = (Vec::new(), Vec::new());
            for iy in 0..steps + 2 {
                for ix in 0..steps + 2 {
                    xs.push(ratio(ix, w) * w);
                    ys.push(ratio(iy, h) * h);
                }
            }
            let (sx, sy) = (xs.clone(), ys.clone());
            let mut ok = self.t.transform(false, &mut xs, &mut ys);
            if let Some(rt) = self.opts.round_trip {
                self.round_trip(&rt, &sx, &sy, &xs, &ys, &mut ok);
            }
            (0..xs.len()).map(|i| (sx[i], sy[i], xs[i], ys[i], ok[i])).collect()
        });
        if self.src[0] == 0 || self.src[1] == 0 {
            return;
        }
        let (x0, y0) = (off[0] as f64, off[1] as f64);
        let (x1, y1) = (x0 + size[0] as f64, y0 + size[1] as f64);
        for &(sx, sy, dx, dy, ok) in grid {
            if ok && dx >= x0 && dx <= x1 && dy >= y0 && dy <= y1 {
                b[0] = b[0].min(sx);
                b[1] = b[1].min(sy);
                b[2] = b[2].max(sx);
                b[3] = b[3].max(sy);
            }
        }
    }

    /// See [`RoundTrip`].
    fn round_trip(&self, rt: &RoundTrip, sx: &[f64], sy: &[f64], dx: &[f64], dy: &[f64], ok: &mut [bool]) {
        let (mut bx, mut by) = (dx.to_vec(), dy.to_vec());
        let back = self.t.transform(true, &mut bx, &mut by);
        for i in 0..ok.len() {
            if !ok[i] {
                continue;
            }
            let mut ex = (bx[i] - sx[i]).abs();
            if let Some(w) = rt.wrap_width {
                ex = ex.min(w - ex);
            }
            let pole = rt.pole_rows.is_some_and(|h| sy[i] <= 0.0 || sy[i] >= h);
            ok[i] = back[i] && (by[i] - sy[i]).abs() <= 1.0 && (pole || ex <= 1.0);
        }
    }

    /// GDAL's `ComputeSourceWindow`.
    fn window(&self, off: [i32; 2], size: [i32; 2]) -> Option<Win> {
        let o = &self.opts;
        let steps = self.steps();
        let mut grid = o.sample_grid;
        let mut all = o.sample_steps == SampleSteps::All;
        if o.sample_steps == SampleSteps::Auto && !grid {
            // A corner that fails to transform: sample every edge pixel
            // (GDAL #9056).
            let (ox, oy) = (off[0] as f64, off[1] as f64);
            let (ex, ey) = (ox + size[0] as f64, oy + size[1] as f64);
            let mut cx = [ox, ex, ox, ex];
            let mut cy = [oy, oy, ey, ey];
            if self.t.transform(true, &mut cx, &mut cy).iter().any(|ok| !ok) {
                all = true;
            }
        }
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        let (mut n, mut failed) = self.sample(off, size, grid, all, steps, &mut b);
        if !grid {
            let [dw, dh] = self.dst.map(|v| v as f64);
            if o.special_points.iter().any(|&[x, y]| 0.0 <= x && dw >= x && 0.0 <= y && dh >= y) {
                grid = true;
                all = false;
                (n, failed) = self.sample(off, size, grid, all, steps, &mut b);
            }
        }
        if !grid && failed > 0 {
            grid = true;
            all = false;
            (n, failed) = self.sample(off, size, grid, all, steps, &mut b);
        }
        if failed > n - 5 {
            return None;
        }
        if grid {
            self.probe_source(off, size, &mut b);
        }
        let [w, h] = self.src;
        let (wf, hf) = (w as f64, h as f64);
        if b[0] > wf || b[2] < 0.0 || b[1] > hf || b[3] < 0.0 {
            return Some(Win { off: [0, 0], size: [0, 0], fill: 0.0, failed, samples: n, full_width: false });
        }
        let snap = |v: f64| {
            let r = v.round();
            if (r - v).abs() < 1e-6 {
                r
            } else {
                v
            }
        };
        let [min_x, min_y, max_x, max_y] = b.map(snap);

        let r = o.alg.radius();
        let mut xscale = (size[0] as f64 / (max_x - min_x)).max(1e-3);
        let mut yscale = (size[1] as f64 / (max_y - min_y)).max(1e-3);
        if let (true, KernelScaling::Fixed { x, y }) = (o.pad_for_fixed_scale, o.scaling) {
            xscale = xscale.min(x.max(1e-3));
            yscale = yscale.min(y.max(1e-3));
        }
        let radius = |s: f64| if s < 0.95 { (r as f64 / s).ceil() as i32 } else { r };
        let (mut xr, mut yr) = (radius(xscale), radius(yscale));
        match o.source_extra {
            Some(e) => {
                xr += e as i32;
                yr += e as i32;
            }
            None if failed > 0 => {
                xr += 10;
                yr += 10;
            }
            None => {}
        }

        let min_xc = min_x.max(0.0) as i32;
        let min_yc = min_y.max(0.0) as i32;
        let max_xc = max_x.ceil().min(wf) as i32;
        let max_yc = max_y.ceil().min(hf) as i32;

        let axis = |min_c: i32, max_c: i32, n: i32, radius: i32| {
            if (max_c - min_c) as f64 > 0.9 * n as f64 {
                (0, n, true)
            } else {
                let o = (min_c - radius).min(n).max(0);
                (o, (max_c - o + radius).min(n - o).max(0), false)
            }
        };
        let (xo, xs, full_width) = axis(min_xc, max_xc, w, xr);
        let (yo, ys, _) = axis(min_yc, max_yc, h, yr);
        let fill = xs as f64 * ys as f64
            / ((max_x - min_x + 2.0 * xr as f64) * (max_y - min_y + 2.0 * yr as f64)).max(1.0);
        Some(Win { off: [xo, yo], size: [xs, ys], fill, failed, samples: n, full_width })
    }
}

/// Warp every chunk of `plan` from `src`, a source buffer covering every
/// chunk's source window (as [`WarpPlan::src_region`] gives), into one
/// target of the plan's window size. Each chunk is warped on its own with
/// [`Chunk::warp_options`] of `opts` (whose own target offset and scaling
/// are replaced), as gdalwarp does; pixels of empty chunks get the fill.
///
/// # Panics
///
/// If `src` does not cover a live chunk's source window.
pub fn warp_chunks<S: Sample, D: Sample>(
    transformer: &impl Transformer,
    plan: &WarpPlan,
    src: &SourceBuffer<'_, S>,
    opts: &WarpOptions,
) -> Vec<D> {
    let nb = src.nbands;
    let [w, h] = plan.dst_size;
    // The fill, from a warp that sees no source.
    let empty: SourceBuffer<'_, S> = SourceBuffer { data: &[], ncol: 0, nrow: 0, nbands: nb, col_off: 0, row_off: 0 };
    let o = WarpOptions { alg: ResampleAlg::NearestNeighbour, scaling: KernelScaling::Point, ..opts.clone() };
    let fill: Vec<D> = warp_to(transformer, &empty, 1, 1, &o);
    let mut out: Vec<D> = Vec::with_capacity(w * h * nb);
    for _ in 0..w * h {
        out.extend_from_slice(&fill);
    }
    let mut data: Vec<S> = Vec::new();
    for c in plan.live() {
        let [sw, sh] = c.src_size;
        let (rx, ry) = (c.src_off[0].checked_sub(src.col_off), c.src_off[1].checked_sub(src.row_off));
        let (Some(rx), Some(ry)) = (rx, ry) else { panic!("source buffer does not cover chunk {c:?}") };
        assert!(rx + sw <= src.ncol && ry + sh <= src.nrow, "source buffer does not cover chunk {c:?}");
        data.clear();
        for row in 0..sh {
            let start = ((ry + row) * src.ncol + rx) * nb;
            data.extend_from_slice(&src.data[start..start + sw * nb]);
        }
        let buf = SourceBuffer { data: &data, ncol: sw, nrow: sh, nbands: nb, col_off: c.src_off[0], row_off: c.src_off[1] };
        let [cw, ch] = c.dst_size;
        let tile: Vec<D> = warp_to(transformer, &buf, cw, ch, &c.warp_options(opts));
        let [dx, dy] = [c.dst_off[0] - plan.dst_off[0], c.dst_off[1] - plan.dst_off[1]];
        for (iy, row) in tile.chunks_exact(cw * nb).enumerate() {
            let at = ((dy + iy) * w + dx) * nb;
            out[at..at + cw * nb].copy_from_slice(row);
        }
    }
    out
}
