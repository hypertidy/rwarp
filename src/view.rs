//! Which chunks, at which level, does a view need?
//!
//! A viewer showing a raster in a CRS other than the raster's own decides,
//! every time the view changes, which source chunks to fetch and at which
//! resolution level. That is the warper's source-window question
//! ([`crate::source_window`], [`crate::plan`]) asked for a screen, with one
//! difference: a warp reads one level, and a screen wants a level per
//! region. [`plan_view`] answers it as a value, with no I/O and no
//! rendering:
//!
//! ```text
//! view (extent, screen size)  +  source (levels: geotransform, size, chunks)
//!       |                             |
//!       +------- CRS transforms ------+
//!                     |
//!                     v
//!             plan_view  ->  ViewPlan
//!                              chunks: (level, col, row), nearest the centre first
//!                              tiles:  per screen tile, the level it needs
//! ```
//!
//! The caller turns keys into byte ranges with whatever store it has (a
//! Zarr or TIFF reader, a byte-reference table) and draws however it draws.
//!
//! ## The rule
//!
//! 1. The screen is split into tiles ([`ViewOptions::tile`], 64 pixels).
//! 2. Each tile is sampled on a small grid ([`ViewOptions::samples`], 9 by
//!    9). Samples that do not transform to the source CRS are dropped, and
//!    with a transform back to the view so are samples that do not come
//!    back to where they started (past the horizon of a globe view).
//! 3. At each sample the *need* is the size of the source cell one screen
//!    pixel covers there: the square root of the Jacobian determinant of
//!    screen pixel to source world, in source world units. The tile's need
//!    is a low quantile of its samples' ([`ViewOptions::quantile`], the
//!    lowest quartile), so the finer part of a tile decides.
//! 4. The tile's level is the coarsest whose cell is no larger than its
//!    need times [`ViewOptions::quality`]: gdalwarp's `-ovr AUTO` rule,
//!    per tile.
//! 5. The tile's footprint is the box of its samples in source world
//!    coordinates, unwrapped about the tile's own middle where the source
//!    wraps ([`Source::period`]), and padded by [`ViewOptions::pad`] cells.
//!    A tile that holds one of the source's poles ([`Source::poles`])
//!    takes the full row of longitudes out to that pole.
//! 6. The chunks under each footprint are merged across tiles. With a
//!    budget ([`ViewOptions::max_chunks`], [`ViewOptions::max_cells`]) the
//!    outermost tiles go one level coarser, in rounds, until the plan fits.
//!    A coarser chunk whose whole area is already planned at one finer
//!    level is dropped ([`ViewOptions::drop_covered`]).
//!
//! One box and one level for the whole view, the rule the mesh viewers
//! use, is [`plan_single_box`], kept for comparison: for a polar view of a
//! global lon/lat pyramid it asks for over six times the cells, and for
//! a lon/lat view across 180 degrees over eight times (`tests/view.rs`,
//! with the cases as data in `tests/fixtures/view/cases.json`).
//!
//! ## What it does not do
//!
//! - Curvilinear and model grids: a level is a geotransform. A level with
//!   rotation terms is planned by the box of its footprint's corners.
//! - Other axes: a plan is for one 2D slice. Time or depth is the
//!   caller's leading index.
//! - Reduced copies: a source with one level gets one level. No planner
//!   makes a store without overviews cheap to look at from far away.
//!
//! ## Example
//!
//! ```
//! use rwarp::view::{plan_view, Level, Source, View, ViewOptions};
//! use rwarp::CrsTransform;
//!
//! // A lon/lat view of a lon/lat pyramid needs no projection.
//! struct Same;
//! impl CrsTransform for Same {
//!     fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
//!         Some((x, y))
//!     }
//! }
//!
//! // 0.01 degree global grid, 512 x 512 chunks, six overviews.
//! let base = Level::new([-180.0, 0.01, 0.0, 90.0, 0.0, -0.01], [36000, 18000], [512, 512]);
//! let source = Source::geographic(base.pyramid(7));
//!
//! // 40 degrees of longitude across 180, on a 1024 x 1024 screen.
//! let view = View::new([160.0, -30.0, 200.0, 10.0], [1024, 1024]);
//! let plan = plan_view(&view, &source, &Same, Some(&Same), &ViewOptions::default());
//!
//! assert_eq!(plan.levels_used(), vec![1]);
//! // 2 x 2 chunks either side of the seam, not the 180 of a single box.
//! assert!(plan.chunks.len() <= 25);
//! ```

use std::collections::{BTreeMap, BTreeSet};

use crate::crs::CrsTransform;
use vaster::inv_geotransform;

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// One resolution level of a source: a grid and its chunking.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Level {
    /// GDAL geotransform `[x0, dx, rx, y0, ry, dy]` of the level, in the
    /// source CRS.
    pub gt: [f64; 6],
    /// `[width, height]` in cells.
    pub size: [usize; 2],
    /// `[width, height]` of a chunk in cells. Edge chunks may be partial.
    pub chunk: [usize; 2],
}

impl Level {
    pub fn new(gt: [f64; 6], size: [usize; 2], chunk: [usize; 2]) -> Self {
        Self { gt, size, chunk }
    }

    /// The size of a cell in source world units: the square root of the
    /// geotransform's determinant.
    pub fn cell(&self) -> f64 {
        (self.gt[1] * self.gt[5] - self.gt[2] * self.gt[4]).abs().sqrt()
    }

    /// `[columns, rows]` of chunks.
    pub fn chunks(&self) -> [usize; 2] {
        [self.size[0].div_ceil(self.chunk[0].max(1)), self.size[1].div_ceil(self.chunk[1].max(1))]
    }

    /// The cells in one chunk (fewer at the right and bottom edges).
    pub fn chunk_cells(&self, col: usize, row: usize) -> usize {
        let w = self.chunk[0].min(self.size[0].saturating_sub(col * self.chunk[0]));
        let h = self.chunk[1].min(self.size[1].saturating_sub(row * self.chunk[1]));
        w * h
    }

    /// This level and `n - 1` more, each half the size of the one before
    /// (rounded up) with cells twice as large and the same chunk shape and
    /// origin: the usual overview pyramid, for sources that describe
    /// theirs by a count.
    pub fn pyramid(&self, n: usize) -> Vec<Level> {
        let mut out = Vec::with_capacity(n);
        let mut lev = *self;
        for _ in 0..n {
            out.push(lev);
            let mut gt = lev.gt;
            gt[1] *= 2.0;
            gt[2] *= 2.0;
            gt[4] *= 2.0;
            gt[5] *= 2.0;
            lev = Level { gt, size: [lev.size[0].div_ceil(2), lev.size[1].div_ceil(2)], chunk: lev.chunk };
        }
        out
    }
}

/// A source: its levels, in any order, and how its CRS wraps.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Source {
    /// The levels. A [`ChunkKey::level`] is an index into this.
    pub levels: Vec<Level>,
    /// The period of the source CRS's x in world units: 360 for a
    /// geographic CRS in degrees. `None` for a CRS that does not wrap.
    pub period: Option<f64>,
    /// The y of the source CRS's singular rows, where every x is the same
    /// place: 90 and -90 for a geographic CRS in degrees.
    pub poles: Vec<f64>,
}

impl Source {
    /// A source in a CRS that does not wrap (projected).
    pub fn new(levels: Vec<Level>) -> Self {
        Self { levels, period: None, poles: Vec::new() }
    }

    /// A source in a geographic CRS in degrees: x wraps every 360, and
    /// y = 90 and y = -90 are the poles.
    pub fn geographic(levels: Vec<Level>) -> Self {
        Self { levels, period: Some(360.0), poles: vec![90.0, -90.0] }
    }
}

#[cfg(any(feature = "proj", feature = "proj4rs"))]
impl Source {
    /// A source in the CRS named: [`Source::geographic`] if it is
    /// geographic in degrees
    /// ([`is_geographic_degrees`](crate::crs::is_geographic_degrees)),
    /// [`Source::new`] otherwise.
    pub fn for_crs(levels: Vec<Level>, crs: &str) -> Result<Self, String> {
        Ok(if crate::crs::is_geographic_degrees(crs)? { Self::geographic(levels) } else { Self::new(levels) })
    }
}

/// A view: an axis-aligned extent in the view's CRS, on a screen.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct View {
    /// `[xmin, ymin, xmax, ymax]` in the view CRS. Screen row 0 is at
    /// `ymax`.
    pub extent: [f64; 4],
    /// `[width, height]` in screen (device) pixels.
    pub size: [usize; 2],
}

impl View {
    pub fn new(extent: [f64; 4], size: [usize; 2]) -> Self {
        Self { extent, size }
    }

    /// The view coordinate of a (fractional) screen position, where
    /// `(0.5, 0.5)` is the centre of the top left pixel.
    #[inline]
    pub fn at(&self, px: f64, py: f64) -> (f64, f64) {
        let [x0, y0, x1, y1] = self.extent;
        (x0 + px / self.size[0] as f64 * (x1 - x0), y1 - py / self.size[1] as f64 * (y1 - y0))
    }

    /// The (fractional) screen position of a view coordinate.
    #[inline]
    pub fn screen(&self, x: f64, y: f64) -> (f64, f64) {
        let [x0, y0, x1, y1] = self.extent;
        ((x - x0) / (x1 - x0) * self.size[0] as f64, (y1 - y) / (y1 - y0) * self.size[1] as f64)
    }
}

/// The size of source cell a screen pixel needs, from the Jacobian of
/// screen pixel to source world at a sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Need {
    /// The square root of the determinant: the side of a square source
    /// cell with the area one screen pixel covers. Where the view
    /// stretches the source one way and squeezes it the other (round a
    /// pole of a lon/lat grid, at the limb of a globe) this is finer than
    /// needed one way and coarser the other.
    #[default]
    Area,
    /// The smaller singular value: the source cell the pixel needs along
    /// the direction it needs most. Never coarser than the screen in any
    /// direction, at the cost of more cells where the view is stretched.
    Shortest,
}

/// Options for [`plan_view`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ViewOptions {
    /// Screen tile size in pixels (default 64).
    pub tile: usize,
    /// Samples along each side of a tile (default 9, at least 2).
    pub samples: usize,
    /// Samples along each side of a tile at the source's edge on the
    /// screen: a tile only some of whose samples reach the source, or
    /// none beside one where some do. The limb of a globe is such an
    /// edge, and there the source moves fastest across the screen. 0 (the
    /// default) is a sample a pixel; any other number no more than
    /// `samples` turns this off.
    pub edge_samples: usize,
    /// How a sample's need is measured (default [`Need::Area`]).
    pub need: Need,
    /// Which quantile of a tile's samples sets its need, 0 to 1 (default
    /// 0.25: the finer part of the tile decides).
    pub quantile: f64,
    /// Multiplies the need before the level is chosen (default 1). Below
    /// 1 asks for finer levels than the screen can show, as a viewer does
    /// to have data ready for zooming in; above 1 for coarser.
    pub quality: f64,
    /// Cells of the tile's level added round its footprint (default 1,
    /// what a bilinear draw needs).
    pub pad: f64,
    /// How far, in screen pixels, a sample may come back from where it
    /// started and still count (default 1). Used when [`plan_view`] is
    /// given the transform back to the view.
    pub round_trip: f64,
    /// The most chunks the plan may hold.
    pub max_chunks: Option<usize>,
    /// The most cells the plan's chunks may hold together (bytes over
    /// bytes per cell, before compression).
    pub max_cells: Option<u64>,
    /// Drop a coarser chunk whose whole area is planned at one finer
    /// level (default true).
    pub drop_covered: bool,
}

impl Default for ViewOptions {
    fn default() -> Self {
        Self {
            tile: 64,
            samples: 9,
            edge_samples: 0,
            need: Need::Area,
            quantile: 0.25,
            quality: 1.0,
            pad: 1.0,
            round_trip: 1.0,
            max_chunks: None,
            max_cells: None,
            drop_covered: true,
        }
    }
}

impl ViewOptions {
    pub fn with_tile(mut self, tile: usize) -> Self {
        self.tile = tile;
        self
    }
    pub fn with_samples(mut self, samples: usize) -> Self {
        self.samples = samples;
        self
    }
    pub fn with_edge_samples(mut self, samples: usize) -> Self {
        self.edge_samples = samples;
        self
    }
    pub fn with_need(mut self, need: Need) -> Self {
        self.need = need;
        self
    }
    pub fn with_quantile(mut self, quantile: f64) -> Self {
        self.quantile = quantile;
        self
    }
    pub fn with_quality(mut self, quality: f64) -> Self {
        self.quality = quality;
        self
    }
    pub fn with_pad(mut self, pad: f64) -> Self {
        self.pad = pad;
        self
    }
    pub fn with_round_trip(mut self, pixels: f64) -> Self {
        self.round_trip = pixels;
        self
    }
    pub fn with_max_chunks(mut self, max: Option<usize>) -> Self {
        self.max_chunks = max;
        self
    }
    pub fn with_max_cells(mut self, max: Option<u64>) -> Self {
        self.max_cells = max;
        self
    }
    pub fn with_drop_covered(mut self, drop: bool) -> Self {
        self.drop_covered = drop;
        self
    }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

/// A chunk of a level: `level` indexes [`Source::levels`], `col` and `row`
/// count chunks from the level's origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChunkKey {
    pub level: usize,
    pub col: usize,
    pub row: usize,
}

/// A chunk the view needs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlannedChunk {
    pub key: ChunkKey,
    /// How many screen tiles ask for it.
    pub tiles: usize,
    /// Screen pixels it serves: each asking tile's visible pixels, shared
    /// equally among that tile's chunks.
    pub screen_px: f64,
    /// Screen pixels from the view's centre to the nearest asking tile's
    /// centre. [`ViewPlan::chunks`] is in this order.
    pub distance: f64,
}

/// What one screen tile needs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TilePlan {
    /// The tile's column and row on the screen.
    pub col: usize,
    pub row: usize,
    /// Samples that reached the source.
    pub valid: usize,
    /// Samples taken: [`ViewOptions::samples`] squared, or
    /// [`ViewOptions::edge_samples`] squared at the source's edge.
    pub samples: usize,
    /// The source cell one screen pixel needs here, in source world
    /// units. Infinite when no sample of the tile had a scale (its
    /// neighbours half a pixel away did not reach the source).
    pub need: f64,
    /// The level that need asks for.
    pub level: usize,
    /// The level planned for the tile: `level`, or a coarser one under a
    /// budget. The tile's area may also be served by finer chunks other
    /// tiles asked for ([`ViewOptions::drop_covered`]).
    pub planned: usize,
    /// The tile holds one of the source's poles.
    pub pole: bool,
}

/// The chunks a view needs: the result of [`plan_view`].
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ViewPlan {
    /// The chunks, nearest the view's centre first.
    pub chunks: Vec<PlannedChunk>,
    /// The screen tiles that see the source, by row then column.
    pub tiles: Vec<TilePlan>,
    /// The plan is over its budget with every tile at the coarsest level.
    pub over_budget: bool,
}

/// What changes from one plan to the next: see [`ViewPlan::diff`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlanDiff {
    /// In the new plan only, in the new plan's order.
    pub fetch: Vec<ChunkKey>,
    /// In both.
    pub keep: Vec<ChunkKey>,
    /// In the old plan only.
    pub drop: Vec<ChunkKey>,
}

impl ViewPlan {
    /// The chunk keys, in the plan's order.
    pub fn keys(&self) -> impl Iterator<Item = ChunkKey> + '_ {
        self.chunks.iter().map(|c| c.key)
    }

    /// The levels the plan reads from, ascending by index.
    pub fn levels_used(&self) -> Vec<usize> {
        self.keys().map(|k| k.level).collect::<BTreeSet<_>>().into_iter().collect()
    }

    /// The cells in the plan's chunks together. Times the bytes per cell,
    /// that is the uncompressed size of the read.
    pub fn cells(&self, source: &Source) -> u64 {
        self.keys().map(|k| source.levels[k.level].chunk_cells(k.col, k.row) as u64).sum()
    }

    /// What to fetch, keep and drop on going from `previous` to this
    /// plan, for panning and zooming.
    pub fn diff(&self, previous: &ViewPlan) -> PlanDiff {
        let old: BTreeSet<ChunkKey> = previous.keys().collect();
        let new: BTreeSet<ChunkKey> = self.keys().collect();
        PlanDiff {
            fetch: self.keys().filter(|k| !old.contains(k)).collect(),
            keep: self.keys().filter(|k| old.contains(k)).collect(),
            drop: previous.keys().filter(|k| !new.contains(k)).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// The planner
// ---------------------------------------------------------------------------

/// A level ready to plan with.
struct Lev {
    index: usize,
    level: Level,
    inv: [f64; 6],
    cell: f64,
    /// World x range of the grid's four corners.
    x_range: [f64; 2],
}

fn usable_levels(source: &Source) -> Vec<Lev> {
    let mut levs: Vec<Lev> = source
        .levels
        .iter()
        .enumerate()
        .filter_map(|(index, level)| {
            let inv = inv_geotransform(&level.gt)?;
            let cell = level.cell();
            if !(cell.is_finite() && cell > 0.0) || level.size.contains(&0) || level.chunk.contains(&0) {
                return None;
            }
            let (w, h) = (level.size[0] as f64, level.size[1] as f64);
            let xs = [0.0, w].map(|p| [0.0, h].map(|l| level.gt[0] + p * level.gt[1] + l * level.gt[2]));
            let xs = [xs[0][0], xs[0][1], xs[1][0], xs[1][1]];
            let x_range =
                [xs.iter().copied().fold(f64::INFINITY, f64::min), xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)];
            Some(Lev { index, level: *level, inv, cell, x_range })
        })
        .collect();
    // Finest first; ties keep the source's order.
    levs.sort_by(|a, b| a.cell.total_cmp(&b.cell).then(a.index.cmp(&b.index)));
    levs
}

/// The position, in `levs` (finest first), of the coarsest level whose
/// cell is no larger than `need`; the finest when none is.
fn pick(levs: &[Lev], need: f64) -> usize {
    let mut at = 0;
    for (i, l) in levs.iter().enumerate() {
        if l.cell <= need * (1.0 + 1e-6) {
            at = i;
        }
    }
    at
}

/// `x` moved by whole periods to within half a period of `about`.
#[inline]
fn unwrap_about(x: f64, about: f64, period: f64) -> f64 {
    about + (x - about + period / 2.0).rem_euclid(period) - period / 2.0
}

/// numpy's default (linear) quantile of a sorted slice.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// A footprint in source world coordinates.
#[derive(Debug, Clone, Copy)]
struct Foot {
    x: [f64; 2],
    y: [f64; 2],
    /// Every longitude: the tile holds a pole, or its samples span more
    /// than half the period.
    all_x: bool,
}

/// The chunks of `lev` under `foot`, padded by `pad` cells.
fn chunks_under(lev: &Lev, foot: &Foot, period: Option<f64>, pad: f64, out: &mut BTreeSet<ChunkKey>) {
    let p = pad * lev.cell;
    let (ymin, ymax) = (foot.y[0] - p, foot.y[1] + p);
    let (mut xmin, mut xmax) = (foot.x[0] - p, foot.x[1] + p);
    let mut shifts = vec![0.0];
    if let Some(per) = period {
        if foot.all_x || xmax - xmin >= per {
            // One whole period, placed on the grid.
            xmin = lev.x_range[0];
            xmax = xmin + per;
        }
        let k0 = ((lev.x_range[0] - xmax) / per).ceil() as i64;
        let k1 = ((lev.x_range[1] - xmin) / per).floor() as i64;
        shifts = (k0..=k1).map(|k| k as f64 * per).collect();
    }
    let (w, h) = (lev.level.size[0] as f64, lev.level.size[1] as f64);
    let [cw, ch] = lev.level.chunk;
    for s in shifts {
        let (mut c0, mut c1, mut r0, mut r1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for x in [xmin + s, xmax + s] {
            for y in [ymin, ymax] {
                let c = lev.inv[0] + x * lev.inv[1] + y * lev.inv[2];
                let r = lev.inv[3] + x * lev.inv[4] + y * lev.inv[5];
                c0 = c0.min(c);
                c1 = c1.max(c);
                r0 = r0.min(r);
                r1 = r1.max(r);
            }
        }
        let (c0, c1, r0, r1) = (c0.max(0.0), c1.min(w), r0.max(0.0), r1.min(h));
        if !(c1 > c0 && r1 > r0) {
            continue;
        }
        let last = |v: f64, n: usize| ((v.ceil() as usize).max(1) - 1) / n;
        for row in (r0.floor() as usize) / ch..=last(r1, ch) {
            for col in (c0.floor() as usize) / cw..=last(c1, cw) {
                out.insert(ChunkKey { level: lev.index, col, row });
            }
        }
    }
}

struct Tile {
    plan: TilePlan,
    foot: Foot,
    /// Position in the finest-first level list.
    at: usize,
    distance: f64,
    keys: BTreeSet<ChunkKey>,
}

/// Plan the chunks a view needs.
///
/// `to_source` maps a view CRS coordinate to a source CRS coordinate.
/// `to_view` is its inverse; with it, samples that do not come back to
/// where they started are dropped (the far side of a globe), and the
/// source's poles are placed on the screen exactly. Without it a tile is
/// taken to hold a pole when its samples span more than half the period.
///
/// A view that sees none of the source, or a source with no usable level,
/// gives an empty plan.
pub fn plan_view(
    view: &View,
    source: &Source,
    to_source: &dyn CrsTransform,
    to_view: Option<&dyn CrsTransform>,
    opts: &ViewOptions,
) -> ViewPlan {
    let levs = usable_levels(source);
    let (sw, sh) = (view.size[0], view.size[1]);
    if levs.is_empty() || sw == 0 || sh == 0 {
        return ViewPlan::default();
    }
    let tile = opts.tile.max(1);
    let ns = opts.samples.max(2);
    let period = source.period.filter(|p| p.is_finite() && *p > 0.0);
    let (tcols, trows) = (sw.div_ceil(tile), sh.div_ceil(tile));

    // The source point under a screen position, if it is really there.
    let reach = |px: f64, py: f64, check: bool| -> Option<(f64, f64)> {
        let (vx, vy) = view.at(px, py);
        let (x, y) = to_source.convert(vx, vy)?;
        if !(x.is_finite() && y.is_finite()) {
            return None;
        }
        if let (true, Some(back)) = (check, to_view) {
            let (bx, by) = back.convert(x, y)?;
            let (qx, qy) = view.screen(bx, by);
            if !((qx - px).abs() <= opts.round_trip && (qy - py).abs() <= opts.round_trip) {
                return None;
            }
        }
        Some((x, y))
    };

    // Which tile holds each pole.
    let mut pole_tiles: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
    if let Some(back) = to_view {
        let x_mid = (levs[0].x_range[0] + levs[0].x_range[1]) / 2.0;
        for &py in &source.poles {
            // Every x is the same place at a pole, but not every backend
            // says so at the pole itself: try a hair short of it too.
            let mid = levs[0].level.gt[3] + levs[0].level.size[1] as f64 / 2.0 * levs[0].level.gt[5];
            let short = py - (py - mid).signum() * py.abs().max(1.0) * 1e-9;
            let hit = [py, short].iter().find_map(|&y| {
                let (vx, vy) = back.convert(x_mid, y)?;
                let (qx, qy) = view.screen(vx, vy);
                (qx.is_finite() && qy.is_finite() && qx >= 0.0 && qy >= 0.0 && qx < sw as f64 && qy < sh as f64)
                    .then_some((qx, qy))
            });
            if let Some((qx, qy)) = hit {
                // The view must see the pole, not a far-side copy of it.
                let seen = reach(qx, qy, false).is_some_and(|(_, y)| (y - py).abs() <= (py - short).abs() * 1e3 + 1e-6);
                if seen {
                    pole_tiles.entry((qx as usize / tile, qy as usize / tile)).or_default().push(py);
                }
            }
        }
    }

    // A tile's samples on an `n` by `n` grid of its pixel centres, first
    // pixel to last: where each lands in the source, and what it needs.
    let sample = |tcol: usize, trow: usize, n: usize, xs: &mut Vec<f64>, ys: &mut Vec<f64>, needs: Option<&mut Vec<f64>>| {
        let (px0, py0) = ((tcol * tile) as f64, (trow * tile) as f64);
        let tw = (tile.min(sw - tcol * tile)) as f64;
        let th = (tile.min(sh - trow * tile)) as f64;
        xs.clear();
        ys.clear();
        let mut needs = needs;
        if let Some(v) = needs.as_deref_mut() {
            v.clear();
        }
        for j in 0..n {
            for i in 0..n {
                let px = px0 + 0.5 + (tw - 1.0) * i as f64 / (n - 1) as f64;
                let py = py0 + 0.5 + (th - 1.0) * j as f64 / (n - 1) as f64;
                let Some((x, y)) = reach(px, py, true) else { continue };
                xs.push(x);
                ys.push(y);
                let Some(needs) = needs.as_deref_mut() else { continue };
                // Half a pixel each way: the Jacobian of screen to source.
                if let (Some((ax, ay)), Some((bx, by))) = (reach(px + 0.5, py, false), reach(px, py + 0.5, false)) {
                    let (mut a, mut b) = (2.0 * (ax - x), 2.0 * (bx - x));
                    let (c, d) = (2.0 * (ay - y), 2.0 * (by - y));
                    if let Some(per) = period {
                        a = 2.0 * unwrap_about(ax - x, 0.0, per);
                        b = 2.0 * unwrap_about(bx - x, 0.0, per);
                    }
                    let det = (a * d - b * c).abs();
                    let need = match opts.need {
                        Need::Area => det.sqrt(),
                        Need::Shortest => {
                            // The smaller singular value.
                            let q = a * a + b * b + c * c + d * d;
                            ((q - (q * q - 4.0 * det * det).max(0.0).sqrt()) / 2.0).max(0.0).sqrt()
                        }
                    };
                    if need.is_finite() && need > 0.0 {
                        needs.push(need);
                    }
                }
            }
        }
    };

    let centre = (sw as f64 / 2.0, sh as f64 / 2.0);
    let mut tiles: Vec<Tile> = Vec::new();
    // Every tile's samples. A tile that is partly seen, or unseen beside
    // one that is seen, is at the source's edge on the screen (the limb of
    // a globe), where the samples' box misses what lies between them and
    // the edge.
    let mut sampled: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::with_capacity(tcols * trows);
    for trow in 0..trows {
        for tcol in 0..tcols {
            let mut s = (Vec::new(), Vec::new(), Vec::new());
            sample(tcol, trow, ns, &mut s.0, &mut s.1, Some(&mut s.2));
            sampled.push(s);
        }
    }
    let seen: Vec<usize> = sampled.iter().map(|s| s.0.len()).collect();
    let at_edge = |tcol: usize, trow: usize| {
        let n = seen[trow * tcols + tcol];
        if n == ns * ns {
            return false;
        }
        n > 0
            || (trow.saturating_sub(1)..=(trow + 1).min(trows - 1))
                .any(|r| (tcol.saturating_sub(1)..=(tcol + 1).min(tcols - 1)).any(|c| seen[r * tcols + c] > 0))
    };
    let edge = if opts.edge_samples == 0 { tile } else { opts.edge_samples };
    for trow in 0..trows {
        for tcol in 0..tcols {
            let (px0, py0) = ((tcol * tile) as f64, (trow * tile) as f64);
            let tw = (tile.min(sw - tcol * tile)) as f64;
            let th = (tile.min(sh - trow * tile)) as f64;
            let (mut xs, mut ys, mut needs) = std::mem::take(&mut sampled[trow * tcols + tcol]);
            let mut samples = ns * ns;
            if edge > ns && at_edge(tcol, trow) {
                // The footprint from the finer grid; the need from the
                // coarser one when it has any.
                if needs.is_empty() {
                    sample(tcol, trow, edge, &mut xs, &mut ys, Some(&mut needs));
                } else {
                    sample(tcol, trow, edge, &mut xs, &mut ys, None);
                }
                samples = edge * edge;
            }
            let valid = xs.len();
            let poles = pole_tiles.get(&(tcol, trow));
            if xs.is_empty() && poles.is_none() {
                continue;
            }
            let mut foot = Foot { x: [f64::INFINITY, f64::NEG_INFINITY], y: [f64::INFINITY, f64::NEG_INFINITY], all_x: false };
            if !xs.is_empty() {
                if let Some(per) = period {
                    let mut sorted = xs.clone();
                    sorted.sort_by(f64::total_cmp);
                    let about = quantile(&sorted, 0.5);
                    for x in xs.iter_mut() {
                        *x = unwrap_about(*x, about, per);
                    }
                }
                for (&x, &y) in xs.iter().zip(&ys) {
                    foot.x = [foot.x[0].min(x), foot.x[1].max(x)];
                    foot.y = [foot.y[0].min(y), foot.y[1].max(y)];
                }
                if let Some(per) = period {
                    foot.all_x = foot.x[1] - foot.x[0] > per / 2.0;
                }
            }
            if let Some(poles) = poles {
                foot.all_x = true;
                for &p in poles {
                    foot.y = [foot.y[0].min(p), foot.y[1].max(p)];
                }
                if xs.is_empty() {
                    foot.x = [levs[0].x_range[0], levs[0].x_range[0]];
                }
            }
            needs.sort_by(f64::total_cmp);
            // A tile with places but no scale (every neighbour failed)
            // takes the coarsest level.
            let need = if needs.is_empty() { f64::INFINITY } else { quantile(&needs, opts.quantile) };
            let at = pick(&levs, need * opts.quality);
            let (cx, cy) = (px0 + tw / 2.0 - centre.0, py0 + th / 2.0 - centre.1);
            tiles.push(Tile {
                plan: TilePlan {
                    col: tcol,
                    row: trow,
                    valid,
                    samples,
                    need,
                    level: levs[at].index,
                    planned: levs[at].index,
                    pole: poles.is_some(),
                },
                foot,
                at,
                distance: (cx * cx + cy * cy).sqrt(),
                keys: BTreeSet::new(),
            });
        }
    }

    for t in tiles.iter_mut() {
        chunks_under(&levs[t.at], &t.foot, period, opts.pad, &mut t.keys);
    }

    // Budget: the outermost tiles go coarser, one level a round.
    let mut over_budget = false;
    if opts.max_chunks.is_some() || opts.max_cells.is_some() {
        let mut count: BTreeMap<ChunkKey, usize> = BTreeMap::new();
        let mut cells: u64 = 0;
        let cells_of = |k: &ChunkKey| source.levels[k.level].chunk_cells(k.col, k.row) as u64;
        for t in &tiles {
            for k in &t.keys {
                let n = count.entry(*k).or_insert(0);
                if *n == 0 {
                    cells += cells_of(k);
                }
                *n += 1;
            }
        }
        let fits = |n: usize, cells: u64| {
            opts.max_chunks.is_none_or(|m| n <= m) && opts.max_cells.is_none_or(|m| cells <= m)
        };
        let mut order: Vec<usize> = (0..tiles.len()).collect();
        order.sort_by(|&a, &b| tiles[b].distance.total_cmp(&tiles[a].distance));
        'rounds: while !fits(count.len(), cells) {
            let mut moved = false;
            for &i in &order {
                let t = &mut tiles[i];
                if t.at + 1 >= levs.len() {
                    continue;
                }
                moved = true;
                for k in &t.keys {
                    let n = count.get_mut(k).expect("counted");
                    *n -= 1;
                    if *n == 0 {
                        count.remove(k);
                        cells -= cells_of(k);
                    }
                }
                t.at += 1;
                t.plan.planned = levs[t.at].index;
                t.keys.clear();
                chunks_under(&levs[t.at], &t.foot, period, opts.pad, &mut t.keys);
                for k in &t.keys {
                    let n = count.entry(*k).or_insert(0);
                    if *n == 0 {
                        cells += cells_of(k);
                    }
                    *n += 1;
                }
                if fits(count.len(), cells) {
                    break 'rounds;
                }
            }
            if !moved {
                over_budget = true;
                break;
            }
        }
    }

    // Merge.
    let mut merged: BTreeMap<ChunkKey, PlannedChunk> = BTreeMap::new();
    for t in &tiles {
        if t.keys.is_empty() {
            continue;
        }
        let tw = tile.min(sw - t.plan.col * tile) as f64;
        let th = tile.min(sh - t.plan.row * tile) as f64;
        let share = tw * th * t.plan.valid as f64 / t.plan.samples as f64 / t.keys.len() as f64;
        for k in &t.keys {
            let c = merged.entry(*k).or_insert(PlannedChunk { key: *k, tiles: 0, screen_px: 0.0, distance: f64::INFINITY });
            c.tiles += 1;
            c.screen_px += share;
            c.distance = c.distance.min(t.distance);
        }
    }

    if opts.drop_covered {
        let have: BTreeSet<ChunkKey> = merged.keys().copied().collect();
        let used: BTreeSet<usize> = have.iter().map(|k| k.level).collect();
        let by_index = |i: usize| levs.iter().position(|l| l.index == i).expect("a planned level");
        for k in &have {
            let at = by_index(k.level);
            let covered = used.iter().any(|&f| {
                let fat = by_index(f);
                fat < at && levs[fat].cell < levs[at].cell && covered_by(&levs[at], k, &levs[fat], &have)
            });
            if covered {
                merged.remove(k);
            }
        }
    }

    let mut chunks: Vec<PlannedChunk> = merged.into_values().collect();
    chunks.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.key.cmp(&b.key)));
    ViewPlan { chunks, tiles: tiles.into_iter().map(|t| t.plan).collect(), over_budget }
}

/// Is the whole of chunk `k` of `coarse` inside planned chunks of `fine`?
/// Only for grids without rotation in the same frame; anything else is
/// not covered.
fn covered_by(coarse: &Lev, k: &ChunkKey, fine: &Lev, have: &BTreeSet<ChunkKey>) -> bool {
    let straight = |l: &Lev| l.level.gt[2] == 0.0 && l.level.gt[4] == 0.0;
    if !(straight(coarse) && straight(fine)) {
        return false;
    }
    let (cl, fl) = (&coarse.level, &fine.level);
    // The chunk's cells, in the coarse grid, as a world box.
    let c0 = (k.col * cl.chunk[0]) as f64;
    let c1 = ((k.col + 1) * cl.chunk[0]).min(cl.size[0]) as f64;
    let r0 = (k.row * cl.chunk[1]) as f64;
    let r1 = ((k.row + 1) * cl.chunk[1]).min(cl.size[1]) as f64;
    let wx = [cl.gt[0] + c0 * cl.gt[1], cl.gt[0] + c1 * cl.gt[1]];
    let wy = [cl.gt[3] + r0 * cl.gt[5], cl.gt[3] + r1 * cl.gt[5]];
    // The same box in the fine grid's cells.
    let fc = wx.map(|x| fine.inv[0] + x * fine.inv[1]);
    let fr = wy.map(|y| fine.inv[3] + y * fine.inv[5]);
    let (fc0, fc1) = (fc[0].min(fc[1]), fc[0].max(fc[1]));
    let (fr0, fr1) = (fr[0].min(fr[1]), fr[0].max(fr[1]));
    // A coarse level's last cell may hang over the fine grid's edge by
    // less than a coarse cell (sizes round up): that much is allowed.
    let over = coarse.cell / fine.cell;
    let (w, h) = (fl.size[0] as f64, fl.size[1] as f64);
    if fc0 < -1e-9 || fr0 < -1e-9 || fc1 > w + over || fr1 > h + over {
        return false;
    }
    let (fc0, fc1, fr0, fr1) = (fc0.max(0.0), fc1.min(w), fr0.max(0.0), fr1.min(h));
    if !(fc1 > fc0 && fr1 > fr0) {
        return false;
    }
    let first = |v: f64, n: usize| ((v + 1e-9).floor() as usize) / n;
    let last = |v: f64, n: usize| (((v - 1e-9).ceil() as usize).max(1) - 1) / n;
    for row in first(fr0, fl.chunk[1])..=last(fr1, fl.chunk[1]) {
        for col in first(fc0, fl.chunk[0])..=last(fc1, fl.chunk[0]) {
            if !have.contains(&ChunkKey { level: fine.index, col, row }) {
                return false;
            }
        }
    }
    true
}

/// One box and one level for the whole view: the rule the mesh viewers
/// use, for comparison with [`plan_view`].
///
/// The view is sampled on a `samples` by `samples` grid; the box is the
/// extent of the samples in the source CRS (longitudes as the transform
/// returns them, wrapped into the source grid's own period), and the level
/// is the coarsest whose cell is no larger than the smaller of the box's
/// width and height per screen pixel shown, times `quality` (the viewers
/// use 0.7). The plan has one tile, `(0, 0)`.
pub fn plan_single_box(
    view: &View,
    source: &Source,
    to_source: &dyn CrsTransform,
    to_view: Option<&dyn CrsTransform>,
    samples: usize,
    quality: f64,
) -> ViewPlan {
    let levs = usable_levels(source);
    let (sw, sh) = (view.size[0] as f64, view.size[1] as f64);
    if levs.is_empty() || sw == 0.0 || sh == 0.0 {
        return ViewPlan::default();
    }
    let ns = samples.max(2);
    let period = source.period.filter(|p| p.is_finite() && *p > 0.0);
    let mut foot = Foot { x: [f64::INFINITY, f64::NEG_INFINITY], y: [f64::INFINITY, f64::NEG_INFINITY], all_x: false };
    let (mut p0, mut p1) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    let mut valid = 0;
    for j in 0..ns {
        for i in 0..ns {
            let px = 0.5 + (sw - 1.0) * i as f64 / (ns - 1) as f64;
            let py = 0.5 + (sh - 1.0) * j as f64 / (ns - 1) as f64;
            let (vx, vy) = view.at(px, py);
            let Some((mut x, y)) = to_source.convert(vx, vy) else { continue };
            if !(x.is_finite() && y.is_finite()) {
                continue;
            }
            if let Some(back) = to_view {
                let Some((bx, by)) = back.convert(x, y) else { continue };
                let (qx, qy) = view.screen(bx, by);
                if !((qx - px).abs() <= 1.0 && (qy - py).abs() <= 1.0) {
                    continue;
                }
            }
            if let Some(per) = period {
                x = levs[0].x_range[0] + (x - levs[0].x_range[0]).rem_euclid(per);
            }
            valid += 1;
            foot.x = [foot.x[0].min(x), foot.x[1].max(x)];
            foot.y = [foot.y[0].min(y), foot.y[1].max(y)];
            p0 = [p0[0].min(px), p0[1].min(py)];
            p1 = [p1[0].max(px), p1[1].max(py)];
        }
    }
    if valid == 0 {
        return ViewPlan::default();
    }
    let need = ((foot.x[1] - foot.x[0]) / (p1[0] - p0[0] + 1.0)).min((foot.y[1] - foot.y[0]) / (p1[1] - p0[1] + 1.0));
    let at = pick(&levs, need * quality);
    let mut keys = BTreeSet::new();
    // The box is in the grid's own period already: no shifting.
    chunks_under(&levs[at], &foot, None, 1.0, &mut keys);
    let n = keys.len().max(1) as f64;
    let chunks = keys
        .into_iter()
        .map(|key| PlannedChunk { key, tiles: 1, screen_px: sw * sh * valid as f64 / (ns * ns) as f64 / n, distance: 0.0 })
        .collect();
    let tile = TilePlan { col: 0, row: 0, valid, samples: ns * ns, need, level: levs[at].index, planned: levs[at].index, pole: false };
    ViewPlan { chunks, tiles: vec![tile], over_budget: false }
}

/// [`plan_view`] between two CRSs by name, with the compiled-in backend
/// ([`crate::crs::crs_pair`]).
#[cfg(any(feature = "proj", feature = "proj4rs"))]
pub fn plan_view_crs(
    view: &View,
    view_crs: &str,
    source: &Source,
    source_crs: &str,
    opts: &ViewOptions,
) -> Result<ViewPlan, String> {
    let (to_source, to_view) = crate::crs::crs_pair(source_crs, view_crs)?;
    Ok(plan_view(view, source, &to_source, Some(&to_view), opts))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Same;
    impl CrsTransform for Same {
        fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
            Some((x, y))
        }
    }

    fn mur() -> Source {
        Source::geographic(Level::new([-180.0, 0.01, 0.0, 90.0, 0.0, -0.01], [36000, 18000], [512, 512]).pyramid(7))
    }

    #[test]
    fn pyramid_halves() {
        let s = mur();
        assert_eq!(s.levels[6].size, [563, 282]);
        assert!((s.levels[6].cell() - 0.64).abs() < 1e-12);
        assert_eq!(s.levels[0].chunks(), [71, 36]);
        assert_eq!(s.levels[0].chunk_cells(70, 35), (36000 - 70 * 512) * (18000 - 35 * 512));
    }

    #[test]
    fn quantile_is_numpys() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile(&v, 0.25), 2.0);
        assert_eq!(quantile(&v, 0.5), 3.0);
        assert_eq!(quantile(&[1.0, 2.0], 0.25), 1.25);
    }

    #[test]
    fn same_scale_picks_the_matching_level() {
        // One screen pixel per 0.04 degree: level 2.
        let view = View::new([0.0, 0.0, 40.96, 40.96], [1024, 1024]);
        let plan = plan_view(&view, &mur(), &Same, Some(&Same), &ViewOptions::default());
        assert_eq!(plan.levels_used(), vec![2]);
        assert_eq!(plan.tiles.len(), 256);
        assert!(plan.tiles.iter().all(|t| t.valid == 81 && (t.need - 0.04).abs() < 1e-9));
        // 1024 cells of level 2, plus the pad: 3 x 3 chunks.
        assert_eq!(plan.chunks.len(), 9);
    }

    #[test]
    fn seam_view_stays_small() {
        let view = View::new([160.0, -30.0, 200.0, 10.0], [1024, 1024]);
        let src = mur();
        let plan = plan_view(&view, &src, &Same, Some(&Same), &ViewOptions::default());
        let single = plan_single_box(&view, &src, &Same, Some(&Same), 64, 0.7);
        assert_eq!(plan.levels_used(), vec![1]);
        assert!(plan.chunks.len() <= 25, "{}", plan.chunks.len());
        assert!(single.chunks.len() >= 6 * plan.chunks.len(), "{} vs {}", single.chunks.len(), plan.chunks.len());
        // Both sides of the seam are there.
        let cols: BTreeSet<usize> = plan.keys().map(|k| k.col).collect();
        assert!(cols.contains(&0) && cols.contains(&(src.levels[1].chunks()[0] - 1)));
    }

    #[test]
    fn a_grid_from_0_to_360_is_found() {
        let src = Source::geographic(Level::new([0.0, 0.25, 0.0, 90.0, 0.0, -0.25], [1440, 720], [100, 100]).pyramid(1));
        let view = View::new([-20.0, -10.0, 20.0, 10.0], [512, 256]);
        let plan = plan_view(&view, &src, &Same, Some(&Same), &ViewOptions::default());
        let cols: BTreeSet<usize> = plan.keys().map(|k| k.col).collect();
        assert!(cols.contains(&0) && cols.contains(&14), "{cols:?}");
        assert!(!cols.contains(&7));
    }

    #[test]
    fn budget_moves_the_rim_coarser() {
        let view = View::new([0.0, 0.0, 81.92, 81.92], [1024, 1024]);
        let src = mur();
        // Finer than needed, as a viewer asks to be ready to zoom.
        let free = plan_view(&view, &src, &Same, Some(&Same), &ViewOptions::default().with_quality(0.25));
        let opts = ViewOptions::default().with_quality(0.25).with_max_chunks(Some(free.chunks.len() / 2));
        let tight = plan_view(&view, &src, &Same, Some(&Same), &opts);
        assert!(tight.chunks.len() <= free.chunks.len() / 2);
        assert!(!tight.over_budget);
        assert!(tight.levels_used().len() > 1);
        // The centre keeps its level; the corner gave way.
        let at = |p: &ViewPlan, c: usize, r: usize| *p.tiles.iter().find(|t| t.col == c && t.row == r).unwrap();
        assert_eq!(at(&tight, 8, 8).planned, at(&tight, 8, 8).level);
        assert_ne!(at(&tight, 0, 0).planned, at(&tight, 0, 0).level);
        // A budget nothing can meet says so.
        let none = plan_view(&view, &src, &Same, Some(&Same), &ViewOptions::default().with_max_chunks(Some(0)));
        assert!(none.over_budget);
    }

    #[test]
    fn diff_for_a_pan() {
        let src = mur();
        let a = plan_view(&View::new([0.0, 0.0, 40.96, 40.96], [1024, 1024]), &src, &Same, None, &ViewOptions::default());
        let b = plan_view(&View::new([20.0, 0.0, 60.96, 40.96], [1024, 1024]), &src, &Same, None, &ViewOptions::default());
        let d = b.diff(&a);
        assert!(!d.fetch.is_empty() && !d.keep.is_empty() && !d.drop.is_empty());
        assert_eq!(d.fetch.len() + d.keep.len(), b.chunks.len());
        assert_eq!(d.drop.len() + d.keep.len(), a.chunks.len());
        assert!(b.diff(&b).fetch.is_empty());
    }

    #[test]
    fn nothing_seen_is_an_empty_plan() {
        struct Never;
        impl CrsTransform for Never {
            fn convert(&self, _: f64, _: f64) -> Option<(f64, f64)> {
                None
            }
        }
        let plan = plan_view(&View::new([0.0, 0.0, 1.0, 1.0], [256, 256]), &mur(), &Never, None, &ViewOptions::default());
        assert!(plan.chunks.is_empty() && plan.tiles.is_empty());
        // Outside the grid: tiles, but no chunks.
        let src = Source::new(vec![Level::new([0.0, 1.0, 0.0, 100.0, 0.0, -1.0], [100, 100], [10, 10])]);
        let plan = plan_view(&View::new([500.0, 500.0, 600.0, 600.0], [256, 256]), &src, &Same, None, &ViewOptions::default());
        assert!(plan.chunks.is_empty() && !plan.tiles.is_empty());
    }

    #[test]
    fn covered_coarse_chunks_are_dropped() {
        // The right half of the view shows the left half's ground again and
        // more, mirrored, at a quarter of the scale.
        struct Bent;
        impl CrsTransform for Bent {
            fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
                Some((if x < 0.0 { x } else { -4.0 * x }, y))
            }
        }
        let src = Source::new(Level::new([-1024.0, 1.0, 0.0, 512.0, 0.0, -1.0], [4096, 1024], [256, 256]).pyramid(3));
        let view = View::new([-512.0, -256.0, 512.0, 256.0], [1024, 512]);
        let keep = plan_view(&view, &src, &Bent, None, &ViewOptions::default().with_drop_covered(false));
        let drop = plan_view(&view, &src, &Bent, None, &ViewOptions::default());
        assert!(keep.levels_used().len() > 1);
        assert!(drop.chunks.len() < keep.chunks.len(), "{} vs {}", drop.chunks.len(), keep.chunks.len());
        let have: BTreeSet<ChunkKey> = drop.keys().collect();
        assert!(keep.keys().filter(|k| k.level == 0).all(|k| have.contains(&k)));
    }
}
