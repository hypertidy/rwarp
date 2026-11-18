//! A [`QuadMesh`] as a warp target: [`MeshTarget`].

use super::QuadMesh;
use crate::transform::{GenImgProjTransformer, SourceCells, SourceSheet, SourceWrap, Transformer};

/// The destination geotransform of a [`MeshTarget`]'s inner transformer:
/// its destination pixel/line are the mesh's world coordinates.
pub const WORLD_GT: [f64; 6] = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// A target whose pixels are the cells of a [`QuadMesh`], as a
/// [`Transformer`] to any source: for warping onto a curvilinear grid
/// given by its corners, such as an ocean model's, and for conservative
/// weights onto it.
///
/// Target pixel `(col, row)` is mesh cell `(row, col)`: a target buffer
/// of `ncol` by `nrow` pixels ([`MeshTarget::target_size`]) holds one
/// value per cell, row-major over the mesh. A target pixel/line goes to
/// the mesh's world coordinates ([`QuadMesh::to_world`]) and then through
/// `inner`, a transformer from those coordinates to the source: one whose
/// destination geotransform is [`WORLD_GT`].
///
/// - **Area methods.** [`exact_weights`](crate::weights::exact_weights)
///   with `dst_ncol, dst_nrow` the mesh's size, and
///   [`warp`](crate::warp::warp) with
///   [`Footprint::Exact`](crate::warp::Footprint::Exact), measure each
///   cell as a target pixel. Its outline is the cell itself: each side is
///   straight in the mesh's world coordinates, in `densify` pieces whose
///   end points go to the source exactly. So `frac` is the cell's share of
///   each source pixel, under the weights' measure in source pixel space
///   ([`lonlat_measure`](crate::weights::lonlat_measure) for areas on the
///   sphere from a source in longitude and latitude), and `dst_area` is
///   the cell's own measure there. Each cell is measured on its own, so
///   the mesh's seam, fold and period don't matter here.
/// - **The source** is the inner transformer's: a grid
///   ([`MeshTarget::from_grid`], whose longitudes wrap at the source's
///   seam and round a pole as GDAL's do), geolocation arrays, or another
///   mesh ([`MeshTransformer`](super::MeshTransformer) with destination
///   geotransform [`WORLD_GT`]), whose cells the outlines are walked onto
///   ([`Transformer::source_cells`]).
/// - **Point kernels** sample the source at each cell's centre, the world
///   position of the target pixel's centre under the mesh's
///   [`CellMap`](super::CellMap). Source to target goes through
///   [`QuadMesh::to_index`].
///
/// ```
/// use rwarp::mesh::{CellMap, MeshTarget, QuadMesh, WORLD_GT};
/// use rwarp::transform::GenImgProjTransformer;
/// use rwarp::weights::{exact_weights, GridWindow, Normalization, WeightOptions};
/// use rwarp::CrsTransform;
///
/// // a full annulus with a seam on the x axis, in metres
/// let (ncol, nrow) = (24, 4);
/// let (mut x, mut y) = (Vec::new(), Vec::new());
/// for j in 0..=nrow {
///     for i in 0..=ncol {
///         let a = ((i % ncol) as f64 / ncol as f64) * std::f64::consts::TAU;
///         let r = 100.0 + 20.0 * j as f64;
///         x.push(r * a.cos());
///         y.push(r * a.sin());
///     }
/// }
/// let mesh = QuadMesh::from_corners(ncol, nrow, x, y, CellMap::EqualArea).unwrap();
///
/// // the source: 20 m pixels, north up, in the mesh's own coordinates
/// struct Same;
/// impl CrsTransform for Same {
///     fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> { Some((x, y)) }
/// }
/// let gt = [-200.0, 20.0, 0.0, 200.0, 0.0, -20.0];
/// let inner = GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same)).unwrap();
/// let t = MeshTarget::new(mesh, inner);
/// let [w, h] = t.target_size();
/// let wts = exact_weights(&t, &GridWindow::new(20, 20), w, h, &WeightOptions::new(1));
///
/// // each cell covers its own area, 400 m^2 a source pixel
/// let covered = wts.dst_covered();
/// for (k, c) in covered.iter().enumerate() {
///     let area = t.target_mesh().cell_area(k / w, k % w);
///     assert!((c * 400.0 - area).abs() < 1e-9 * area, "cell {k}");
/// }
/// // a constant field stays constant
/// let s = wts.normalised(Normalization::DestArea);
/// let mut sum = vec![0.0; w * h];
/// for i in 0..wts.len() {
///     sum[wts.dst_row[i] as usize * w + wts.dst_col[i] as usize] += s[i];
/// }
/// assert!(sum.iter().all(|&v| (v - 1.0).abs() < 1e-12));
/// ```
pub struct MeshTarget<T> {
    mesh: QuadMesh,
    inner: T,
}

impl MeshTarget<GenImgProjTransformer> {
    /// From the grid `src_gt`, `src_size` pixels (`[width, height]`) in
    /// `src_crs`, onto the mesh in `mesh_crs`, with the compiled-in CRS
    /// backend. As gdalwarp does by default, a source in longitude and
    /// latitude has its longitudes wrapped round its centre (GDAL's
    /// `CENTER_LONG`), and an outline across its seam or round a pole is
    /// unwrapped and closed along the pole's row
    /// ([`Transformer::source_wrap`]). Fails if a CRS isn't understood or
    /// the geotransform isn't invertible.
    pub fn from_grid(
        src_crs: &str,
        src_gt: [f64; 6],
        src_size: [usize; 2],
        mesh: QuadMesh,
        mesh_crs: &str,
    ) -> Result<Self, String> {
        let inner =
            GenImgProjTransformer::new(src_crs, src_gt, mesh_crs, WORLD_GT)?.with_source_size(src_size);
        Ok(MeshTarget::new(mesh, inner))
    }
}

impl<T: Transformer> MeshTarget<T> {
    /// Onto `mesh` through `inner`, a transformer from the mesh's world
    /// coordinates to the source: its destination geotransform is
    /// [`WORLD_GT`].
    pub fn new(mesh: QuadMesh, inner: T) -> Self {
        MeshTarget { mesh, inner }
    }

    /// The target's cells.
    pub fn target_mesh(&self) -> &QuadMesh {
        &self.mesh
    }

    /// The transformer from the mesh's world coordinates to the source.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// The target's size in pixels, `[ncol, nrow]` of the mesh.
    pub fn target_size(&self) -> [usize; 2] {
        [self.mesh.ncol(), self.mesh.nrow()]
    }

    /// Target pixel/line to the mesh's world coordinates, in place.
    fn to_world(&self, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        x.iter_mut()
            .zip(y.iter_mut())
            .map(|(x, y)| match self.mesh.to_world([*x, *y]) {
                Some([a, b]) => {
                    (*x, *y) = (a, b);
                    true
                }
                None => {
                    (*x, *y) = (f64::INFINITY, f64::INFINITY);
                    false
                }
            })
            .collect()
    }
}

/// Points that failed either step go to infinity, with `false`.
fn merge(first: Vec<bool>, mut second: Vec<bool>, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
    for (k, ok) in second.iter_mut().enumerate() {
        *ok &= first[k] && x[k].is_finite() && y[k].is_finite();
        if !*ok {
            (x[k], y[k]) = (f64::INFINITY, f64::INFINITY);
        }
    }
    second
}

/// For a source with world cells: the mesh's world coordinates, then the
/// inner transformer's [`SourceCells::dst_to_world`].
impl<T: Transformer> SourceCells for MeshTarget<T> {
    fn dst_to_world(&self, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let first = self.to_world(x, y);
        let second = match self.inner.source_cells() {
            Some(cells) => cells.dst_to_world(x, y),
            None => vec![false; x.len()],
        };
        merge(first, second, x, y)
    }

    fn mesh(
        &self,
        col_off: usize,
        row_off: usize,
        ncol: usize,
        nrow: usize,
    ) -> Result<QuadMesh, String> {
        match self.inner.source_cells() {
            Some(cells) => cells.mesh(col_off, row_off, ncol, nrow),
            None => Err("the source has no world cells".to_string()),
        }
    }

    fn lon_wrap(&self) -> bool {
        self.inner.source_cells().is_some_and(|c| c.lon_wrap())
    }

    fn worlds(&self) -> usize {
        self.inner.source_cells().map_or(1, |c| c.worlds())
    }

    fn dst_to_world_in(&self, world: usize, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let first = self.to_world(x, y);
        let second = match self.inner.source_cells() {
            Some(cells) => cells.dst_to_world_in(world, x, y),
            None => vec![false; x.len()],
        };
        merge(first, second, x, y)
    }

    fn sheets(
        &self,
        col_off: usize,
        row_off: usize,
        ncol: usize,
        nrow: usize,
    ) -> Result<Vec<SourceSheet>, String> {
        match self.inner.source_cells() {
            Some(cells) => cells.sheets(col_off, row_off, ncol, nrow),
            None => Err("the source has no world cells".to_string()),
        }
    }
}

/// `dst_to_src`: target pixel/line (the mesh's index space) to world, then
/// through the inner transformer; the reverse through it and
/// [`QuadMesh::to_index`]. Points off the mesh fail.
impl<T: Transformer> Transformer for MeshTarget<T> {
    fn source_wrap(&self) -> Option<SourceWrap> {
        self.inner.source_wrap()
    }

    fn source_cells(&self) -> Option<&dyn SourceCells> {
        self.inner.source_cells().map(|_| self as &dyn SourceCells)
    }

    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        if dst_to_src {
            let first = self.to_world(x, y);
            let second = self.inner.transform(true, x, y);
            return merge(first, second, x, y);
        }
        let first = self.inner.transform(false, x, y);
        let (ncol, nrow) = (self.mesh.ncol(), self.mesh.nrow());
        let mut hint = None;
        let second = x
            .iter_mut()
            .zip(y.iter_mut())
            .zip(&first)
            .map(|((x, y), &ok)| {
                let i = ok.then(|| self.mesh.to_index([*x, *y], hint)).flatten();
                if let Some([c, r]) = i {
                    hint = Some(((r as usize).min(nrow - 1), (c as usize).min(ncol - 1)));
                    (*x, *y) = (c, r);
                }
                i.is_some()
            })
            .collect();
        merge(first, second, x, y)
    }
}
