//! A [`QuadMesh`] as a warp source: [`MeshTransformer`], and
//! [`QuadMesh::window`] for source windows.

use vaster::inv_geotransform;

use super::{Period, QuadMesh};
use crate::crs::{crs_pair, CrsTransform};
use crate::transform::{SourceCells, Transformer};

impl QuadMesh {
    /// The cells of the window `ncol` by `nrow` at `(col_off, row_off)` as
    /// a mesh of their own, with this mesh's cell map: cell `(row, col)` of
    /// the window is cell `(row_off + row, col_off + col)` here, with the
    /// same corners. The whole mesh is cloned.
    ///
    /// Walls that meet in the world, such as a seam or a fold, still meet
    /// in the window only if both are in it. Each cell keeps its own
    /// equal-area split, so coverage of a cell in the window is its
    /// coverage in the whole mesh, to rounding. A declared period
    /// ([`QuadMesh::with_period`]) carries over.
    ///
    /// Errors when the window is empty or reaches beyond the mesh.
    pub fn window(
        &self,
        col_off: usize,
        row_off: usize,
        ncol: usize,
        nrow: usize,
    ) -> Result<QuadMesh, String> {
        if ncol == 0 || nrow == 0 || col_off + ncol > self.ncol || row_off + nrow > self.nrow {
            return Err(format!(
                "window of {ncol} x {nrow} cells at ({col_off}, {row_off}) is empty or outside the {} x {} mesh",
                self.ncol, self.nrow
            ));
        }
        if (col_off, row_off, ncol, nrow) == (0, 0, self.ncol, self.nrow) {
            return Ok(self.clone());
        }
        let n = (ncol + 1) * (nrow + 1);
        let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for j in row_off..=row_off + nrow {
            let k = j * (self.ncol + 1) + col_off;
            x.extend_from_slice(&self.x[k..=k + ncol]);
            y.extend_from_slice(&self.y[k..=k + ncol]);
        }
        let mut w = QuadMesh::from_corners(ncol, nrow, x, y, self.map)?;
        // part of a mesh that doesn't overlap itself a period on doesn't
        // either: no need to check again
        w.period = self.period.map(|p| Period {
            period: p.period,
            range: w.x_range(),
        });
        Ok(w)
    }
}

/// A source whose pixels are the cells of a [`QuadMesh`], as a
/// [`Transformer`] from a destination grid's pixels: for warping data on a
/// curvilinear grid given by its corners, such as an ocean model's, and
/// for conservative weights from it.
///
/// Source pixel `(col, row)` is mesh cell `(row, col)`, so a source buffer
/// holds one value per cell, row-major over the mesh. The mesh's corners
/// are in its world coordinates, in the CRS `mesh_crs`; the destination is
/// a grid with geotransform `dst_gt` in `dst_crs`.
///
/// - **Area methods.** The transformer reports the mesh as the source's
///   cells ([`Transformer::source_cells`]), so
///   [`exact_weights`](crate::weights::exact_weights) and
///   [`warp`](crate::warp::warp) with
///   [`Footprint::Exact`](crate::warp::Footprint::Exact) walk each
///   destination pixel's outline onto the mesh in world coordinates. A
///   weight is then the fraction of a cell's world area that the outline
///   covers, across the mesh's seam and fold
///   ([module docs](crate::mesh#seams-and-folds)), and `Average` weights
///   each cell by its world area.
/// - **Cells.** Destination outlines are straight in destination pixel
///   space, each side in `densify` pieces whose end points go to world
///   coordinates exactly. Source cells are straight between their corners
///   in world coordinates, and areas are planar there. Give the mesh in an
///   equal-area projection for areas on the sphere: for a tripolar ocean
///   grid, Lambert azimuthal equal-area on the north pole.
///   `tests/esmf_tripolar.rs` measures the difference from ESMF's
///   great-circle cells.
/// - **Longitude.** A mesh in longitude and latitude with a declared
///   period ([`QuadMesh::with_period`], 360 in degrees) takes outlines
///   that cross its seam, wherever it is, and outlines round a pole,
///   closed along latitude 90 or -90. Areas are then planar in square
///   degrees: a weight is the covered fraction of a cell's square
///   degrees, which differs from its fraction of the cell's area on the
///   sphere as the cosine of latitude varies across the cell.
/// - **Point kernels** map a destination pixel into the mesh's index space
///   through the mesh's own [`CellMap`](crate::mesh::CellMap)
///   ([`QuadMesh::to_index`], at any whole period with a declared one).
///   They read neighbouring source pixels across the mesh's index space,
///   not across a seam or fold.
///
/// ```
/// use rwarp::mesh::{CellMap, MeshTransformer, QuadMesh};
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
/// // the destination: 20 m pixels, north up, in the mesh's own coordinates
/// struct Same;
/// impl CrsTransform for Same {
///     fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> { Some((x, y)) }
/// }
/// let gt = [-200.0, 20.0, 0.0, 210.0, 0.0, -20.0];
/// let t = MeshTransformer::with_backend(mesh, gt, Box::new(Same), Box::new(Same)).unwrap();
/// let w = exact_weights(&t, &t.source_window(), 20, 21, &WeightOptions::new(1));
///
/// // pixel (10, 16) straddles the seam: it takes cells from both ends of
/// // the mesh, and their covered areas add up to the pixel's 400 m^2
/// let here: Vec<usize> = (0..w.len()).filter(|&i| (w.dst_row[i], w.dst_col[i]) == (10, 16)).collect();
/// assert!(here.iter().any(|&i| w.src_col[i] == 0) && here.iter().any(|&i| w.src_col[i] == 23));
/// let area: f64 = here.iter().map(|&i| w.frac[i] * w.src_area(i)).sum();
/// assert!((area - 400.0).abs() < 1e-9, "{area}");
///
/// // a constant field stays constant
/// let s = w.normalised(Normalization::FracArea);
/// let mut sum = vec![0.0; 20 * 21];
/// for i in 0..w.len() {
///     sum[w.dst_row[i] as usize * 20 + w.dst_col[i] as usize] += s[i];
/// }
/// assert!(sum.iter().all(|&v| v == 0.0 || (v - 1.0).abs() < 1e-12));
/// ```
pub struct MeshTransformer {
    mesh: QuadMesh,
    dst_gt: [f64; 6],
    dst_inv_gt: [f64; 6],
    crs_dst_to_mesh: Box<dyn CrsTransform>,
    crs_mesh_to_dst: Box<dyn CrsTransform>,
}

impl MeshTransformer {
    /// The mesh in `mesh_crs` to the grid `dst_gt` in `dst_crs`, with the
    /// compiled-in CRS backend. Fails if a CRS isn't understood or the
    /// geotransform isn't invertible.
    pub fn new(
        mesh: QuadMesh,
        mesh_crs: &str,
        dst_gt: [f64; 6],
        dst_crs: &str,
    ) -> Result<Self, String> {
        let (dst_to_mesh, mesh_to_dst) = crs_pair(mesh_crs, dst_crs)?;
        Self::with_backend(mesh, dst_gt, dst_to_mesh, mesh_to_dst)
    }

    /// With caller-supplied CRS transforms: destination coordinates to the
    /// mesh's world coordinates, and back. Fails if the geotransform isn't
    /// invertible.
    pub fn with_backend(
        mesh: QuadMesh,
        dst_gt: [f64; 6],
        crs_dst_to_mesh: Box<dyn CrsTransform>,
        crs_mesh_to_dst: Box<dyn CrsTransform>,
    ) -> Result<Self, String> {
        let dst_inv_gt =
            inv_geotransform(&dst_gt).ok_or_else(|| "geotransform not invertible".to_string())?;
        Ok(MeshTransformer {
            mesh,
            dst_gt,
            dst_inv_gt,
            crs_dst_to_mesh,
            crs_mesh_to_dst,
        })
    }

    /// The source's cells.
    pub fn source_mesh(&self) -> &QuadMesh {
        &self.mesh
    }

    /// A source window of the whole mesh, for
    /// [`exact_weights`](crate::weights::exact_weights).
    pub fn source_window(&self) -> crate::weights::GridWindow {
        crate::weights::GridWindow::new(self.mesh.ncol(), self.mesh.nrow())
    }

    /// Destination pixel/line to the mesh's world coordinates.
    fn to_world(&self, px: f64, py: f64) -> Option<[f64; 2]> {
        let g = &self.dst_gt;
        let (gx, gy) = (g[0] + px * g[1] + py * g[2], g[3] + px * g[4] + py * g[5]);
        let (wx, wy) = self.crs_dst_to_mesh.convert(gx, gy)?;
        (wx.is_finite() && wy.is_finite()).then_some([wx, wy])
    }
}

/// Points that fail go to infinity, with `false`, as in the other
/// transformers.
fn put(x: &mut f64, y: &mut f64, r: Option<[f64; 2]>) -> bool {
    match r {
        Some([a, b]) if a.is_finite() && b.is_finite() => {
            (*x, *y) = (a, b);
            true
        }
        _ => {
            (*x, *y) = (f64::INFINITY, f64::INFINITY);
            false
        }
    }
}

impl SourceCells for MeshTransformer {
    fn dst_to_world(&self, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        x.iter_mut()
            .zip(y.iter_mut())
            .map(|(x, y)| put(x, y, self.to_world(*x, *y)))
            .collect()
    }

    fn mesh(
        &self,
        col_off: usize,
        row_off: usize,
        ncol: usize,
        nrow: usize,
    ) -> Result<QuadMesh, String> {
        self.mesh.window(col_off, row_off, ncol, nrow)
    }
}

/// `dst_to_src`: destination pixel/line to the mesh's index space
/// `[col, row]`, which is source pixel/line; the reverse through
/// [`QuadMesh::to_world`]. Points off the mesh fail.
impl Transformer for MeshTransformer {
    fn source_cells(&self) -> Option<&dyn SourceCells> {
        Some(self)
    }

    fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        let (ncol, nrow) = (self.mesh.ncol(), self.mesh.nrow());
        let mut hint = None;
        x.iter_mut()
            .zip(y.iter_mut())
            .map(|(x, y)| {
                let r = if dst_to_src {
                    self.to_world(*x, *y)
                        .and_then(|p| self.mesh.to_index(p, hint))
                        .inspect(|i| {
                            hint = Some((
                                (i[1] as usize).min(nrow - 1),
                                (i[0] as usize).min(ncol - 1),
                            ));
                        })
                } else {
                    self.mesh
                        .to_world([*x, *y])
                        .and_then(|p| self.crs_mesh_to_dst.convert(p[0], p[1]))
                        .map(|(gx, gy)| {
                            let g = &self.dst_inv_gt;
                            [g[0] + gx * g[1] + gy * g[2], g[3] + gx * g[4] + gy * g[5]]
                        })
                };
                put(x, y, r)
            })
            .collect()
    }
}
