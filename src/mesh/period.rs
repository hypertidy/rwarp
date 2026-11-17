//! A declared period of world x: [`QuadMesh::with_period`],
//! [`Rectilinear::with_period`](super::Rectilinear::with_period), and
//! [`unwrap_grid`] and [`unwrap_bounds`] for coordinates stored with a
//! jump at the seam.

use super::QuadMesh;

/// A declared period of world x, with the x range of the grid it is
/// declared on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Period {
    pub(crate) period: f64,
    pub(crate) range: [f64; 2],
}

/// Errors unless `period` is finite and positive.
fn check(period: f64) -> Result<(), String> {
    if period.is_finite() && period > 0.0 {
        Ok(())
    } else {
        Err(format!("a period must be finite and positive, got {period}"))
    }
}

impl Period {
    /// `period`, checked, over `range`.
    pub(crate) fn new(period: f64, range: [f64; 2]) -> Result<Self, String> {
        check(period)?;
        Ok(Period { period, range })
    }

    /// The whole turns `k`, as `(first, last)` (none when `first > last`),
    /// at which `[lo, hi]` moved by `k` periods overlaps the range with
    /// area: one that only touches it doesn't count.
    pub(crate) fn turns(&self, lo: f64, hi: f64) -> (i64, i64) {
        let [x0, x1] = self.range;
        (
            (((x0 - hi) / self.period).floor() as i64).saturating_add(1),
            (((x1 - lo) / self.period).ceil() as i64).saturating_sub(1),
        )
    }

    /// The shifts of whole periods at which `x` falls in the range, ends
    /// included: 0 first, whether or not it does, then the others in
    /// order (at most 8).
    pub(crate) fn point_shifts(&self, x: f64) -> impl Iterator<Item = f64> {
        let [x0, x1] = self.range;
        let p = self.period;
        let (k0, k1) = if x.is_finite() {
            (((x0 - x) / p).ceil() as i64, ((x1 - x) / p).floor() as i64)
        } else {
            (1, 0)
        };
        std::iter::once(0.0).chain(
            (k0..=k1.min(k0.saturating_add(8)))
                .filter(|&k| k != 0)
                .map(move |k| k as f64 * p),
        )
    }
}

/// For corner arrays that don't make a mesh: a hint when their x values
/// look like longitudes jumping at a seam (all within one period of 0,
/// and two neighbours along a row more than half a period apart, for a
/// period of 360 or `2 * PI`), naming `helper` to unwrap them.
pub(crate) fn seam_hint(nx: usize, x: &[f64], helper: &str) -> Option<String> {
    for p in [360.0, std::f64::consts::TAU] {
        if x.iter().any(|v| v.abs() > p) {
            continue;
        }
        let Some(k) = (0..x.len().saturating_sub(1))
            .find(|&k| (k + 1) % nx != 0 && (x[k + 1] - x[k]).abs() > p / 2.0)
        else {
            continue;
        };
        return Some(format!(
            "; corners ({}, {}) and ({}, {}) are {} apart in x: longitudes across a seam? \
             Unwrap them (`{helper}`) and declare the period (`QuadMesh::with_period`)",
            k / nx,
            k % nx,
            k / nx,
            k % nx + 1,
            (x[k + 1] - x[k]).abs()
        ));
    }
    None
}

/// The least and greatest of `v`.
pub(crate) fn range_of(v: &[f64]) -> [f64; 2] {
    v.iter()
        .fold([f64::INFINITY, f64::NEG_INFINITY], |[a, b], &x| [a.min(x), b.max(x)])
}

impl QuadMesh {
    /// The mesh with world x periodic with `period`: 360 for longitude in
    /// degrees, `2 * PI` in radians. A point and the same point a whole
    /// number of periods away are then one place, as on a cylinder, and
    /// the mesh's seam (column line `ncol` one period from column line 0)
    /// is crossed like any side, though its two walls are different
    /// places in the plane.
    ///
    /// - [`QuadMesh::burn`] and [`QuadMesh::cover_ring`] take each
    ///   polygon, holes and all, at every whole period at which its
    ///   exterior ring overlaps the mesh's x range, and add up the pieces.
    ///   A polygon is taken as it is in the plane: give one that crosses
    ///   the seam with its longitudes continuous (170 to 190, or -190 to
    ///   -170), not jumping by 360 (which is a polygon the other way
    ///   round), and close one round a pole along the pole, as Natural
    ///   Earth's Antarctica is. A polygon wider than a period overlaps
    ///   itself, and the overlap counts twice.
    /// - [`QuadMesh::locate`], [`QuadMesh::locate_all`] and
    ///   [`QuadMesh::to_index`] find a point at any whole period
    ///   ([`QuadMesh::contains`] and [`QuadMesh::local`] are about one
    ///   cell, in the plane; [`QuadMesh::ring_to_index`] takes the ring as
    ///   it is).
    /// - As a warp source ([`MeshTransformer`](super::MeshTransformer)),
    ///   each destination pixel's outline is unwrapped (no step more than
    ///   half a period) and walked at each whole period at which it
    ///   reaches the mesh. An outline round a pole is closed along y =
    ///   `period / 4` or `-period / 4`, whichever it comes nearer: latitude
    ///   90 or -90, in the same unit as the longitude.
    ///
    /// The corners must be continuous, with no jump of a period between
    /// neighbours: [`unwrap_grid`] and [`unwrap_bounds`] make them so.
    /// The mesh needn't go all the way round, so a regional grid in
    /// longitudes from 120 to 300 takes polygons given from -180 to 180.
    ///
    /// Errors when `period` isn't finite and positive, when a cell spans
    /// half a period or more in x (a cell round a pole, as at the north of
    /// a tripolar grid, which isn't one sheet in longitude and latitude),
    /// or when a cell overlaps a cell a whole period away (the mesh goes
    /// round more than once), found as [`QuadMesh::find_overlap`] finds
    /// overlaps.
    pub fn with_period(mut self, period: f64) -> Result<QuadMesh, String> {
        let p = Period::new(period, self.x_range())?;
        for r in 0..self.nrow {
            for c in 0..self.ncol {
                let q = self.corners(r, c);
                let [a, b] = range_of(&q.map(|v| v[0]));
                if b - a >= period / 2.0 {
                    return Err(format!(
                        "cell ({r}, {c}) spans {} in x, half the period ({period}) or more: round a \
                         pole? A grid with a pole inside a cell isn't one sheet in longitude and \
                         latitude: drop those rows, or give it in a polar projection",
                        b - a
                    ));
                }
            }
        }
        self.period = Some(p);
        if let Some([a, b]) = self.overlap(false) {
            return Err(format!(
                "cells ({}, {}) and ({}, {}) overlap a whole number of periods ({period}) apart: \
                 the mesh goes round more than once",
                a.0, a.1, b.0, b.1
            ));
        }
        Ok(self)
    }

    /// The declared period of world x ([`QuadMesh::with_period`]).
    pub fn period(&self) -> Option<f64> {
        self.period.map(|p| p.period)
    }

    /// The least and greatest world x of the corners.
    pub(crate) fn x_range(&self) -> [f64; 2] {
        range_of(&self.x)
    }
}

/// Unwrap the x values of a grid of points in place, by continuity, for a
/// period of `period`: each value is moved by whole periods to within half
/// a period of the one before it along its row, and the first of each row
/// to within half a period of the first of the row above. `x` is row-major,
/// `nx` values a row and `ny` rows: corners for
/// [`QuadMesh::from_corners`] (`ncol + 1` by `nrow + 1`), or centres for
/// [`QuadMesh::from_centres`] (`ncol` by `nrow`).
///
/// For longitudes stored in `[-180, 180)` or `[0, 360)` on a grid that
/// crosses where they jump. Unwrapped, the grid is one sheet in the plane;
/// declare the period on the mesh with [`QuadMesh::with_period`].
///
/// ```
/// use rwarp::mesh::{unwrap_grid, CellMap, QuadMesh};
///
/// // 10 degree cells from 150 E to 150 W, stored in [-180, 180)
/// let lon: Vec<f64> = (0..=6).map(|i| 150.0 + 10.0 * i as f64).collect();
/// let mut x: Vec<f64> = [lon.clone(), lon].concat();
/// for v in &mut x {
///     *v = (*v + 180.0).rem_euclid(360.0) - 180.0;
/// }
/// let y = [[10.0; 7], [0.0; 7]].concat();
/// assert!(QuadMesh::from_corners(6, 1, x.clone(), y.clone(), CellMap::EqualArea).is_err());
///
/// unwrap_grid(7, 2, &mut x, 360.0).unwrap();
/// assert_eq!(&x[..7], &[150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0]);
/// let mesh = QuadMesh::from_corners(6, 1, x, y, CellMap::EqualArea)
///     .unwrap()
///     .with_period(360.0)
///     .unwrap();
/// // 165 W is in column 4, whichever way it's given
/// assert_eq!(mesh.locate([-165.0, 5.0], None), Some((0, 4)));
/// assert_eq!(mesh.locate([195.0, 5.0], None), Some((0, 4)));
/// ```
///
/// Errors when `x` doesn't have `nx * ny` values, or `period` isn't
/// finite and positive.
pub fn unwrap_grid(nx: usize, ny: usize, x: &mut [f64], period: f64) -> Result<(), String> {
    check(period)?;
    if x.len() != nx * ny {
        return Err(format!("{} values, expected nx * ny = {}", x.len(), nx * ny));
    }
    for k in 1..nx * ny {
        let prev = if k % nx == 0 { x[k - nx] } else { x[k - 1] };
        x[k] -= ((x[k] - prev) / period).round() * period;
    }
    Ok(())
}

/// Unwrap cell bounds in place, by continuity, for a period of `period`:
/// the layout of [`QuadMesh::from_cf_bounds`], `nrow * ncol * 4` values
/// row-major with the vertex last. Each cell's first vertex is moved by
/// whole periods to within half a period of the first vertex of the cell
/// before it along its row (of the first cell of the row above, for the
/// first of a row), and its other vertices to within half a period of its
/// first. Cells must be narrower than half a period.
///
/// For CF bounds or SCRIP corners in longitude on a grid that crosses
/// where its longitudes jump, as at the seam of a global ocean grid.
/// Unwrapped, neighbours share their corners; declare the period on the
/// mesh with [`QuadMesh::with_period`].
///
/// Errors when `xb` doesn't have `nrow * ncol * 4` values, or `period`
/// isn't finite and positive.
pub fn unwrap_bounds(ncol: usize, nrow: usize, xb: &mut [f64], period: f64) -> Result<(), String> {
    check(period)?;
    if xb.len() != ncol * nrow * 4 {
        return Err(format!(
            "{} values, expected nrow * ncol * 4 = {}",
            xb.len(),
            ncol * nrow * 4
        ));
    }
    let near = |v: f64, to: f64| v - ((v - to) / period).round() * period;
    for c in 0..ncol * nrow {
        let at = c * 4;
        if c > 0 {
            let prev = if c % ncol == 0 { xb[at - ncol * 4] } else { xb[at - 4] };
            xb[at] = near(xb[at], prev);
        }
        let first = xb[at];
        for v in &mut xb[at + 1..at + 4] {
            *v = near(*v, first);
        }
    }
    Ok(())
}
