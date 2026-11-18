//! Geometries in another CRS onto a grid in its world: what
//! [`QuadMesh::burn_from`](super::QuadMesh::burn_from) and
//! [`Rectilinear::burn_from`](super::Rectilinear::burn_from) do before
//! they burn (curvilinear stage 5b, `design/curvilinear-frontend.md`).
//!
//! Polygon edges are straight in the geometries' CRS. Each is cut into
//! equal pieces no longer than the grid's tallest cell is high there, and
//! each piece densified by bigcurve into the grid's world, as
//! [`MeshSet::burn`](super::MeshSet::burn) does for a band.

use bigcurve::{from_fn, Options, Planar, Resampler};
use controlledburn::{Coord, Geometry, Polygon};

use super::set::{project, Project};
use crate::crs::CrsTransform;

/// Where a grid's cells are in the geometries' CRS: the y range of its
/// vertices there, widened by its tallest cell each way, and that height.
#[derive(Debug, Clone, Copy)]
pub(super) struct Reach {
    pub(super) y: [f64; 2],
    pub(super) step: f64,
}

/// The [`Reach`] of an `ncol` by `nrow` grid whose vertex `(j, i)` (row
/// line `j`, column line `i`) is `vertex(j, i)` in its world, through
/// `from_world`. Cells with a vertex that doesn't go back are left out of
/// it; Err when none goes.
pub(super) fn reach(
    ncol: usize,
    nrow: usize,
    vertex: impl Fn(usize, usize) -> [f64; 2],
    from_world: &dyn CrsTransform,
) -> Result<Reach, String> {
    let w = ncol + 1;
    let ys: Vec<f64> = (0..=nrow)
        .flat_map(|j| (0..w).map(move |i| (j, i)))
        .map(|(j, i)| {
            let p = vertex(j, i);
            match from_world.convert(p[0], p[1]) {
                Some((x, y)) if x.is_finite() && y.is_finite() => y,
                _ => f64::NAN,
            }
        })
        .collect();
    let (mut y0, mut y1, mut tall) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
    for j in 0..nrow {
        for i in 0..ncol {
            let c = [ys[j * w + i], ys[j * w + i + 1], ys[(j + 1) * w + i + 1], ys[(j + 1) * w + i]];
            if c.iter().any(|v| v.is_nan()) {
                continue;
            }
            let (lo, hi) = c.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
            (y0, y1, tall) = (y0.min(lo), y1.max(hi), tall.max(hi - lo));
        }
    }
    if tall <= 0.0 {
        tall = y1 - y0;
    }
    if !(tall > 0.0 && tall.is_finite()) {
        return Err("no cell of the grid goes to the geometries' CRS".into());
    }
    Ok(Reach { y: [y0 - tall, y1 + tall], step: tall })
}

/// `geoms` (in the geometries' CRS) in the grid's world, one for one:
/// each polygon cut to `reach`, densified into the world to `tolerance`
/// and split where it meets itself (a multipolygon of its parts, empty
/// when nothing is left), and each point taken there (left out when it
/// doesn't go, or is beyond `reach`). With the world's `period`, rings
/// are unwrapped by it.
pub(super) fn to_world(
    geoms: &[Geometry],
    to_world: &dyn CrsTransform,
    reach: Reach,
    tolerance: f64,
    period: Option<f64>,
) -> Result<Vec<Geometry>, String> {
    if !(tolerance > 0.0 && tolerance.is_finite()) {
        return Err(format!("the tolerance must be positive, not {tolerance}"));
    }
    let forward = from_fn(|x: f64, y: f64| to_world.convert(x, y).map(|(a, b)| [a, b]));
    let mut rs = Resampler::with_edge(forward, Planar::new(), &Options::new(tolerance));
    let point = |c: &Coord| -> Option<Coord> {
        if !(c.y >= reach.y[0] && c.y <= reach.y[1] && c.x.is_finite()) {
            return None;
        }
        to_world.convert(c.x, c.y).filter(|(x, y)| x.is_finite() && y.is_finite()).map(|(x, y)| Coord { x, y })
    };
    let mut out = Vec::with_capacity(geoms.len());
    for (k, g) in geoms.iter().enumerate() {
        let polys = match g {
            Geometry::Polygon(p) => std::slice::from_ref(p),
            Geometry::MultiPolygon(ps) => ps.as_slice(),
            Geometry::Point(c) => {
                out.push(match point(c) {
                    Some(w) => Geometry::Point(w),
                    None => Geometry::MultiPoint(Vec::new()),
                });
                continue;
            }
            Geometry::MultiPoint(cs) => {
                out.push(Geometry::MultiPoint(cs.iter().filter_map(point).collect()));
                continue;
            }
            Geometry::LineString(_) | Geometry::MultiLineString(_) => {
                return Err(format!("geometry {k}: lines are not supported by the grid front ends yet"));
            }
        };
        let parts = project(polys, reach.y, reach.step, period, &mut rs).map_err(|e| match e {
            Project::Failed(v) => format!("geometry {k}: the point ({}, {}) doesn't go to the grid's world", v[0], v[1]),
            Project::Unclosed => format!(
                "geometry {k}: a ring goes round a pole of the grid's world (it doesn't close within its period)"
            ),
        })?;
        out.push(Geometry::MultiPolygon(parts.iter().map(|p| p.polygon()).collect::<Vec<Polygon>>()));
    }
    Ok(out)
}
