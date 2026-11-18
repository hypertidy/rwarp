//! `MeshSet`: a whole global grid as rectangular pieces in two bands, each
//! band in its own polar LAEA, with damaged land cells left out.
//!
//! The grid is analytic, written as CF bounds in longitude and latitude:
//! 48 columns of 7.5 degrees, regular rows from 78 S to 60 N, then a
//! tripolar cap of 8 rows from 60 N with a seam and a fold (corners in
//! north-polar LAEA, as `tests/mesh_glued.rs`'s disk), the geographic
//! pole inside one of its cells. Two blocks of land have a corner moved
//! well away, so the cells round it fold over or overlap others: one in
//! the south band, one in the north. The bands are south-polar and
//! north-polar LAEA on one sphere, split at the equator.

use std::collections::{HashMap, HashSet};

use controlledburn::{BurnMode, BurnResult, Coord, Geometry, Polygon};
use rwarp::mesh::{Band, Halo, MeshSet, MeshTransformer, QuadMesh, SetTransformer, WORLD_GT};
use rwarp::transform::{CenterLong, GenImgProjTransformer};
use rwarp::warp::{warp_to, warp_to_par, DstNodata, Footprint, ResampleAlg, SourceBuffer, WarpOptions};
use rwarp::weights::{exact_weights, exact_weights_par, lonlat_measure, ApplyOptions, GridWindow, Normalization, WeightOptions, Weights};
use rwarp::{CrsTransform, Transformer};

const R: f64 = 6371229.0;
const NC: usize = 48;
/// Regular rows, 6 degrees from 78 S to 60 N; row line 13 is the equator.
const NREG: usize = 23;
const NCAP: usize = 8;
const NR: usize = NREG + NCAP;
const EQUATOR: usize = 13;

fn laea_n(lon: f64, lat: f64) -> [f64; 2] {
    let rho = 2.0 * R * ((90.0 - lat).to_radians() / 2.0).sin();
    [rho * lon.to_radians().sin(), -rho * lon.to_radians().cos()]
}
fn laea_n_inv(x: f64, y: f64) -> [f64; 2] {
    [x.atan2(-y).to_degrees(), 90.0 - 2.0 * (x.hypot(y) / (2.0 * R)).asin().to_degrees()]
}
fn laea_s(lon: f64, lat: f64) -> [f64; 2] {
    let rho = 2.0 * R * ((90.0 + lat).to_radians() / 2.0).sin();
    [rho * lon.to_radians().sin(), rho * lon.to_radians().cos()]
}
fn laea_s_inv(x: f64, y: f64) -> [f64; 2] {
    [x.atan2(y).to_degrees(), -90.0 + 2.0 * (x.hypot(y) / (2.0 * R)).asin().to_degrees()]
}

/// Longitude and latitude to band `0` (south) or `1` (north)'s world, or
/// back with `.1`.
struct World(usize, bool);
impl CrsTransform for World {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let p = match (self.0, self.1) {
            (0, false) => laea_s(x, y),
            (0, true) => laea_s_inv(x, y),
            (_, false) => laea_n(x, y),
            (_, true) => laea_n_inv(x, y),
        };
        Some((p[0], p[1]))
    }
}

/// Corners `(j, i)` moved by `(dlon, dlat)`.
type Moved = [((usize, usize), (f64, f64))];

/// Corner `(j, i)` in longitude and latitude, with corners `moved`.
fn corner(j: usize, i: usize, moved: &Moved) -> (f64, f64) {
    let (lon, lat) = if j <= NREG {
        (7.5 * i as f64, -78.0 + 6.0 * j as f64)
    } else {
        // the cap: from the circle of 60 N (t = 0) to the fold (t = 1),
        // built in a frame turned a quarter turn from north-polar LAEA
        let t = (j - NREG) as f64 / NCAP as f64;
        let rho0 = 2.0 * R * 15f64.to_radians().sin();
        let s = 0.35 * rho0;
        let i = i % NC;
        let th = std::f64::consts::TAU * i as f64 / NC as f64;
        let phi = std::f64::consts::TAU * i.min(NC - i) as f64 / NC as f64;
        let (u, v) = ((1.0 - t) * rho0 * th.cos() + t * s * phi.cos(), (1.0 - t) * rho0 * th.sin());
        let (x, y) = (v + t * 0.13 * s, -u + t * 0.07 * s);
        let [lon, lat] = laea_n_inv(x, y);
        (lon, lat)
    };
    match moved.iter().find(|m| m.0 == (j, i)) {
        Some(&(_, (dx, dy))) => (lon + dx, lat + dy),
        None => (lon, lat),
    }
}

/// The damaged corners: one inside a block of land in each band.
const MOVED: [((usize, usize), (f64, f64)); 2] = [((4, 32), (-12.0, 0.5)), ((17, 23), (31.0, 4.0))];

/// Land: a block round each damaged corner, and a strip along the south.
fn land() -> Vec<bool> {
    (0..NC * NR)
        .map(|k| {
            let (r, c) = (k / NC, k % NC);
            r == 0 || ((2..7).contains(&r) && (28..37).contains(&c)) || ((15..20).contains(&r) && (18..29).contains(&c))
        })
        .collect()
}

fn bounds(moved: &Moved) -> (Vec<f64>, Vec<f64>) {
    let (mut xb, mut yb) = (Vec::new(), Vec::new());
    for j in 0..NR {
        for i in 0..NC {
            for (jj, ii) in [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)] {
                let (x, y) = corner(jj, ii, moved);
                xb.push(x);
                yb.push(y);
            }
        }
    }
    (xb, yb)
}

fn set(moved: &Moved, land: Option<&[bool]>) -> Result<MeshSet, String> {
    let (xb, yb) = bounds(moved);
    let (s, n) = (World(0, false), World(1, false));
    let bands = [Band { rows: 0..EQUATOR, to_world: &s }, Band { rows: EQUATOR..NR, to_world: &n }];
    MeshSet::from_cf_bounds(NC, NR, &xb, &yb, Halo::NONE, &bands, land)
}

/// A global 2.5 degree grid in longitude and latitude.
const GT: [f64; 6] = [-180.0, 2.5, 0.0, 90.0, 0.0, -2.5];
const GW: usize = 144;
const GH: usize = 72;

fn to_grid(s: &MeshSet, densify: usize) -> Weights {
    s.weights_to(
        |w, mesh| MeshTransformer::with_backend(mesh, GT, Box::new(World(w, false)), Box::new(World(w, true))),
        GW,
        GH,
        &WeightOptions::new(densify),
    )
    .unwrap()
}

type Key = (u32, u32, u32, u32);

fn entries(w: &Weights) -> HashMap<Key, f64> {
    (0..w.len()).map(|k| ((w.dst_row[k], w.dst_col[k], w.src_row[k], w.src_col[k]), w.frac[k])).collect()
}

/// The damaged cells are left out, and nothing else but land; each band's
/// pieces are in its own world, and cover every cell kept exactly once.
#[test]
fn damaged_land_is_left_out() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let dropped: HashSet<_> = s.dropped().iter().copied().collect();
    eprintln!("{} pieces, {} cells left out: {:?}", s.pieces().len(), dropped.len(), s.dropped());
    for &((j, i), _) in &MOVED {
        // the four cells round each damaged corner fold over or overlap
        assert!(dropped.contains(&(j - 1, i - 1)) || dropped.contains(&(j, i - 1)), "round ({j}, {i})");
    }
    assert!(dropped.iter().all(|&(r, c)| land[r * NC + c]), "only land is left out");
    let mut owner = vec![0; NC * NR];
    for p in s.pieces() {
        let [c0, r0] = p.offset();
        let m = p.mesh();
        assert_eq!(p.world(), if r0 < EQUATOR { 0 } else { 1 });
        assert!(r0 + m.nrow() <= EQUATOR || r0 >= EQUATOR, "a piece across the bands");
        for r in 0..m.nrow() {
            for c in 0..m.ncol() {
                owner[(r0 + r) * NC + c0 + c] += 1;
            }
        }
    }
    for (k, &o) in owner.iter().enumerate() {
        let want = if dropped.contains(&(k / NC, k % NC)) { 0 } else { 1 };
        assert_eq!(o, want, "cell ({}, {})", k / NC, k % NC);
    }
    assert!(s.pieces().len() >= 6);

    // undamaged, nothing is left out: one piece a band
    let clean = set(&[], Some(&land)).unwrap();
    assert!(clean.dropped().is_empty());
    assert_eq!(clean.pieces().len(), 2);
}

/// Damaged cells that may not be left out are refused, named by their
/// place in the file; and the bands must tile the rows.
#[test]
fn what_is_refused() {
    let e = set(&MOVED, None).unwrap_err();
    eprintln!("{e}");
    assert!(e.contains("may not be left out"), "{e}");
    let (xb, yb) = bounds(&[]);
    let (s, n) = (World(0, false), World(1, false));
    for rows in [[0..10, 13..NR], [0..13, 13..NR - 1], [0..14, 13..NR]] {
        let bands = [Band { rows: rows[0].clone(), to_world: &s }, Band { rows: rows[1].clone(), to_world: &n }];
        assert!(MeshSet::from_cf_bounds(NC, NR, &xb, &yb, Halo::NONE, &bands, None).is_err(), "{rows:?}");
    }
    // the file's numbering, with a halo of one row of land first
    let (xb, yb) = bounds(&MOVED);
    let bands = [Band { rows: 0..EQUATOR - 1, to_world: &s }, Band { rows: EQUATOR - 1..NR - 1, to_world: &n }];
    let halo = Halo { rows: [1, 0], ..Halo::NONE };
    let e = MeshSet::from_cf_bounds(NC, NR, &xb, &yb, halo, &bands, None).unwrap_err();
    assert!(e.contains("cell (3, 31)") || e.contains("cell (3, 32)") || e.contains("cell (4, 31)"), "{e}");
}

/// Weights from the set to a global grid: every cell kept is shared out
/// in full, across the seam, the fold, the pole and the two bands; cells
/// left out have none; and the cells away from the damage have the
/// undamaged set's weights.
#[test]
fn weights_to_a_grid() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let clean = set(&[], Some(&land)).unwrap();
    let dropped: HashSet<_> = s.dropped().iter().copied().collect();
    for densify in [1, 4] {
        let w = to_grid(&s, densify);
        let mut sums = vec![0.0; NC * NR];
        for k in 0..w.len() {
            sums[w.src_row[k] as usize * NC + w.src_col[k] as usize] += w.frac[k];
        }
        let worst = (0..NC * NR)
            .filter(|k| !dropped.contains(&(k / NC, k % NC)))
            .map(|k| (sums[k] - 1.0).abs())
            .fold(0.0, f64::max);
        let none = dropped.iter().all(|&(r, c)| sums[r * NC + c] == 0.0);
        // Where the bands meet, each side's cells are straight in their own
        // LAEA, and bow away from the equator between corners: a lens up
        // to R dlam^2 / 4 high on either side, dlam the cells' width. Target
        // pixels next to it are short of their area by up to that over
        // their height, dlam^2 / (4 dphi): 9.8% for these 7.5 degree cells
        // and 2.5 degree pixels.
        let covered = w.dst_covered();
        let short = |rows: std::ops::Range<usize>| {
            rows.flat_map(|r| (0..GW).map(move |c| r * GW + c))
                .map(|k| (1.0 - covered[k] / w.dst_area[k]).abs())
                .fold(0.0, f64::max)
        };
        let (dlam, dphi) = (7.5f64.to_radians(), 2.5f64.to_radians());
        let gap = short(35..37);
        // away from the bands' meeting and the damage: the cap from 75 N to
        // 55 N, round the fold, and 10 S to 20 S
        let away = short(6..14).max(short(40..44));
        eprintln!("densify {densify}: {} entries; kept cells shared out to {worst:.1e}; pixels next to the equator short by up to {gap:.1e} (bound {:.1e}), elsewhere {away:.1e}", w.len(), dlam * dlam / (4.0 * dphi));
        assert!(worst < 1e-12, "{worst:e}");
        assert!(none);
        assert!(gap > 1e-3 && gap < dlam * dlam / (4.0 * dphi), "{gap:e}");
        assert!(away < 1e-12, "{away:e}");

        // away from the damage, the undamaged set's weights
        let near = |r: u32, c: u32| {
            MOVED.iter().any(|&((j, i), _)| (r as i64 - j as i64).abs() <= 3 && (c as i64 - i as i64).abs() <= 3)
        };
        let (a, b) = (entries(&w), entries(&to_grid(&clean, densify)));
        let d = a
            .iter()
            .filter(|(k, _)| !near(k.2, k.3))
            .map(|(k, v)| (v - b.get(k).copied().unwrap_or(0.0)).abs())
            .chain(b.iter().filter(|(k, _)| !near(k.2, k.3)).map(|(k, v)| (v - a.get(k).copied().unwrap_or(0.0)).abs()))
            .fold(0.0, f64::max);
        eprintln!("  away from the damage, against the undamaged set: {d:.1e}");
        assert!(d < 1e-12, "{d:e}");
    }
}

/// Weights from a global grid onto the set, areas on the sphere: every
/// cell kept is covered in full and its area on the sphere is its LAEA
/// area to second order in `densify`; cells left out get nothing.
#[test]
fn weights_onto_the_set() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let dropped: HashSet<_> = s.dropped().iter().copied().collect();
    let steradian = (2.5f64.to_radians()).powi(2) * R * R;
    let mut last = f64::INFINITY;
    for densify in [4, 16] {
        let opts = WeightOptions::new(densify).with_measure(lonlat_measure(GT).unwrap());
        let w = s
            .weights_onto(
                |k| {
                    GenImgProjTransformer::with_backend(GT, WORLD_GT, Box::new(World(k, true)), Box::new(World(k, false)))
                        .map(|t| t.with_center_long(CenterLong::At(0.0)).with_source_size([GW, GH]))
                },
                &GridWindow::new(GW, GH),
                &opts,
            )
            .unwrap();
        let covered = w.dst_covered();
        let (mut cover, mut area) = (0.0f64, 0.0f64);
        for (k, &cov) in covered.iter().enumerate() {
            let (r, c) = (k / NC, k % NC);
            if dropped.contains(&(r, c)) {
                assert!(w.dst_area[k].is_nan() && cov == 0.0, "cell ({r}, {c})");
                continue;
            }
            let (p, pr, pc) = s.locate_cell(r, c).unwrap();
            let a = p.mesh().cell_area(pr, pc);
            if a == 0.0 {
                continue;
            }
            cover = cover.max((cov / w.dst_area[k] - 1.0).abs());
            area = area.max((w.dst_area[k] * steradian - a).abs() / a);
        }
        eprintln!("densify {densify:2}: {} entries; covered to {cover:.1e}; cell areas on the sphere vs LAEA {area:.1e}", w.len());
        assert!(cover < 1e-12, "{cover:e}");
        assert!(area < last / 8.0, "{area:e} after {last:e}");
        last = area;
    }
}

/// The same set from the same cells, with NEMO's layout round them: the
/// cyclic halo's two columns and a first row of land dropped.
#[test]
fn a_halo_gives_the_same_set() {
    let land = land();
    let plain = set(&MOVED, Some(&land)).unwrap();
    // file: a land row first, columns NC and NC + 1 repeating 0 and 1
    let (fc, fr) = (NC + 2, NR + 1);
    let (xb0, yb0) = bounds(&MOVED);
    let (mut xb, mut yb) = (Vec::new(), Vec::new());
    let mut fland = Vec::new();
    for j in 0..fr {
        for i in 0..fc {
            let (sj, si) = (j.saturating_sub(1), i % NC);
            for v in 0..4 {
                let k = (sj * NC + si) * 4 + v;
                // the land row: collapsed onto the first corner of the row above
                let k = if j == 0 { sj * NC * 4 + si * 4 } else { k };
                xb.push(xb0[k]);
                yb.push(yb0[k]);
            }
            fland.push(j == 0 || land[sj * NC + si]);
        }
    }
    let (s, n) = (World(0, false), World(1, false));
    let bands = [Band { rows: 0..EQUATOR, to_world: &s }, Band { rows: EQUATOR..NR, to_world: &n }];
    let halo = Halo { cols: [0, 2], rows: [1, 0], ..Halo::NONE };
    let h = MeshSet::from_cf_bounds(fc, fr, &xb, &yb, halo, &bands, Some(&fland)).unwrap();
    assert_eq!(h.size(), [NC, NR]);
    assert_eq!((h.file_size(), h.halo()), ([fc, fr], halo));
    // a field in the file's layout to the set's numbering and back; this
    // halo isn't NEMO's whole, so it isn't filled
    let field: Vec<u16> = (0..fc * fr).map(|k| (k % fc % NC + 100 * (k / fc)) as u16).collect();
    let set_field = h.from_file_layout(&field, 1).unwrap();
    assert!((0..NC * NR).all(|k| set_field[k] == field[(k / NC + 1) * fc + k % NC]));
    let mut back = h.to_file_layout(&set_field, 1, 9999).unwrap();
    assert!((0..fc * fr).all(|k| back[k] == if k < fc || k % fc >= NC { 9999 } else { field[k] }));
    assert!(h.fill_halo(&mut back, 1, rwarp::mesh::Stagger::T).unwrap_err().contains("NEMO"));
    let mut signed: Vec<i32> = back.iter().map(|&v| v as i32).collect();
    assert!(h.fill_halo_vector(&mut signed, 1, rwarp::mesh::Stagger::V).unwrap_err().contains("NEMO"));
    assert_eq!(h.dropped(), plain.dropped());
    assert_eq!(h.pieces().len(), plain.pieces().len());
    for (a, b) in h.pieces().iter().zip(plain.pieces()) {
        assert_eq!(a.offset(), b.offset());
        for r in 0..a.mesh().nrow() {
            for c in 0..a.mesh().ncol() {
                assert_eq!(a.mesh().corners(r, c), b.mesh().corners(r, c));
            }
        }
    }
}

/// Each piece is measured over the target pixels it reaches (round the
/// pole and across the target's seam, the whole grid): the same weights
/// as each piece over the whole grid, bit for bit at densify 1 and 4.
#[test]
fn pieces_reach_all_their_pixels() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    for densify in [1, 4] {
        let ours = entries(&to_grid(&s, densify));
        let mut whole = HashMap::new();
        for p in s.pieces() {
            let [c0, r0] = p.offset();
            let k = p.world();
            let t = MeshTransformer::with_backend(p.mesh().clone(), GT, Box::new(World(k, false)), Box::new(World(k, true))).unwrap();
            let w = rwarp::weights::exact_weights(&t, &t.source_window(), GW, GH, &WeightOptions::new(densify));
            for i in 0..w.len() {
                whole.insert((w.dst_row[i], w.dst_col[i], w.src_row[i] + r0 as u32, w.src_col[i] + c0 as u32), w.frac[i]);
            }
        }
        assert_eq!(ours.len(), whole.len(), "densify {densify}");
        assert!(ours.iter().all(|(k, v)| whole.get(k) == Some(v)), "densify {densify}");
    }
}

/// Burns onto the set take the bands it was built with.
fn burn(s: &MeshSet, geoms: &[Geometry]) -> Result<BurnResult, String> {
    let (south, north) = (World(0, false), World(1, false));
    let bands = [Band { rows: 0..EQUATOR, to_world: &south }, Band { rows: EQUATOR..NR, to_world: &north }];
    s.burn(geoms, &bands, 0.1, BurnMode::Coverage)
}

/// Each cell's burned fraction for geometry `id`, `(row, col)` to
/// fraction; a cell met twice is an error.
fn fractions(r: &BurnResult, id: i32) -> HashMap<(usize, usize), f64> {
    let mut out = HashMap::new();
    let mut put = |row: i32, col: i32, f: f64| {
        assert!(out.insert((row as usize, col as usize), f).is_none(), "cell ({row}, {col}) twice");
    };
    for run in r.runs.iter().filter(|x| x.id == id) {
        for col in run.col_start..run.col_end {
            put(run.row, col, 1.0);
        }
    }
    for e in r.edges.iter().filter(|x| x.id == id) {
        put(e.row, e.col, e.fraction as f64);
    }
    out
}

/// A ring from `[lon, lat]` corners, closed.
fn ring(pts: &[[f64; 2]]) -> Vec<Coord> {
    pts.iter().chain(pts.first()).map(|p| Coord { x: p[0], y: p[1] }).collect()
}

/// The box `lon0..lon1` by `lat0..lat1`, its rings (outer first) and holes.
fn lonlat_box(b: [f64; 4]) -> Vec<Coord> {
    let [x0, x1, y0, y1] = b;
    ring(&[[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
}

/// The area of a regular cell's box `[lon0, lon1, lat0, lat1]` covers
/// (grid lines, lat0 < lat1, longitudes taken modulo 360), in its band's
/// LAEA. Cell edges are chords of the parallels there and the box's
/// edges along a parallel are arcs (densified): the arc at latitude `L`
/// bows past the chord by the segment `S(L) = rho(L)^2 (dlam - sin dlam)
/// / 2`, away from the band's pole, into the cell on that side of `L`,
/// and covers the band's pole side of `L`.
fn box_area(r: usize, c: usize, area: f64, b: [f64; 4]) -> f64 {
    let [lon0, lon1, lat0, lat1] = b;
    let mid = 7.5 * (c as f64 + 0.5);
    if ![-360.0, 0.0, 360.0].iter().any(|k| (lon0..=lon1).contains(&(mid + k))) {
        return 0.0;
    }
    let (a, b) = (-78.0 + 6.0 * r as f64, -72.0 + 6.0 * r as f64);
    let dlam = 7.5f64.to_radians();
    let south = r < EQUATOR;
    let seg = |l: f64| {
        let rho = 2.0 * R * ((90.0 + if south { l } else { -l }).to_radians() / 2.0).sin();
        rho * rho * (dlam - dlam.sin()) / 2.0
    };
    let within = lat0 <= a && b <= lat1;
    if south {
        // the cell is the band between the arcs, with S(a) and less S(b);
        // S(L) lies just south of L
        let s = |l: f64| if lat0 < l && l <= lat1 { seg(l) } else { 0.0 };
        (if within { area - seg(a) + seg(b) } else { 0.0 }) + s(a) - s(b)
    } else {
        // with S(b) and less S(a); S(L) lies just north of L
        let s = |l: f64| if lat0 <= l && l < lat1 { seg(l) } else { 0.0 };
        (if within { area + seg(a) - seg(b) } else { 0.0 }) + s(b) - s(a)
    }
}

/// Cells with a corner moved, whose shape isn't the regular one.
fn moved_cell(r: usize, c: usize) -> bool {
    MOVED.iter().any(|&((j, i), _)| (r == j || r + 1 == j) && (c == i || c + 1 == i))
}

/// Burned fractions against `want(row, col, area)` on every cell kept, and
/// nothing on cells left out: the worst difference.
fn against(s: &MeshSet, got: &HashMap<(usize, usize), f64>, want: impl Fn(usize, usize, f64) -> Option<f64>) -> f64 {
    let dropped: HashSet<_> = s.dropped().iter().copied().collect();
    assert!(got.keys().all(|k| !dropped.contains(k)), "a cell left out was burned");
    let mut worst = 0.0f64;
    for r in 0..NR {
        for c in 0..NC {
            let Some((p, pr, pc)) = s.locate_cell(r, c) else { continue };
            let area = p.mesh().cell_area(pr, pc);
            let Some(f) = want(r, c, area) else { continue };
            let g = got.get(&(r, c)).copied().unwrap_or(0.0);
            assert!((g - f).abs() < 2e-6, "cell ({r}, {c}): burned {g}, expected {f}");
            worst = worst.max((g - f).abs());
        }
    }
    worst
}

/// A box round the whole world covers every cell kept in full, in both
/// bands, round the pole and the fold: each band takes only the box's
/// part near its own cells, so the far pole doesn't turn it inside out.
/// The same in the grid's own longitudes, 0 to 360.
#[test]
fn burn_the_whole_world() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let geoms: Vec<Geometry> = [[-180.0, 180.0, -90.0, 90.0], [0.0, 360.0, -90.0, 90.0]]
        .iter()
        .map(|&b| Geometry::Polygon(Polygon::new(vec![lonlat_box(b)])))
        .collect();
    let r = burn(&s, &geoms).unwrap();
    for id in 0..2 {
        let got = fractions(&r, id);
        let worst = against(&s, &got, |_, _, _| Some(1.0));
        eprintln!("whole world ({id}): {} cells, worst {worst:.1e}", got.len());
        assert_eq!(got.len(), NC * NR - s.dropped().len());
    }
}

/// Boxes against their areas in closed form: the two polar caps (only the
/// near band's cells; the far band gets nothing), a box across the bands
/// with a hole, and a box across 180 written past it and split at it.
#[test]
fn burn_boxes_against_their_areas() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let poly = |rings: Vec<Vec<Coord>>| Geometry::Polygon(Polygon::new(rings));
    let (outer, hole) = ([-30.0, 30.0, -30.0, 30.0], [-15.0, 15.0, -12.0, 12.0]);
    let geoms = vec![
        poly(vec![lonlat_box([-180.0, 180.0, -90.0, -60.0])]),
        poly(vec![lonlat_box([-180.0, 180.0, 60.0, 90.0])]),
        poly(vec![lonlat_box(outer), lonlat_box(hole).into_iter().rev().collect()]),
        poly(vec![lonlat_box([150.0, 210.0, -60.0, -30.0])]),
        Geometry::MultiPolygon(vec![
            Polygon::new(vec![lonlat_box([150.0, 180.0, -60.0, -30.0])]),
            Polygon::new(vec![lonlat_box([-180.0, -150.0, -60.0, -30.0])]),
        ]),
        // a cap cut at 232.5 E, the east wall of a piece by the damage
        poly(vec![lonlat_box([-127.5, 232.5, -90.0, -48.0])]),
    ];
    let r = burn(&s, &geoms).unwrap();
    let regular = |r: usize, c: usize| r < NREG && !moved_cell(r, c);

    // the south cap: rows to 60 S in full, the row above it its slivers
    let got = fractions(&r, 0);
    let w = against(&s, &got, |r, c, a| regular(r, c).then(|| box_area(r, c, a, [-180.0, 180.0, -90.0, -60.0]) / a));
    assert!(got.keys().all(|&(r, _)| r < 4), "the south cap reached row {:?}", got.keys().map(|k| k.0).max());
    eprintln!("south cap: {} cells, worst {w:.1e}", got.len());

    // the north cap: the cap's cells in full, round the fold and the pole
    let got = fractions(&r, 1);
    let w = against(&s, &got, |r, c, a| {
        if r >= NREG {
            Some(1.0)
        } else {
            regular(r, c).then(|| box_area(r, c, a, [-180.0, 180.0, 60.0, 90.0]) / a)
        }
    });
    assert!(got.keys().all(|&(r, _)| r >= NREG - 1));
    eprintln!("north cap: {} cells, worst {w:.1e}", got.len());

    // across the bands, with a hole: each band's cells to the equator,
    // none in the gap between them
    let got = fractions(&r, 2);
    let w = against(&s, &got, |r, c, a| {
        regular(r, c).then(|| (box_area(r, c, a, outer) - box_area(r, c, a, hole)) / a)
    });
    eprintln!("across the bands, with a hole: {} cells, worst {w:.1e}", got.len());
    assert!(got.contains_key(&(EQUATOR - 1, 3)) && got.contains_key(&(EQUATOR, 3)));
    assert!(!got.contains_key(&(EQUATOR - 1, 1)) && !got.contains_key(&(EQUATOR, 1)), "inside the hole");

    // a cap whose cut runs along a piece's wall: its slit is taken out,
    // not walked
    assert!(s.pieces().iter().any(|p| p.offset() == [0, 3] && p.mesh().ncol() == 31));
    let got = fractions(&r, 5);
    let w = against(&s, &got, |r, c, a| regular(r, c).then(|| box_area(r, c, a, [-127.5, 232.5, -90.0, -48.0]) / a));
    eprintln!("a cap cut along a wall: {} cells, worst {w:.1e}", got.len());

    // across 180: as written past it, and split at it
    let b = [150.0, 210.0, -60.0, -30.0];
    for id in [3, 4] {
        let got = fractions(&r, id);
        let w = against(&s, &got, |r, c, a| regular(r, c).then(|| box_area(r, c, a, b) / a));
        eprintln!("across 180 ({id}): {} cells, worst {w:.1e}", got.len());
        assert!(!got.is_empty());
    }
    let (one, two) = (fractions(&r, 3), fractions(&r, 4));
    assert_eq!(one.len(), two.len());
    let d = one.iter().map(|(k, v)| (v - two[k]).abs()).fold(0.0, f64::max);
    assert!(d < 1e-6, "{d:e}");
}

/// `burn_par` gives `burn`'s records on any number of threads.
#[test]
fn burn_par_is_burn() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let (south, north) = (World(0, false), World(1, false));
    let bands = [Band { rows: 0..EQUATOR, to_world: &south }, Band { rows: EQUATOR..NR, to_world: &north }];
    let poly = |b| Geometry::Polygon(Polygon::new(vec![lonlat_box(b)]));
    let geoms = vec![
        poly([-180.0, 180.0, -90.0, 90.0]),
        poly([-180.0, 180.0, 60.0, 90.0]),
        poly([-127.5, 232.5, -90.0, -48.0]),
        poly([150.0, 210.0, -60.0, -30.0]),
        Geometry::MultiPoint((0..40).map(|k| Coord { x: 9.0 * k as f64, y: -75.0 + 4.0 * k as f64 }).collect()),
    ];
    let one = s.burn(&geoms, &bands, 0.1, BurnMode::Coverage).unwrap();
    for threads in [0, 2, 5] {
        assert_eq!(one, s.burn_par(&geoms, &bands, 0.1, BurnMode::Coverage, threads).unwrap(), "{threads} threads");
        // projected on the threads too, each through its own copy
        let par = s.burn_par_with(&geoms, &bands, 0.1, BurnMode::Coverage, threads, |k| Ok(World(k, false)));
        assert_eq!(one, par.unwrap(), "{threads} threads, projected on them");
    }
}

/// With a CRS backend's transforms (libproj's can't be shared between
/// threads), each thread building its own: `burn`'s records.
#[cfg(any(feature = "proj", feature = "proj4rs"))]
#[test]
fn burn_par_with_a_backend() {
    use rwarp::crs::crs_pair;
    let lonlat = format!("+proj=longlat +R={R}");
    let laea = [format!("+proj=laea +lat_0=-90 +R={R}"), format!("+proj=laea +lat_0=90 +R={R}")];
    let to_world = |k: usize| crs_pair(&lonlat, &laea[k]).map(|(_, f)| f);
    let (south, north) = (to_world(0).unwrap(), to_world(1).unwrap());
    let bands = [Band { rows: 0..EQUATOR, to_world: &*south }, Band { rows: EQUATOR..NR, to_world: &*north }];
    // the backend's world is the analytic one, to rounding
    let (x, y) = north.convert(30.0, 45.0).unwrap();
    let want = laea_n(30.0, 45.0);
    assert!((x - want[0]).abs() < 1e-6 && (y - want[1]).abs() < 1e-6, "({x}, {y}) against {want:?}");
    let land = land();
    let (xb, yb) = bounds(&MOVED);
    let s = MeshSet::from_cf_bounds(NC, NR, &xb, &yb, Halo::NONE, &bands, Some(&land)).unwrap();
    let poly = |b| Geometry::Polygon(Polygon::new(vec![lonlat_box(b)]));
    let geoms = vec![
        poly([-180.0, 180.0, -90.0, 90.0]),
        poly([-180.0, 180.0, 60.0, 90.0]),
        poly([-127.5, 232.5, -90.0, -48.0]),
        Geometry::MultiPolygon(vec![
            Polygon::new(vec![lonlat_box([150.0, 180.0, -60.0, 30.0])]),
            Polygon::new(vec![lonlat_box([-180.0, -150.0, -60.0, 30.0])]),
        ]),
    ];
    let one = s.burn(&geoms, &bands, 0.1, BurnMode::Coverage).unwrap();
    let whole: f64 = fractions(&one, 0).values().sum();
    assert!(whole > 1000.0, "{whole}");
    for threads in [0, 3] {
        let par = s.burn_par_with(&geoms, &bands, 0.1, BurnMode::Coverage, threads, to_world);
        assert_eq!(one, par.unwrap(), "{threads} threads");
    }
}

/// Points are found in the cell that holds them, in either band and in
/// the cap; points in a cell left out, in the gap where the bands meet,
/// or off the grid are found nowhere.
#[test]
fn burn_points() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let dropped: HashSet<_> = s.dropped().iter().copied().collect();
    let centre = |r: usize, c: usize| Coord { x: 7.5 * (c as f64 + 0.5), y: -75.0 + 6.0 * r as f64 };
    let pts = vec![
        centre(1, 3),
        centre(EQUATOR - 1, 40),
        centre(EQUATOR, 5),
        Coord { x: 100.0 - 360.0, y: 20.0 },
        // on the equator half way between corners: in the gap
        Coord { x: 3.75, y: 0.0 },
        Coord { x: 0.0, y: -85.0 },
        Coord { x: 10.0, y: 90.0 },
    ];
    // every cell left out, at its own corners' centre: in no cell left out
    let gone: Vec<Coord> = s.dropped().iter().map(|&(r, c)| centre(r, c)).collect();
    let geoms = vec![Geometry::MultiPoint(pts.clone()), Geometry::Point(pts[1]), Geometry::MultiPoint(gone)];
    let r = burn(&s, &geoms).unwrap();
    let found: Vec<(i32, i32, i32)> = r.points.iter().filter(|p| p.id < 2).map(|p| (p.id, p.row, p.col)).collect();
    eprintln!("{found:?}");
    let e = EQUATOR as i32;
    assert_eq!(&found[..4], &[(0, 1, 3), (0, e - 1, 40), (0, e, 5), (0, 16, 13)]);
    // not the gap, nor south of the grid; the pole in one of the cap's cells
    assert_eq!(found.len(), 6, "{found:?}");
    assert!(found[4].0 == 0 && found[4].1 >= NREG as i32);
    assert_eq!(found[5], (1, e - 1, 40));
    assert!(r.points.iter().all(|p| !dropped.contains(&(p.row as usize, p.col as usize))));
    assert!(r.runs.is_empty() && r.edges.is_empty());
}

/// What a burn onto a set refuses: lines, bands other than the set's,
/// a tolerance that isn't positive, and modes other than coverage.
#[test]
fn burn_refusals() {
    let s = set(&[], None).unwrap();
    let (south, north) = (World(0, false), World(1, false));
    let bands = [Band { rows: 0..EQUATOR, to_world: &south }, Band { rows: EQUATOR..NR, to_world: &north }];
    let poly = vec![Geometry::Polygon(Polygon::new(vec![lonlat_box([0.0, 30.0, 0.0, 30.0])]))];
    let line = vec![poly[0].clone(), Geometry::LineString(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 1.0, y: 1.0 }])];
    let e = s.burn(&line, &bands, 1.0, BurnMode::Coverage).unwrap_err();
    assert!(e.contains("geometry 1") && e.contains("lines"), "{e}");
    let one = [Band { rows: 0..NR, to_world: &north }];
    assert!(s.burn(&poly, &one, 1.0, BurnMode::Coverage).unwrap_err().contains("the bands must be"));
    let moved = [Band { rows: 0..EQUATOR + 1, to_world: &south }, Band { rows: EQUATOR + 1..NR, to_world: &north }];
    assert!(s.burn(&poly, &moved, 1.0, BurnMode::Coverage).is_err());
    for tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(s.burn(&poly, &bands, tol, BurnMode::Coverage).unwrap_err().contains("tolerance"));
    }
    assert!(s.burn(&poly, &bands, 1.0, BurnMode::Approx).is_err());
    assert!(s.burn(&poly, &bands, 1.0, BurnMode::Coverage).is_ok());
}


// ---------------------------------------------------------------------------
// warp() from the set: SetTransformer
// ---------------------------------------------------------------------------

/// The set as a warp source onto a lon/lat grid `gt`.
fn set_transformer(s: &MeshSet, gt: [f64; 6]) -> SetTransformer<'_> {
    let crs: Vec<(Box<dyn CrsTransform>, Box<dyn CrsTransform>)> = vec![
        (Box::new(World(0, false)), Box::new(World(0, true))),
        (Box::new(World(1, false)), Box::new(World(1, true))),
    ];
    SetTransformer::with_backend(s, gt, crs).unwrap()
}

/// The largest difference between two tables' fractions, an entry in one
/// only counting whole.
fn max_diff(a: &HashMap<Key, f64>, b: &HashMap<Key, f64>) -> f64 {
    a.iter()
        .map(|(k, v)| (v - b.get(k).copied().unwrap_or(0.0)).abs())
        .chain(b.iter().filter(|(k, _)| !a.contains_key(k)).map(|(_, v)| v.abs()))
        .fold(0.0, f64::max)
}

/// `exact_weights` through the set's transformer are `weights_to`'s: the
/// same cells walked in the same worlds, outlines from every band in one
/// table. Source windows of it tile it.
#[test]
fn exact_weights_from_the_set_are_weights_to() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let t = set_transformer(&s, GT);
    for densify in [1, 4] {
        let w = exact_weights(&t, &t.source_window(), GW, GH, &WeightOptions::new(densify));
        let want = to_grid(&s, densify);
        let d = max_diff(&entries(&w), &entries(&want));
        // each target pixel's area, from the band it covers most of
        let mut darea = 0.0f64;
        for k in 0..GW * GH {
            let (a, b) = (w.dst_area[k], want.dst_area[k]);
            if a.is_finite() && b.is_finite() {
                darea = darea.max((a - b).abs() / b);
            }
        }
        eprintln!("densify {densify}: {} entries ({} from weights_to), fractions to {d:.1e}, target areas to {darea:.1e}", w.len(), want.len());
        assert!(d < 1e-12, "{d:e}");
        assert!(darea < 1e-9, "{darea:e}");
        assert!(w.src_cell_area.len() == w.len() && w.src_cell_area.iter().all(|a| *a > 0.0));
    }
    // 2 x 2 source windows, split across the bands' meeting and a column
    let whole = entries(&exact_weights(&t, &t.source_window(), GW, GH, &WeightOptions::new(2)));
    let mut tiled = HashMap::new();
    for (c0, c1) in [(0, 20), (20, NC)] {
        for (r0, r1) in [(0, EQUATOR + 2), (EQUATOR + 2, NR)] {
            let src = GridWindow::new(c1 - c0, r1 - r0).with_offset(c0, r0);
            let w = exact_weights(&t, &src, GW, GH, &WeightOptions::new(2));
            assert!(w.src_col.iter().all(|&c| (c0..c1).contains(&(c as usize))));
            assert!(w.src_row.iter().all(|&r| (r0..r1).contains(&(r as usize))));
            tiled.extend(entries(&w));
        }
    }
    let d = max_diff(&whole, &tiled);
    eprintln!("{} entries whole, {} in windows, to {d:.1e}", whole.len(), tiled.len());
    assert!(d < 1e-12, "{d:e}");
}

/// `warp()`'s `Average` from the set is `apply(FracArea)` of its
/// warp-compatible weights, bit for bit, and a constant field stays
/// constant; `Mode` of a field of bands takes the band covering most.
#[test]
fn warp_average_from_the_set_is_apply() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let t = set_transformer(&s, GT);
    let data: Vec<f64> = (0..NC * NR).map(|k| 10.0 + ((k % NC) as f64 * 0.4).sin() * ((k / NC) as f64 * 0.3).cos()).collect();
    let buf = SourceBuffer { data: &data, ncol: NC, nrow: NR, nbands: 1, col_off: 0, row_off: 0 };
    for densify in [1, 3] {
        let opts = WarpOptions::new(ResampleAlg::Average).with_footprint(Footprint::Exact { densify });
        let warped = warp_to::<f64, f64>(&t, &buf, GW, GH, &opts);
        let w = exact_weights(&t, &GridWindow::from(&buf), GW, GH, &WeightOptions::warp_compatible(densify));
        let applied: Vec<f64> = w.apply(&buf, Normalization::FracArea, &ApplyOptions::from(&opts));
        let differ = warped.iter().zip(&applied).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let computed = warped.iter().filter(|v| (5.0..15.0).contains(*v)).count();
        eprintln!("densify {densify}: {computed} pixels computed, {differ} differ");
        assert_eq!(differ, 0);
        assert!(computed > GW * GH * 9 / 10, "{computed}");
    }
    // a constant: every pixel the set reaches, to rounding
    let one = vec![1.0f64; NC * NR];
    let buf1 = SourceBuffer { data: &one, ..buf };
    let opts = WarpOptions::new(ResampleAlg::Average).with_footprint(Footprint::Exact { densify: 2 }).with_dst_nodata(DstNodata::Value(-1.0));
    let out = warp_to::<f64, f64>(&t, &buf1, GW, GH, &opts);
    let worst = out.iter().filter(|v| **v != -1.0).map(|v| (v - 1.0).abs()).fold(0.0, f64::max);
    assert!(worst < 1e-14, "{worst:e}");
    // Mode of each cell's band: a pixel across the equator takes the band
    // it covers most, the others their own
    let bands: Vec<f64> = (0..NC * NR).map(|k| ((k / NC) >= EQUATOR) as u8 as f64).collect();
    let bufb = SourceBuffer { data: &bands, ..buf };
    let opts = WarpOptions::new(ResampleAlg::Mode).with_footprint(Footprint::Exact { densify: 2 });
    let out = warp_to::<f64, f64>(&t, &bufb, GW, GH, &opts);
    // 2.5 degree rows: 35 is 2.5 N to 0, 36 is 0 to 2.5 S
    assert!(out[30 * GW..36 * GW].iter().all(|&v| v == 1.0));
    assert!(out[36 * GW..60 * GW].iter().all(|&v| v == 0.0));
}

/// Point kernels land in the cell that holds the pixel's centre, and read
/// their neighbours across the set's index space: bilinear over a field
/// linear in the cells' indices is linear, across pieces and bands. Cells
/// left out hold no point.
#[test]
fn point_kernels_read_across_pieces() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let t = set_transformer(&s, GT);
    let dropped: HashSet<_> = s.dropped().iter().copied().collect();
    let near = |r: usize, c: usize| MOVED.iter().any(|&((j, i), _)| (r as i64 - j as i64).abs() <= 3 && (c as i64 - i as i64).abs() <= 3);
    // nearest of each cell's number: the cell in the regular rows from
    // the pixel centre's longitude and latitude, away from the damage
    let ids: Vec<f64> = (0..NC * NR).map(|k| k as f64).collect();
    let buf = SourceBuffer { data: &ids, ncol: NC, nrow: NR, nbands: 1, col_off: 0, row_off: 0 };
    let opts = WarpOptions::new(ResampleAlg::NearestNeighbour).with_dst_nodata(DstNodata::Value(-1.0));
    let out = warp_to::<f64, f64>(&t, &buf, GW, GH, &opts);
    let (mut checked, mut gone) = (0, 0);
    for j in 0..GH {
        for i in 0..GW {
            let (lon, lat) = ((-178.75 + 2.5 * i as f64).rem_euclid(360.0), 88.75 - 2.5 * j as f64);
            if !(-78.0..60.0).contains(&lat) {
                continue;
            }
            let (r, c) = (((lat + 78.0) / 6.0) as usize, (lon / 7.5) as usize);
            if near(r, c) {
                continue;
            }
            let v = out[j * GW + i];
            if dropped.contains(&(r, c)) {
                assert_eq!(v, -1.0, "({r}, {c}) is left out");
                gone += 1;
            } else {
                assert_eq!(v, (r * NC + c) as f64, "pixel ({j}, {i}) at ({lon}, {lat})");
                checked += 1;
            }
        }
    }
    eprintln!("nearest: {checked} pixels in their cells, {gone} in cells left out");
    assert!(checked > 5000);
    // no point lands in a cell left out, on a finer grid over the damage
    let (mut x, mut y): (Vec<f64>, Vec<f64>) = (0..400 * 200).map(|k| ((k % 400) as f64 + 0.5, (k / 400) as f64 + 0.5)).unzip();
    let fine = [-180.0, 0.9, 0.0, 90.0, 0.0, -0.9];
    let tf = set_transformer(&s, fine);
    let ok = tf.transform(true, &mut x, &mut y);
    let landed: HashSet<(usize, usize)> = (0..x.len()).filter(|&k| ok[k]).map(|k| (y[k] as usize, x[k] as usize)).collect();
    let bad: Vec<_> = landed.iter().filter(|c| dropped.contains(c)).collect();
    eprintln!("{} cells hold a point of the fine grid; in cells left out: {bad:?}", landed.len());
    assert!(landed.len() > NC * NR * 3 / 4, "{}", landed.len());
    assert!(bad.is_empty());
    let moved_cells = dropped.iter().filter(|&&(r, c)| near(r, c)).count();
    assert!(moved_cells > 0);
    // bilinear of 3 col + 5 row is 3 (x - 1/2) + 5 (y - 1/2) at the centre's
    // place (x, y) in the set's index space
    let lin: Vec<f64> = (0..NC * NR).map(|k| 3.0 * (k % NC) as f64 + 5.0 * (k / NC) as f64).collect();
    let buf = SourceBuffer { data: &lin, ..buf };
    let opts = WarpOptions::new(ResampleAlg::Bilinear).with_dst_nodata(DstNodata::Value(-1.0));
    let out = warp_to::<f64, f64>(&t, &buf, GW, GH, &opts);
    let piece = |r: usize, c: usize| s.locate_cell(r, c).map(|(p, _, _)| p.offset());
    let (mut across, mut bands, mut worst) = (0, 0, 0.0f64);
    for j in 0..GH {
        for i in 0..GW {
            let (mut x, mut y) = ([i as f64 + 0.5], [j as f64 + 0.5]);
            if !t.transform(true, &mut x, &mut y)[0] {
                continue;
            }
            let (bx, by) = (x[0] - 0.5, y[0] - 0.5);
            if !(bx >= 1.0 && by >= 0.0 && bx <= (NC - 2) as f64 && by <= (NR - 1) as f64) {
                continue;
            }
            let (c, r) = (bx as usize, by as usize);
            let four = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)];
            if four.iter().any(|&(r, c)| r >= NR || dropped.contains(&(r, c))) || r >= NREG {
                continue;
            }
            let ps: HashSet<_> = four.iter().map(|&(r, c)| piece(r, c)).collect();
            if ps.len() > 1 {
                across += 1;
                bands += (r + 1 == EQUATOR) as usize;
            }
            let d = (out[j * GW + i] - (3.0 * bx + 5.0 * by)).abs();
            worst = worst.max(d);
        }
    }
    eprintln!("bilinear: linear to {worst:.1e}, {across} pixels read more than one piece, {bands} both bands");
    assert!(worst < 1e-9, "{worst:e}");
    assert!(across > 50 && bands > 50, "{across}, {bands}");
}

/// A south-polar target round the South Pole: its outline there goes to
/// the north band's edge and round the whole band, turned the other way,
/// and is not measured there. Every pixel takes the south band only.
#[test]
fn outlines_round_the_far_pole_are_not_turned_inside_out() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    // the target in south-polar LAEA, band 0's world: band 1 through lon/lat
    struct Then(World, World);
    impl CrsTransform for Then {
        fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
            let (a, b) = self.0.convert(x, y)?;
            self.1.convert(a, b)
        }
    }
    struct Same;
    impl CrsTransform for Same {
        fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
            Some((x, y))
        }
    }
    let crs: Vec<(Box<dyn CrsTransform>, Box<dyn CrsTransform>)> = vec![
        (Box::new(Same), Box::new(Same)),
        (Box::new(Then(World(0, true), World(1, false))), Box::new(Then(World(1, true), World(0, false)))),
    ];
    // 9 x 9 pixels of 600 km, the pole at the centre of the middle one
    let gt = [-2700e3, 600e3, 0.0, 2700e3, 0.0, -600e3];
    let t = SetTransformer::with_backend(&s, gt, crs).unwrap();
    let bands: Vec<f64> = (0..NC * NR).map(|k| ((k / NC) >= EQUATOR) as u8 as f64).collect();
    let buf = SourceBuffer { data: &bands, ncol: NC, nrow: NR, nbands: 1, col_off: 0, row_off: 0 };
    for alg in [ResampleAlg::Average, ResampleAlg::Max] {
        let opts = WarpOptions::new(alg).with_footprint(Footprint::Exact { densify: 8 }).with_dst_nodata(DstNodata::Value(-1.0));
        let out = warp_to::<f64, f64>(&t, &buf, 9, 9, &opts);
        eprintln!("{alg:?}: {out:?}");
        // 78 S is 1332 km from the pole: the middle pixel reaches no cell
        assert_eq!(out[4 * 9 + 4], -1.0);
        assert!(out.iter().all(|&v| v == -1.0 || v == 0.0), "{out:?}");
        assert!(out.iter().filter(|&&v| v == 0.0).count() >= 40);
    }
}

/// A set of one band in one piece warps as its mesh does through
/// `MeshTransformer`, bit for bit, for area methods and point kernels.
#[test]
fn a_set_of_one_piece_is_its_mesh() {
    // 12 columns of 30 degrees from the equator to the pole, north-polar
    // LAEA on the sphere
    let (nc, nr) = (12, 3);
    let (mut xb, mut yb) = (Vec::new(), Vec::new());
    for j in 0..nr {
        for i in 0..nc {
            for (di, dj) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
                xb.push(-180.0 + 30.0 * (i + di) as f64);
                yb.push(30.0 * (j + dj) as f64);
            }
        }
    }
    let north = World(1, false);
    let bands = [Band { rows: 0..nr, to_world: &north }];
    let s = MeshSet::from_cf_bounds(nc, nr, &xb, &yb, Halo::NONE, &bands, None).unwrap();
    assert_eq!(s.pieces().len(), 1);
    let gt = [-180.0, 5.0, 0.0, 90.0, 0.0, -5.0];
    let crs: Vec<(Box<dyn CrsTransform>, Box<dyn CrsTransform>)> = vec![(Box::new(World(1, false)), Box::new(World(1, true)))];
    let set_t = SetTransformer::with_backend(&s, gt, crs).unwrap();
    let mesh_t = MeshTransformer::with_backend(s.pieces()[0].mesh().clone(), gt, Box::new(World(1, false)), Box::new(World(1, true))).unwrap();
    let data: Vec<f64> = (0..nc * nr).map(|k| ((k * 7) % 11) as f64 + 0.5 * (k as f64).sin()).collect();
    let buf = SourceBuffer { data: &data, ncol: nc, nrow: nr, nbands: 1, col_off: 0, row_off: 0 };
    for alg in [ResampleAlg::Average, ResampleAlg::Mode, ResampleAlg::Max, ResampleAlg::Bilinear, ResampleAlg::Cubic, ResampleAlg::NearestNeighbour] {
        let opts = WarpOptions::new(alg).with_footprint(Footprint::Exact { densify: 3 });
        let a = warp_to::<f64, f64>(&set_t, &buf, 72, 18, &opts);
        let b = warp_to::<f64, f64>(&mesh_t, &buf, 72, 18, &opts);
        let differ = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let computed = a.iter().filter(|v| **v != 0.0).count();
        eprintln!("{alg:?}: {computed} pixels computed, {differ} differ");
        assert_eq!(differ, 0, "{alg:?}");
        assert!(computed > 72 * 18 / 3, "{alg:?}: {computed}");
    }
}

/// On several threads, each with a transformer of its own over the set:
/// `warp_to`'s values bit for bit, area methods and point kernels, and
/// the target round the far pole too.
#[test]
fn warp_par_from_the_set_is_warp() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let t = set_transformer(&s, GT);
    let data: Vec<f64> = (0..NC * NR).map(|k| 10.0 + ((k % NC) as f64 * 0.4).sin() * ((k / NC) as f64 * 0.3).cos()).collect();
    let buf = SourceBuffer { data: &data, ncol: NC, nrow: NR, nbands: 1, col_off: 0, row_off: 0 };
    for alg in [ResampleAlg::Average, ResampleAlg::Mode, ResampleAlg::Max, ResampleAlg::Bilinear, ResampleAlg::NearestNeighbour] {
        let opts = WarpOptions::new(alg).with_footprint(Footprint::Exact { densify: 2 }).with_dst_nodata(DstNodata::Value(-1.0));
        let one = warp_to::<f64, f64>(&t, &buf, GW, GH, &opts);
        for threads in [0, 3] {
            let par = warp_to_par::<f64, f64, _>(|| Ok(set_transformer(&s, GT)), &buf, GW, GH, &opts, threads).unwrap();
            let differ = one.iter().zip(&par).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(differ, 0, "{alg:?}, {threads} threads");
        }
    }
}

/// `exact_weights_par` through a `SetTransformer` a thread: `exact_weights`'
/// table, entry for entry and bit for bit.
#[test]
fn exact_weights_par_from_the_set() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let t = set_transformer(&s, GT);
    let opts = WeightOptions::new(3);
    let one = exact_weights(&t, &t.source_window(), GW, GH, &opts);
    for threads in [0, 3] {
        let par = exact_weights_par(|| Ok(set_transformer(&s, GT)), &t.source_window(), GW, GH, &opts, threads).unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert!(one.dst_row == par.dst_row && one.dst_col == par.dst_col && one.src_row == par.src_row && one.src_col == par.src_col);
        assert!(bits(&one.frac) == bits(&par.frac) && bits(&one.dst_area) == bits(&par.dst_area) && bits(&one.src_cell_area) == bits(&par.src_cell_area));
    }
}

/// `Footprint::Adaptive`'s refinement from the set, each world refining
/// its own outlines with the tolerance in its metres. `exact_weights`
/// through the set's transformer are still `weights_to`'s; at 10 m they
/// are 1.6e-5 from densify 64's, where densify 4 is 6.3e-5 (both worst
/// in the cap's small cells round the pole, which get a tolerance in
/// metres as a larger fraction of themselves: 2.5e-4 at 100 m); on
/// threads bit for bit; and `warp()` is `apply` of them, on threads too.
#[test]
fn adaptive_edges_from_the_set() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let t = set_transformer(&s, GT);
    let src = t.source_window();
    let opts = WeightOptions::new(1).with_refine_tolerance(10.0);
    let w = exact_weights(&t, &src, GW, GH, &opts);
    let want = s
        .weights_to(|w, mesh| MeshTransformer::with_backend(mesh, GT, Box::new(World(w, false)), Box::new(World(w, true))), GW, GH, &opts)
        .unwrap();
    let d = max_diff(&entries(&w), &entries(&want));
    let fine = entries(&exact_weights(&t, &src, GW, GH, &WeightOptions::new(64)));
    let (e, e4) = (max_diff(&entries(&w), &fine), max_diff(&entries(&exact_weights(&t, &src, GW, GH, &WeightOptions::new(4))), &fine));
    eprintln!("10 m: weights_to's to {d:.1e}; densify 64's to {e:.1e} (densify 4 to {e4:.1e})");
    assert!(d < 1e-12, "{d:e}");
    assert!(e < 3e-5 && e4 > 3e-5, "{e:e} {e4:e}");
    for threads in [0, 3] {
        let par = exact_weights_par(|| Ok(set_transformer(&s, GT)), &src, GW, GH, &opts, threads).unwrap();
        assert!(same_weights(&w, &par), "{threads} threads");
    }
    let data: Vec<f64> = (0..NC * NR).map(|k| 10.0 + ((k % NC) as f64 * 0.4).sin() * ((k / NC) as f64 * 0.3).cos()).collect();
    let buf = SourceBuffer { data: &data, ncol: NC, nrow: NR, nbands: 1, col_off: 0, row_off: 0 };
    let wopts = WarpOptions::new(ResampleAlg::Average).with_footprint(Footprint::Adaptive { densify: 1, tolerance: 10.0 });
    let warped = warp_to::<f64, f64>(&t, &buf, GW, GH, &wopts);
    let applied: Vec<f64> = exact_weights(&t, &GridWindow::from(&buf), GW, GH, &WeightOptions::warp_compatible(1).with_refine_tolerance(10.0))
        .apply(&buf, Normalization::FracArea, &ApplyOptions::from(&wopts));
    let computed = warped.iter().filter(|v| (5.0..15.0).contains(*v)).count();
    assert!(computed > GW * GH * 9 / 10, "{computed}");
    assert!(warped.iter().zip(&applied).all(|(a, b)| a.to_bits() == b.to_bits()));
    let par = warp_to_par::<f64, f64, _>(|| Ok(set_transformer(&s, GT)), &buf, GW, GH, &wopts, 3).unwrap();
    assert!(warped.iter().zip(&par).all(|(a, b)| a.to_bits() == b.to_bits()));
}

/// The same table bit for bit, NaN areas included.
fn same_weights(a: &Weights, b: &Weights) -> bool {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    (a.dst_row == b.dst_row && a.dst_col == b.dst_col && a.src_row == b.src_row && a.src_col == b.src_col)
        && bits(&a.frac) == bits(&b.frac)
        && bits(&a.dst_area) == bits(&b.dst_area)
        && bits(&a.src_cell_area) == bits(&b.src_cell_area)
        && (a.dst_off, a.dst_size) == (b.dst_off, b.dst_size)
}

/// `weights_to_par` and `weights_onto_par` give `weights_to`'s and
/// `weights_onto`'s tables bit for bit, on thread counts that measure the
/// bands' main pieces on every thread and the strips one a thread; the
/// first piece's error comes back.
#[test]
fn set_weights_on_threads() {
    let land = land();
    let s = set(&MOVED, Some(&land)).unwrap();
    let mt = |w: usize, mesh: QuadMesh| MeshTransformer::with_backend(mesh, GT, Box::new(World(w, false)), Box::new(World(w, true)));
    let gw = GridWindow::new(GW, GH);
    for densify in [1, 4] {
        let opts = WeightOptions::new(densify);
        let to = s.weights_to(mt, GW, GH, &opts).unwrap();
        let onto_opts = opts.with_measure(lonlat_measure(GT).unwrap());
        let inner = |w: usize| {
            GenImgProjTransformer::with_backend(GT, WORLD_GT, Box::new(World(w, true)), Box::new(World(w, false)))
                .map(|t| t.with_center_long(CenterLong::At(0.0)).with_source_size([GW, GH]))
        };
        let onto = s.weights_onto(inner, &gw, &onto_opts).unwrap();
        for threads in [0, 2, 3, 8] {
            assert!(same_weights(&to, &s.weights_to_par(mt, GW, GH, &opts, threads).unwrap()), "to, densify {densify}, {threads} threads");
            assert!(same_weights(&onto, &s.weights_onto_par(inner, &gw, &onto_opts, threads).unwrap()), "onto, densify {densify}, {threads} threads");
        }
    }
    // a transformer that fails for band 1: its first piece's error
    let fails = |w: usize, mesh: QuadMesh| if w == 1 { Err(format!("band {w}")) } else { mt(w, mesh) };
    assert_eq!(s.weights_to(fails, GW, GH, &WeightOptions::new(1)).unwrap_err(), "band 1");
    assert_eq!(s.weights_to_par(fails, GW, GH, &WeightOptions::new(1), 3).unwrap_err(), "band 1");
}

