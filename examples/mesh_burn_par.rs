//! Time `QuadMesh::burn_par` against `QuadMesh::burn`: Natural Earth 10m
//! land, one polygon per part, onto a warped global lon/lat mesh.
//!
//! ```sh
//! cargo run --release --example mesh_burn_par -- land.wkb [step]
//! ```
//!
//! `land.wkb` is a u32 count, then each polygon as a u32 length and ISO
//! WKB (little endian). `step` is the mesh's cell size in degrees
//! (default 0.25).

use std::time::Instant;

use controlledburn::{parse_wkb, BurnOptions, Geometry};
use rwarp::mesh::{CellMap, QuadMesh};

fn read(path: &str) -> Vec<Geometry> {
    let b = std::fs::read(path).expect("read the WKB file");
    let u = |p: usize| u32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as usize;
    let mut pos = 4;
    (0..u(0))
        .map(|_| {
            let n = u(pos);
            pos += 4 + n;
            parse_wkb(&b[pos - n..pos]).expect("a WKB polygon")
        })
        .collect()
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let geoms = read(args.get(1).ok_or("usage: mesh_burn_par land.wkb [step]")?);
    let step: f64 = args.get(2).map_or(0.25, |s| s.parse().unwrap());
    let (nc, nr) = ((360.0 / step) as usize, (180.0 / step) as usize);
    // a smooth warp of the regular grid, a few tenths of a cell
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=nr {
        for i in 0..=nc {
            let (fi, fj) = (i as f64, j as f64);
            x.push(-180.0 + step * (fi + 0.3 * (fj * 0.05).sin() * (fi / nc as f64 * 3.1).sin()));
            y.push(90.0 - step * (fj + 0.3 * (fi * 0.04).sin() * (fj / nr as f64 * 3.1).sin()));
        }
    }
    let t = Instant::now();
    let m = QuadMesh::from_corners(nc, nr, x, y, CellMap::EqualArea)?;
    println!(
        "mesh {nc} x {nr}: {:.2} s; {} polygons",
        t.elapsed().as_secs_f64(),
        geoms.len()
    );
    // one geometry is never split: the largest bounds the speedup
    let nv = |g: &Geometry| match g {
        Geometry::Polygon(p) => p.rings.iter().map(Vec::len).sum::<usize>(),
        _ => 0,
    };
    let (big, all) = (
        geoms.iter().map(nv).max().unwrap_or(0),
        geoms.iter().map(nv).sum::<usize>(),
    );
    println!(
        "largest polygon: {big} of {all} vertices ({:.0}%)",
        100.0 * big as f64 / all as f64
    );
    let t = Instant::now();
    let big_g = geoms.iter().max_by_key(|g| nv(g)).unwrap().clone();
    m.burn(std::slice::from_ref(&big_g), BurnOptions::coverage())?;
    println!(
        "burn of the largest alone: {:.2} s",
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let one = m.burn(&geoms, BurnOptions::coverage())?;
    let t1 = t.elapsed().as_secs_f64();
    println!(
        "burn            {t1:6.2} s  ({} runs, {} edges)",
        one.runs.len(),
        one.edges.len()
    );
    for threads in [2, 4, 8, 16, 32] {
        let t = Instant::now();
        let par = m.burn_par(&geoms, BurnOptions::coverage(), threads)?;
        let tp = t.elapsed().as_secs_f64();
        assert!(
            par == one,
            "burn_par differs from burn at {threads} threads"
        );
        println!(
            "burn_par({threads:2})    {tp:6.2} s  {:5.1}x, same result",
            t1 / tp
        );
    }
    Ok(())
}
