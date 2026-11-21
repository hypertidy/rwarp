//! View plans against the cases in `tests/fixtures/view/cases.json`
//! (pole, seam, horizon, small and whole-world views; a polar source on
//! the globe), which any other planner can run too.
//!
//! Each case is checked two ways: the plan's counts against the fixture,
//! and the plan against the screen, pixel by pixel, without the planner's
//! own rule: every visible pixel must find its source cell in a planned
//! chunk no coarser than its tile's level.
//!
//! `RWARP_VIEW_PRINT=1 cargo test --test view -- --nocapture` prints the
//! `expect` blocks for the cases as they are planned now.

#![cfg(any(feature = "proj", feature = "proj4rs"))]

use std::collections::BTreeSet;

use rwarp::crs::{crs_pair, CrsTransform};
use rwarp::view::{plan_single_box, plan_view, ChunkKey, Level, Source, View, ViewOptions, ViewPlan};
use serde_json::Value;

struct Same;
impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

fn nums<const N: usize>(v: &Value) -> [f64; N] {
    let a = v.as_array().expect("an array");
    assert_eq!(a.len(), N);
    std::array::from_fn(|i| a[i].as_f64().expect("a number"))
}

fn source(v: &Value) -> (String, Source) {
    let b = &v["base"];
    let size = nums::<2>(&b["size"]).map(|n| n as usize);
    let chunk = nums::<2>(&b["chunk"]).map(|n| n as usize);
    let levels = Level::new(nums::<6>(&b["gt"]), size, chunk).pyramid(v["levels"].as_u64().unwrap() as usize);
    let src = if v["geographic"].as_bool().unwrap() { Source::geographic(levels) } else { Source::new(levels) };
    (v["crs"].as_str().unwrap().to_string(), src)
}

type Pair = (Box<dyn CrsTransform>, Box<dyn CrsTransform>);

fn transforms(view_crs: &str, source_crs: &str) -> Pair {
    if view_crs == source_crs {
        return (Box::new(Same), Box::new(Same));
    }
    // (view to source, source to view)
    crs_pair(source_crs, view_crs).expect("a CRS pair")
}

fn summary(plan: &ViewPlan, src: &Source) -> Value {
    serde_json::json!({"chunks": plan.chunks.len(), "cells": plan.cells(src), "levels": plan.levels_used()})
}

/// Every visible pixel (every fourth each way) is in a planned chunk no
/// coarser than its tile's level. Returns the pixels checked.
fn check_screen(name: &str, view: &View, src: &Source, pair: &Pair, plan: &ViewPlan, tile: usize) -> usize {
    let have: BTreeSet<ChunkKey> = plan.keys().collect();
    let period = src.period;
    let mut checked = 0;
    let mut missed = 0;
    for py in (2..view.size[1]).step_by(4) {
        for px in (2..view.size[0]).step_by(4) {
            let (fx, fy) = (px as f64 + 0.5, py as f64 + 0.5);
            let (vx, vy) = view.at(fx, fy);
            let Some((x, y)) = pair.0.convert(vx, vy) else { continue };
            let Some((bx, by)) = pair.1.convert(x, y) else { continue };
            let (qx, qy) = view.screen(bx, by);
            if !((qx - fx).abs() <= 0.5 && (qy - fy).abs() <= 0.5) {
                continue;
            }
            let t = plan.tiles.iter().find(|t| t.col == px / tile && t.row == py / tile);
            // A pixel in no tile must be off the source's grid.
            let limit = t.map_or(f64::INFINITY, |t| src.levels[t.planned].cell() * (1.0 + 1e-9));
            let mut inside_any = false;
            let mut found = false;
            for (li, lev) in src.levels.iter().enumerate() {
                if lev.cell() > limit {
                    continue;
                }
                for k in [-1.0, 0.0, 1.0] {
                    let xs = x + k * period.unwrap_or(0.0);
                    let c = (xs - lev.gt[0]) / lev.gt[1];
                    let r = (y - lev.gt[3]) / lev.gt[5];
                    if c < 0.0 || r < 0.0 || c >= lev.size[0] as f64 || r >= lev.size[1] as f64 {
                        continue;
                    }
                    inside_any = true;
                    let key = ChunkKey { level: li, col: c as usize / lev.chunk[0], row: r as usize / lev.chunk[1] };
                    found |= have.contains(&key);
                    if period.is_none() {
                        break;
                    }
                }
            }
            if inside_any {
                checked += 1;
                if !found {
                    missed += 1;
                }
            }
        }
    }
    assert_eq!(missed, 0, "{name}: {missed} of {checked} visible pixels have no planned chunk");
    checked
}

#[test]
fn cases() {
    let text = include_str!("fixtures/view/cases.json");
    let doc: Value = serde_json::from_str(text).unwrap();
    assert_eq!(doc["format"], "view-plan-cases/1");
    let print = std::env::var_os("RWARP_VIEW_PRINT").is_some();
    let mut wrong = Vec::new();
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (source_crs, src) = source(&doc["sources"][case["source"].as_str().unwrap()]);
        let view_crs = case["view"]["crs"].as_str().unwrap();
        let view = View::new(nums::<4>(&case["view"]["extent"]), nums::<2>(&case["view"]["size"]).map(|n| n as usize));
        let pair = transforms(view_crs, &source_crs);
        let opts = ViewOptions::default();

        let per_tile = plan_view(&view, &src, &pair.0, Some(&pair.1), &opts);
        let all = plan_view(&view, &src, &pair.0, Some(&pair.1), &opts.clone().with_drop_covered(false));
        let single = plan_single_box(&view, &src, &pair.0, Some(&pair.1), 64, 0.7);

        let checked = check_screen(name, &view, &src, &pair, &per_tile, opts.tile);
        assert_eq!(check_screen(name, &view, &src, &pair, &all, opts.tile), checked);
        assert!(checked > 1000, "{name}: only {checked} pixels see the source");
        // Dropping covered chunks only takes chunks away.
        let keep: BTreeSet<ChunkKey> = all.keys().collect();
        assert!(per_tile.keys().all(|k| keep.contains(&k)), "{name}");
        assert_eq!(per_tile.tiles, all.tiles, "{name}");

        let got = serde_json::json!({
            "per_tile": summary(&per_tile, &src),
            "per_tile_all": summary(&all, &src),
            "single_box": summary(&single, &src),
            "pole_tiles": per_tile.tiles.iter().filter(|t| t.pole).count(),
        });
        if print {
            println!("{name}\n     \"expect\": {got}");
        } else if case["expect"] != got {
            wrong.push(format!("{name}\n  expected {}\n  got      {got}", case["expect"]));
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

/// How long a plan takes: `cargo test --release --test view -- --ignored --nocapture`.
#[test]
#[ignore]
fn timing() {
    let doc: Value = serde_json::from_str(include_str!("fixtures/view/cases.json")).unwrap();
    for case in doc["cases"].as_array().unwrap() {
        let (source_crs, src) = source(&doc["sources"][case["source"].as_str().unwrap()]);
        let view = View::new(nums::<4>(&case["view"]["extent"]), nums::<2>(&case["view"]["size"]).map(|n| n as usize));
        let pair = transforms(case["view"]["crs"].as_str().unwrap(), &source_crs);
        let t = std::time::Instant::now();
        for _ in 0..10 {
            plan_view(&view, &src, &pair.0, Some(&pair.1), &ViewOptions::default());
        }
        eprintln!("{:6.2} ms  {}", t.elapsed().as_secs_f64() * 100.0, case["name"].as_str().unwrap());
    }
}

/// The plan, the rule that needs no backend: with only the transform to
/// the source, the pole is found by its tile's spread of longitudes.
#[test]
fn pole_without_the_way_back() {
    let (_, src) = source(
        &serde_json::from_str::<Value>(include_str!("fixtures/view/cases.json")).unwrap()["sources"]["mur"],
    );
    let pair = transforms("EPSG:3031", "EPSG:4326");
    let view = View::new([-4e6, -4e6, 4e6, 4e6], [1024, 1024]);
    // Keep the pole tile's own level in the plan to look at.
    let opts = ViewOptions::default().with_drop_covered(false);
    let with = plan_view(&view, &src, &pair.0, Some(&pair.1), &opts);
    let without = plan_view(&view, &src, &pair.0, None, &opts);
    check_screen("pole, one way", &view, &src, &pair, &without, opts.tile);
    // The four tiles that meet at the pole each see every longitude.
    let lev = |p: &ViewPlan| p.tiles.iter().find(|t| t.col == 8 && t.row == 8).unwrap().level;
    assert_eq!(lev(&with), lev(&without));
    let row_full = |p: &ViewPlan| {
        let l = lev(p);
        let n = src.levels[l].chunks();
        (0..n[0]).all(|col| p.keys().any(|k| k == ChunkKey { level: l, col, row: n[1] - 1 }))
    };
    assert!(row_full(&with) && row_full(&without));
    assert_eq!(with.tiles.iter().filter(|t| t.pole).count(), 1);
}
