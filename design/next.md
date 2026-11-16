# State and what's next (2026-11-12)

Where rwarp stands at the end of the 2026-11-05 to 2026-11-12 session,
and what to pick up. The long-term framing is in #52 and on the roadmap
board (<https://github.com/orgs/hypertidy/projects/1>). The open
decisions, with options and figures, are in [decisions.md](decisions.md).
This file is a snapshot, so replace it rather than append to it.

## Released

| crate | version | date | what |
|---|---|---|---|
| rwarp | 0.3.3 | 2026-11-12 | antimeridian split reads two edge windows, not the whole width; `merge_tile` |
| rwarp | 0.4.0 | 2026-11-12 | controlledburn 0.3.1 (breaking for rwarp: its types are in rwarp's API); N6 recorded |
| rwarp | 0.4.1 | 2026-11-16 | `QuadMesh::burn_par`, `Rectilinear::burn_par` (#59, #63) |
| controlledburn | 0.3.2 | 2026-11-12 | `burn_wkb_par`; burns the same as 0.3.1 |
| controlledburn | 0.3.1 | 2026-11-12 | `relate_cells` (0.3.0: `burn_par`, `burn_window`, `merge_parts`, `f32` edges kept) |
| bigcurve (crate) | 0.3.0 | 2026-11-04 | |

`v0.4.0` (on `6a4c337`) and `v0.4.1` (on `e4c3e49`) were tagged on
2026-11-16, each on the commit its crate was published from.

## Open

- **#59, `burn_par` for `QuadMesh` and `Rectilinear`**, merged
  2026-11-13 and not yet released; it goes in 0.4.1. Both stages run in
  parallel: the ring walk into index space, on chunks that threads take
  in turn, and controlledburn's `burn_par`. Results are record for
  record the same as `burn`. On Natural Earth 10m land onto a 1440 x 720
  mesh: 1.16 s serial, 0.31 s on 4 threads, 0.24 s on 8. It goes no
  faster after that, because the largest polygon takes 0.33 s alone and
  one geometry is never split. With one thread it stays on the calling
  thread, so it works on wasm.
- **#60**, merged 2026-11-16, moved rwarp-r's `Cargo.lock` to rwarp
  0.3.3. Main's R-CMD-check had failed since 2026-11-08 because the lock
  pinned 0.3.2 while `Cargo.toml` asked for 0.3.3, and
  `cargo vendor --locked` refuses that. `tools/vendor.sh` used to discard
  cargo's error output, so the log showed only exit code 101. Now it
  keeps the errors. Main's R-CMD-check passes again.
- **#61**, merged 2026-11-16, and its follow-up: softer landing pages for
  the book and the R README (the usual tools first, then rwarp's
  equivalents, then when you might reach for rwarp), and stale pages
  fixed.

## Next, in order

1. **rwarp-r is on rwarp 0.4.1** (with controlledburn 0.3.2, #64). For
   later bumps: change `rwarp-r/src/rust/Cargo.toml`, run
   `cargo update -p rwarp`, and commit the lock with it. The lock and the
   manifest must move together, or `vendor.sh` fails as above.
2. **The rest of controlledburn 0.3:**
   - `burn_window` could replace rwarp's own windowed burn in
     `src/cutline.rs` (`Cutline::burn_window`, which builds a grid for
     the window). Check that the records are identical first.
   - `burn_window` could also split one large geometry into tiles, which
     is the remaining limit on `burn_par`.
   - `relate_cells` finds cells a polygon only touches along a wall or
     at a corner, which `burn` gives no record for. GDAL parity doesn't
     need it; it would help explain cutline edge differences, if any
     turn up.
3. **The stages on the board**: curvilinear stage 3 (#36: exact
   footprint for geolocation sources, antimeridian), then stage 4 (#37:
   periodic and tripolar grids and CF bounds, which takes the ESMF
   comparison onto ocean grids). N1 (#43) is decided (coverage uses
   `EqualArea` on every mesh). #36 is built in two PRs: the antimeridian
   and pole handling of the exact footprint (#70), then the equal-area
   footprint for geolocation sources (on by default, falling back where
   the arrays' cells are not whole). Geolocation sources across the
   antimeridian are stage 3b, their own issue. #37 also waits on N3 and
   N4.
4. **Decisions waiting on the maintainer**: N2 to N5 (#44 to #47) and C1
   to C4 (#48 to #51). N1 (2026-11-16) and N6 (`f32` edges,
   controlledburn 0.3.0) are decided.
5. **rwarp-r**: `warp_buffer()` binds `warp()` for one pre-fetched
   window (#54, #65), and `warp_resample()`, which still calls the
   deprecated `warp::warp_resample` shim, is deprecated. The Sentinel-2
   showcase (#55) is the `sentinel2-warp` vignette: two UTM zones, three
   float bands, onto EPSG:3577, equal to gdalwarp's mosaic to 1e-11.
   It found that R's `warp_plan()` couldn't describe gdalwarp's working
   memory, so for several Float64 bands with nodata its chunks, and so
   its kernel scales, differed from gdalwarp's (93 DN at 50 m). #67 adds
   `nbands`, `working_type`, `src_nodata`, `dst_nodata`, `memory_limit`
   and `scale_origin`: with gdalwarp's rules the chunks are gdalwarp's
   (2 and 2 at 100 m, 15 and 19 at 50 m) and the vignette matches without
   `-wm`.

## Parked

- **CRAN for rwarp-r** (#42): not being targeted for now (2026-11-12).
  The build layout is ready (#32): vendored crates, an offline build,
  and a PROJ >= 9.6.2 configure check. The Windows result from
  win-builder was never reviewed.
- **Upstreaming the GDAL findings** (#34): parked until the PSC business
  settles.
- **Great-circle cell edges** (#41): resumes with meshes and grids
  defined by great circles, through bigcurve.
- **garry** (Hugh Graham's belian-earth/garry): the MUR demo is merged
  in wxylib/refharvest (`demo/garry-mur`, PR #6). Running the garry
  steps needs garry 0.2.0. The issue for garry is drafted and not
  posted. One finding belongs to refharvest's COG stage (its issue #5):
  MUR rewrites recent days, so a refs table goes stale unless it
  records each object's ETag or size.

## Gotchas

- proj-sys 0.27 needs a system PROJ >= 9.6.2. Below that it silently
  builds a bundled static PROJ, which leaves sqlite3 symbols undefined.
  rwarp-r's CI runs in `ghcr.io/hypertidy/gdal-r-python-extras-rust`
  for that reason.
- R-CMD-check only runs for changes under `rwarp-r/`. A crate release
  that rwarp-r depends on doesn't trigger it, so rerun it by hand.
- Main's version leads the release. After publishing, the next
  unreleased version and its changelog section go in with the next PR.
