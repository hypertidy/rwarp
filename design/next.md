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
| controlledburn | 0.3.1 | 2026-11-12 | `relate_cells` (0.3.0: `burn_par`, `burn_window`, `merge_parts`, `f32` edges kept) |
| bigcurve (crate) | 0.3.0 | 2026-11-04 | |

`v0.4.0` is not tagged yet: `git tag v0.4.0 6a4c337`, then push the tag.

## Open

- **#59, `burn_par` for `QuadMesh` and `Rectilinear`**, which goes in
  0.4.1. Both stages run in parallel: the ring walk into index space,
  on chunks that threads take in turn, and controlledburn's `burn_par`.
  Results are record for record the same as `burn`. On Natural Earth
  10m land onto a 1440 x 720 mesh: 1.16 s serial, 0.31 s on 4 threads,
  0.24 s on 8. It goes no faster after that, because the largest polygon
  takes 0.33 s alone and one geometry is never split. With one thread
  it stays on the calling thread, so it works on wasm.
- **This PR** moves rwarp-r's `Cargo.lock` to rwarp 0.3.3. Main's
  R-CMD-check has failed since 2026-11-08 because the lock pinned 0.3.2
  while `Cargo.toml` asked for 0.3.3, and `cargo vendor --locked`
  refuses that. `tools/vendor.sh` used to discard cargo's error output,
  so the log showed only exit code 101. Now it keeps the errors.

## Next, in order

1. Merge this PR and #59, tag `v0.4.0`, and publish 0.4.1 when ready.
2. **rwarp-r on rwarp 0.4.x**, when R should get controlledburn 0.3 or
   `burn_par`: bump `rwarp-r/src/rust/Cargo.toml`, run
   `cargo update -p rwarp`, and commit the lock. The lock and the
   manifest must move together, or `vendor.sh` fails as above.
3. **The rest of controlledburn 0.3:**
   - `burn_window` could replace rwarp's own windowed burn in
     `src/cutline.rs` (`Cutline::burn_window`, which builds a grid for
     the window). Check that the records are identical first.
   - `burn_window` could also split one large geometry into tiles, which
     is the remaining limit on `burn_par`.
   - `relate_cells` finds cells a polygon only touches along a wall or
     at a corner, which `burn` gives no record for. GDAL parity doesn't
     need it; it would help explain cutline edge differences, if any
     turn up.
4. **The stages on the board**: curvilinear stage 3 (#36: exact
   footprint for geolocation sources, antimeridian), then stage 4 (#37:
   periodic and tripolar grids and CF bounds, which takes the ESMF
   comparison onto ocean grids). Both depend on decision N1 (#43); #37
   also on N3 and N4.
5. **Decisions waiting on the maintainer**: N1 to N5 (#43 to #47) and C1
   to C4 (#48 to #51). N6 is decided (`f32` edges, controlledburn 0.3.0).
6. **rwarp-r**: bind `warp()` (#54) and the Sentinel-2/HLS showcase
   (#55). The R binding still calls the deprecated
   `warp::warp_resample` shim (integer data, near and bilinear only).

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
