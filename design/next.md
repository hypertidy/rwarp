# State and what's next (2026-11-19)

Where rwarp stands at the end of the 2026-11-16 to 2026-11-19 session,
and what to pick up. The long-term framing is in #52 and on the roadmap
board (<https://github.com/orgs/hypertidy/projects/1>). The open
decisions, with options and figures, are in [decisions.md](decisions.md).
This file is a snapshot, so replace it rather than append to it.

## Released

| crate | version | date | what |
|---|---|---|---|
| rwarp | 0.5.0 | 2026-11-18 | `MeshSet` and NEMO's halos, world-coordinate measure for curvilinear sources, threads (#99; see the changelog) |
| rwarp | 0.4.1 | 2026-11-16 | `QuadMesh::burn_par`, `Rectilinear::burn_par` (#59, #63) |
| rwarp | 0.4.0 | 2026-11-12 | controlledburn 0.3.1 (breaking for rwarp: its types are in rwarp's API); N6 recorded |
| rwarp | 0.3.3 | 2026-11-12 | antimeridian split reads two edge windows, not the whole width; `merge_tile` |
| controlledburn | 0.3.2 | 2026-11-12 | `burn_wkb_par`; burns the same as 0.3.1 |
| controlledburn | 0.3.1 | 2026-11-12 | `relate_cells` (0.3.0: `burn_par`, `burn_window`, `merge_parts`, `f32` edges kept) |
| bigcurve (crate) | 0.3.0 | 2026-11-04 | |

`v0.4.0` (on `6a4c337`) and `v0.4.1` (on `e4c3e49`) were tagged on
2026-11-16, each on the commit its crate was published from; `v0.5.0`
(on `67eb7c4`, #99's merge) on 2026-11-18. Main is now 0.5.1
(unreleased, #100); a breaking change before the next release makes it
0.6.0.

## Since 0.4.1

Everything from #64 to #98, by the changelog's 0.5.0 section:

- **Curvilinear stage 3 (#36)**: the exact footprint over geolocation
  sources measures in their world coordinates, across the antimeridian
  and round the poles (#70, #71, #72).
- **Stage 4 (#37, closed 2026-11-18)**, all of 4a to 4i in
  `curvilinear-frontend.md`: the walk across seams and folds, CF bounds
  and halos (N3), ESMF on a tripolar grid, a mesh as a warp source and target, a declared period,
  `MeshSet` for a whole ocean grid in pieces with weights, burning
  (serial and threaded, projection on the threads) and `warp()` from it
  (`SetTransformer`). Then the file's layout and NEMO's halo exchange:
  `from_file_layout`, `to_file_layout`, `fill_halo` (#95), vector
  components across the fold (#96), U, V and F points (#97), and sets of
  U, V and F cells, `Halo::nemo` (#98).
- **Threads**, bit for bit with one thread: `warp_par` and `warp_to_par`
  (#91), `warp_chunks_par` (#92), `exact_weights_par` (#93),
  `MeshSet::weights_to_par` and `weights_onto_par` (#94).
- **Decisions** N1 (#43), N3 and N4 (#77) decided and built.
- **rwarp-r**: `warp_buffer()` (#54, #65), the Sentinel-2 vignette
  (#55), `warp_plan()` with gdalwarp's working-memory rules (#67), and
  the Get started vignette (#74). It moved to rwarp 0.5.0 on
  2026-11-18, its tests unchanged.

## Since 0.5.0

In main for 0.5.1 (unreleased), by the changelog's 0.5.1 section:

- **Curvilinear stage 5 (#38, closed 2026-11-18)**, 5a and 5b in
  `curvilinear-frontend.md`: `Footprint::Adaptive` and
  `WeightOptions::refine_tolerance`, the exact footprint's edges refined
  by bigcurve (#103), and `burn_from` on a `QuadMesh` or `Rectilinear`
  for polygons in another CRS (#104).
- **Conservative stage 5 (#39, closed 2026-11-18)**, section 7 of
  `conservative.md`: `mesh::PolyMesh`, a mesh of polygons (ICON, MPAS,
  UGRID, ESMF meshes), and conservative weights between it and a grid
  both ways, `weights_onto` and `weights_to` (#106); great-circle edges,
  `with_great_circles`, edges cut at the equator and nodes on the poles
  taken along their meridians (#107); grids with world cells on the other
  side, geolocation arrays, a `QuadMesh` or a `MeshSet`, through the
  exact footprint's world-cell measure factored out as `WorldCells`
  (#108); and threads, `weights_onto_par` and `weights_to_par`, bit for
  bit (#109). Checked against ESMF (Cartesian to 6e-14; spherical, the
  faces' areas to 8e-8) and against closed forms, clipping and burns.
- **#76** fixed (#101); Open says what rwarp-r needs for it.

## Open

- **Left out of stage 4** (#37's closing comment), each an issue if
  wanted: TempestRemap on the tripolar grid (ESMF was run there;
  TempestRemap is an oracle only for regular grids), and cheaper libproj
  copies for the threads (`proj_clone`, which the `proj` crate doesn't
  expose).
- **Left out of stage 5** (#38's closing comment), each an issue if
  wanted: great-circle polygon edges for `burn_from` (close to #41), a
  tolerance in cells rather than world units, and bilinear cells
  through bigcurve.
- **Left out of conservative stage 5** (#39): mesh to mesh (polygon
  overlay, out of scope as the issue says); and a grid's own cells as
  great circles, ESMF's lon/lat cells (rwarp's are bounded by parallels,
  so the spherical maps against ESMF differ by those cells, `S` to
  3.8e-3 at 5 degrees; see #41). Reading UGRID, MPAS and ICON files is
  wxylib's.
- **#76** is fixed in the crate (#101; 0.5.1, unreleased): with
  `split_antimeridian`, a full-width window with no jump takes GDAL's
  fill ratio split again. rwarp-r gets it with its next crate update;
  its docs already say `split_antimeridian = FALSE` is gdalwarp's plan,
  which stays true.

## Next, in order

1. **The next stage on the board**: conservative stage 6 (#40, second
   order). 6a is built for sources measured in their pixels (section 8 of
   conservative.md); it needs controlledburn's release with
   `CoverageWindow::set_moments` (controlledburn-rs #51). Left: world-cell
   and mesh sources.
2. **Decisions**: none waiting. N2, N5 and C1 to C4 were decided on
   2026-11-19 (#111). N5 (#47) is built, and adaptive predicates weren't
   needed (see N5 in decisions.md).
3. **The rest of controlledburn 0.3**: `burn_window` could replace
   `Cutline::burn_window` (check the records are identical first) and
   split one large geometry into tiles for `burn_par`; `relate_cells`
   finds cells a polygon only touches.

## Parked

- **CRAN for rwarp-r** (#42): not being targeted for now (2026-11-12).
  The build layout is ready (#32): vendored crates, an offline build,
  and a PROJ >= 9.6.2 configure check. The Windows result from
  win-builder was never reviewed.
- **Upstreaming the GDAL findings** (#34): parked until the PSC business
  settles.
- **Great-circle cell edges** (#41): a `PolyMesh`'s faces can be great
  circles (#107). A grid's own cells as great circles (ESMF's) would need
  curved source cells, a separate engine from controlledburn's
  (`conservative.md` 5.2).
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
- Moving rwarp-r to a new rwarp: change `rwarp-r/src/rust/Cargo.toml`,
  run `cargo update -p rwarp` there, and commit the lock with it (the
  lock and the manifest must move together, or `vendor.sh` fails). To
  install it locally, run `tools/vendor.sh` first, and remove a stale
  `src/vendor` left by a failed build: the build uses that directory
  before the tarball.
- Main's version leads the release. After publishing, the next
  unreleased version and its changelog section go in with the next PR.
