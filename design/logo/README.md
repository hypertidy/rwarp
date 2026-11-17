# The rwarp logo

`rwarp-hex.svg` is the hex sticker; `favicon.svg` is the same mark without
the field or the name, for small sizes. Both are drawn from rwarp's own
output:

- **The map** (bottom left) is the target, EPSG:3031 at 10 km over the box of
  vapour's `sst.tif` (the Get started article), cut into a regular 6 x 6 grid
  of tiles. It is filled with GHRSST MUR v2 sea surface temperature for
  2025-11-26, warped by rwarp to 5 km and drawn as 1 C bands. Only the
  bottom-left half shows, cut square to the hex's 30 degree diagonal.
  Tasmania is the notch near the bottom.
- **The plan** (top right) has one row per tile that shows in the map, in
  plan order and in the tile's colour. Each row is two-toned by the number of
  source pixels its tile reads, from `warp_plan()` for that tile against the
  full-resolution MUR grid.
- **The name** runs down the cut between them.

## Remaking it

From the repository root:

```sh
Rscript design/logo/make_logo_data.R /tmp/logo     # rwarp, gdalraster, isoband; reads MUR from the network
python3 design/logo/make_logo.py /tmp/logo design/logo
Rscript design/logo/render_assets.R                # rsvg, magick
```

`make_logo_data.R` warps the field and plans the tiles (4 x 4 to 6 x 6, for
trying other cuts) and writes them as JSON. `make_logo.py` draws the SVGs;
regenerating from scratch gives the same bytes. `render_assets.R` writes
`rwarp-r/man/figures/logo.png` (278 px high) and the favicons in
`rwarp-r/pkgdown/favicon/`, rendered locally rather than with
`pkgdown::build_favicons()`, which sends the logo to an online service.

The design came out of a brainstorm on 2026-11-16 (a plan view beside a map
view, on navy); `make_logo.py`'s arguments to `sticker()` are the levers:
the tiling `n`, the tint, the zoom (`mw`, `at`) and the cut (`s0`).
