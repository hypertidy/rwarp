# Cutlines

A cutline limits a warp to the source pixels inside a polygon, as
`gdalwarp -cutline` does. rwarp takes the polygon in **source pixel/line
coordinates** of the full source image, which is where gdalwarp puts it
before warping:

```rust,ignore
use rwarp::cutline::{Cutline, CutlineRule};

// Polygons: an exterior ring and its holes, each ring a list of [x, y].
let cut = Cutline::new(vec![vec![exterior, hole]]);
let opts = WarpOptions::new(ResampleAlg::Bilinear).with_cutline(cut);
let out = warp(&transformer, &source_buffer, dst_ncol, dst_nrow, &opts);
```

Several polygons are a multipolygon (their union).

## From world coordinates

- **`Cutline::from_geotransform(polygons, src_gt)`**: the polygons are in
  the source raster's CRS. The map to pixels is affine, so nothing is
  densified, as in gdalwarp.
- **`Cutline::from_transformer(polygons, &transformer)`**: the polygons are
  in whatever a `Transformer` maps to source pixels. For a cutline in
  another CRS, build a `GenImgProjTransformer` from that CRS to the source
  with the identity geotransform `[0, 1, 0, 0, 0, 1]` on the cutline side,
  so its "pixels" are CRS coordinates:

  ```rust,ignore
  let t = GenImgProjTransformer::new(src_crs, src_gt, "EPSG:4326", [0.0, 1.0, 0.0, 0.0, 0.0, 1.0])?;
  let cut = Cutline::from_transformer(lonlat_polygons, &t)?;
  ```

  Edges are densified as gdalwarp densifies them, so that each transformed
  edge is about one source pixel long.

## Rules

| Rule | A source pixel counts when | GDAL |
|---|---|---|
| `CutlineRule::Centre` (default) | its centre is inside | default |
| `CutlineRule::AllTouched` | the polygon covers any of it | `-wo CUTLINE_ALL_TOUCHED=YES` |
| `CutlineRule::Exact` | always, weighted by the fraction of it covered | (none) |

`with_blend_dist(d)` is gdalwarp's `-cblend d`: within `d` source pixels of
the boundary the density ramps from 1 inside to 0 outside, 0.5 on the line.

**Centre ties.** A pixel centre exactly on the boundary follows GDAL: on a
left edge it is out, on a right edge in, on a top edge in, on a bottom
edge out, and a hole whose top edge runs along a line of centres does not
cut that line. On a diagonal edge through centres the answer is a rounding
error in GDAL too, and the two can disagree.

**All touched** counts a pixel the polygon covers by more than 1e-9 of its
area. GDAL walks each boundary line instead, which differs in three
places: GDAL also burns a pixel the boundary only touches at its corner
(below-right of the corner), skips an edge within 1e-4 of a pixel wall,
and burns a single column for an edge less than 0.01 pixels from
vertical.

## What a cutline does to each method

The cutline becomes a density per source pixel, computed over the source
buffer only.

- **Point methods** (nearest, bilinear, cubic, cubic spline, lanczos): a
  target pixel whose sample point lands outside the cutline gets no
  value; pixels outside are dropped from every kernel, like nodata, so
  cubic falls back to bilinear near the boundary.
- **Area methods** (average, mode, min, max, quantiles, RMS): pixels
  outside the cutline are left out, as nodata pixels are.
- **Blend distance**: point methods interpolate the density along with the
  value. With a destination nodata the value is kept wherever the density
  is at least 0.0001; without one it is multiplied by the density, fading
  to the zero fill, which is what gdalwarp does when warping into a new
  file. Area methods only use the density as validity, so for them the
  blend distance widens the cutline.
- **Exact**: area methods weight each pixel by footprint times coverage
  (so a pixel half inside counts half in `Average`); point kernels weight
  each tap by its coverage, and a target pixel gets a value when its
  sample point's pixel is at least half covered.

## Cost

The cutline is rasterized once per `warp` call, over the source buffer:
the centre rule with `controlledburn`'s centre-rule burn, all touched and
exact with its sparse coverage burn, and the blend ramp only within the
blend distance of each edge. For a 14-vertex star over a 4096 x 4096
buffer that is about 15 ms (centre), 20 ms (all touched, exact) and 20 ms
(blend distance 10) in a release build.
