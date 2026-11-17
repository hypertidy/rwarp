# Render the logo's raster assets for the R package and its pkgdown site.
#
#   Rscript design/logo/render_assets.R        (from the repository root)
#
# From design/logo/rwarp-hex.svg and favicon.svg (make_logo.py) writes
#   rwarp-r/man/figures/logo.png          the hex, 278 px high (usethis's size)
#   rwarp-r/pkgdown/favicon/              the files pkgdown's head template links:
#     favicon.svg, favicon-96x96.png, favicon.ico (16, 32, 48),
#     apple-touch-icon.png (180), web-app-manifest-192x192.png, -512x512.png,
#     site.webmanifest
# Rendered here rather than with pkgdown::build_favicons(), which sends the
# logo to an online service. Needs rsvg and magick.

library(magick)

logo <- "design/logo/rwarp-hex.svg"
mark <- "design/logo/favicon.svg"
navy <- "#13283a"
fav <- "rwarp-r/pkgdown/favicon"
dir.create("rwarp-r/man/figures", showWarnings = FALSE, recursive = TRUE)
dir.create(fav, showWarnings = FALSE, recursive = TRUE)

rsvg::rsvg_png(logo, "rwarp-r/man/figures/logo.png", height = 278)

hex <- function(h) image_read(rsvg::rsvg_png(mark, height = h))
## the hex centred on a square, transparent or filled
square <- function(size, bg = "none", fill = 0.92) {
  h <- round(size * fill)
  image_composite(image_blank(size, size, bg), hex(h), operator = "over", gravity = "center")
}

file.copy(mark, file.path(fav, "favicon.svg"), overwrite = TRUE)
image_write(square(96), file.path(fav, "favicon-96x96.png"), format = "png")
image_write(c(square(16, fill = 1), square(32, fill = 1), square(48, fill = 1)),
            file.path(fav, "favicon.ico"), format = "ico")
image_write(square(180, navy, 0.8), file.path(fav, "apple-touch-icon.png"), format = "png")
image_write(square(192, navy, 0.7), file.path(fav, "web-app-manifest-192x192.png"), format = "png")
image_write(square(512, navy, 0.7), file.path(fav, "web-app-manifest-512x512.png"), format = "png")
writeLines(jsonlite::toJSON(list(
  name = "rwarp", short_name = "rwarp",
  icons = list(
    list(src = "web-app-manifest-192x192.png", sizes = "192x192", type = "image/png", purpose = "maskable"),
    list(src = "web-app-manifest-512x512.png", sizes = "512x512", type = "image/png", purpose = "maskable")
  ),
  theme_color = navy, background_color = navy, display = "standalone"
), auto_unbox = TRUE, pretty = TRUE), file.path(fav, "site.webmanifest"))
