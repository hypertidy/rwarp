#' Describe a CRS string in all PROJ output formats
#'
#' A diagnostic and exploration tool: pass any CRS string and get back every
#' representation PROJ can produce.  Useful for verifying that a string is
#' recognised as a CRS (not just a coordinate operation) and for converting
#' between WKT variants, PROJ strings, and PROJJSON.
#'
#' @param definition Character string: EPSG code (`"EPSG:4326"`), WKT, or PROJ
#'   string.  For bare `+proj=` strings, append `+type=crs` to force CRS
#'   interpretation; without it PROJ treats the string as a coordinate
#'   *operation*, which lacks datum information.
#'
#' @return A named list with elements `input`, `id`, `description`,
#'   `definition`, `wkt2_2019`, `wkt2_2019_simplified`, `wkt1_gdal`,
#'   `wkt1_esri`, `proj5`, `proj4`, `projjson`, `error`.
#'   Missing representations are `NA_character_`.
#'
#' @export
#' @examples
#' d <- crs_describe("EPSG:3857")
#' cat(d$wkt2_2019_simplified)
#'
#' # Gotcha: bare PROJ string is treated as an *operation*, not a CRS.
#' # It has no WKT1 form.
#' crs_describe("+proj=laea")$wkt1_gdal
#'
#' # Fix: add +type=crs
#' crs_describe("+proj=laea +datum=WGS84 +type=crs")$wkt1_gdal
crs_describe <- function(definition) {
  stopifnot(is.character(definition), length(definition) == 1L)
  .Call(wrap__crs_describe, definition)
}
