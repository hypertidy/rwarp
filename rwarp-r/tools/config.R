# Note: Any variables prefixed with `.` are used for text
# replacement in the Makevars.in and Makevars.win.in

# check the packages MSRV first
source("tools/msrv.R")

# PROJ. The Rust binding (proj-sys 0.27) accepts a system PROJ only from
# 9.6.2. With an older one it quietly builds its own static PROJ, which the
# link below does not complete (undefined sqlite3 symbols) and which would
# read the older system proj.db at run time. Stop here instead.
proj_min <- "9.6.2"
is_windows <- .Platform[["OS.type"]] == "windows"
pkgconf <- Sys.which("pkg-config")
if (!nzchar(pkgconf)) {
  stop(
    "\n------------------------ [PKG-CONFIG NOT FOUND]------------------------\n",
    "'pkg-config' is needed to find PROJ (>= ", proj_min, ").\n",
    "-----------------------------------------------------------------------"
  )
}
proj_found <- suppressWarnings(
  system2(pkgconf, c("--modversion", "proj"), stdout = TRUE, stderr = FALSE)
)
proj_ok <- system2(
  pkgconf, c(paste0("--atleast-version=", proj_min), "proj"),
  stdout = FALSE, stderr = FALSE
) == 0
if (!proj_ok) {
  stop(
    "\n------------------------ [PROJ TOO OLD]--------------------------------\n",
    "rwarp needs PROJ >= ", proj_min, " (found: ",
    if (length(proj_found)) proj_found else "none", ").\n",
    " - Debian testing/unstable: apt-get install libproj-dev\n",
    " - Ubuntu: a recent PROJ from ppa:ubuntugis/ubuntugis-unstable\n",
    " - Fedora: dnf install proj-devel\n",
    " - macOS: brew install proj\n",
    "If PROJ is installed elsewhere, set PKG_CONFIG_PATH to its pkgconfig.\n",
    "-----------------------------------------------------------------------"
  )
}
.proj_libs <- paste(
  system2(pkgconf, c(if (is_windows) "--static", "--libs", "proj"), stdout = TRUE),
  collapse = " "
)
message("Using PROJ ", proj_found)

# check DEBUG and NOT_CRAN environment variables
env_debug <- Sys.getenv("DEBUG")
env_not_cran <- Sys.getenv("NOT_CRAN")

# check if the vendored zip file exists
vendor_exists <- file.exists("src/rust/vendor.tar.xz")

is_not_cran <- env_not_cran != ""
is_debug <- env_debug != ""

if (is_debug) {
  # if we have DEBUG then we set not cran to true
  # CRAN is always release build
  is_not_cran <- TRUE
  message("Creating DEBUG build.")
}

if (!is_not_cran) {
  message("Building for CRAN.")
}

# we set cran flags only if NOT_CRAN is empty and if
# the vendored crates are present.
.cran_flags <- ifelse(
  !is_not_cran && vendor_exists,
  "-j 2 --offline",
  ""
)

# when DEBUG env var is present we use `--debug` build
.profile <- ifelse(is_debug, "", "--release")
.clean_targets <- ifelse(is_debug, "", "$(TARGET_DIR)")

# We specify this target when building for webR
webr_target <- "wasm32-unknown-emscripten"

# here we check if the platform we are building for is webr
is_wasm <- identical(R.version$platform, webr_target)

# print to terminal to inform we are building for webr
if (is_wasm) {
  message("Building for WebR")
}

# we check if we are making a debug build or not
# if so, the LIBDIR environment variable becomes:
# LIBDIR = $(TARGET_DIR)/{wasm32-unknown-emscripten}/debug
# this will be used to fill out the LIBDIR env var for Makevars.in
target_libpath <- if (is_wasm) "wasm32-unknown-emscripten" else NULL
cfg <- if (is_debug) "debug" else "release"

# used to replace @LIBDIR@
.libdir <- paste(c(target_libpath, cfg), collapse = "/")

# use this to replace @TARGET@
# we specify the target _only_ on webR
# there may be use cases later where this can be adapted or expanded
.target <- ifelse(is_wasm, paste0("--target=", webr_target), "")

# add panic exports only for WASM builds
.panic_exports <- ifelse(
  is_wasm,
  "CARGO_PROFILE_DEV_PANIC=\"abort\" CARGO_PROFILE_RELEASE_PANIC=\"abort\" ",
  ""
)

# read in the Makevars.in file checking
is_windows <- .Platform[["OS.type"]] == "windows"

# if windows we replace in the Makevars.win.in
mv_fp <- ifelse(
  is_windows,
  "src/Makevars.win.in",
  "src/Makevars.in"
)

# set the output file
mv_ofp <- ifelse(
  is_windows,
  "src/Makevars.win",
  "src/Makevars"
)

# delete the existing Makevars{.win/.wasm}
if (file.exists(mv_ofp)) {
  message("Cleaning previous `", mv_ofp, "`.")
  invisible(file.remove(mv_ofp))
}

# read as a single string
mv_txt <- readLines(mv_fp)

# replace placeholder values
new_txt <- gsub("@CRAN_FLAGS@", .cran_flags, mv_txt) |>
  gsub("@PROFILE@", .profile, x = _) |>
  gsub("@CLEAN_TARGET@", .clean_targets, x = _) |>
  gsub("@LIBDIR@", .libdir, x = _) |>
  gsub("@TARGET@", .target, x = _) |>
  gsub("@PANIC_EXPORTS@", .panic_exports, x = _) |>
  gsub("@PROJ_LIBS@", .proj_libs, x = _, fixed = TRUE)

message("Writing `", mv_ofp, "`.")
con <- file(mv_ofp, open = "wb")
writeLines(new_txt, con, sep = "\n")
close(con)

message("`tools/config.R` has finished.")
