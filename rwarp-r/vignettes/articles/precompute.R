# Precompute the website-only articles. They read from the network, so they
# are knitted by hand and the output committed: pkgdown renders the knitted
# .Rmd (plain markdown and figures) without running anything.
#
# From rwarp-r/, with the package installed from this checkout:
#   Rscript vignettes/articles/precompute.R
# then commit the .Rmd files and figures/ beside them.
old <- setwd("vignettes/articles")
on.exit(setwd(old))
for (name in c("sentinel2-warp", "ghrsst-polar")) {
  knitr::knit(paste0(name, ".Rmd.orig"), paste0(name, ".Rmd"), envir = new.env())
}
