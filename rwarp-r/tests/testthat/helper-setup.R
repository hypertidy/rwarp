## Skip network tests unless VWARP_NET_TESTS=true
skip_if_no_net <- function() {
  testthat::skip_if(
    Sys.getenv("VWARP_NET_TESTS") != "true",
    "Network tests skipped (set VWARP_NET_TESTS=true to enable)"
  )
}
