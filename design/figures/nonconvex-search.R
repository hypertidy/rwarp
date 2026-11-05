# Random search behind the note in nonconvex.svg (Rscript, repo root).
src <- readLines("design/figures/make-figures.R")
s <- grep("^# --- 2. non-convex", src); e <- grep("^P_hit <-", src)
h <- grep("^# --- geometry helpers", src); h2 <- grep("^# --- 1. cell map", src)
eval(parse(text = src[h:(h2 - 1)])); eval(parse(text = src[(s + 1):(e - 1)]))
set.seed(1); n <- 0; miss <- 0; ex <- NULL
simple <- function(Q) {  # no two non-adjacent sides cross
  x <- function(a, b, c, d) { o <- function(p, q, r) sign((q[1]-p[1])*(r[2]-p[2])-(q[2]-p[2])*(r[1]-p[1])); o(a,b,c)!=o(a,b,d) && o(c,d,a)!=o(c,d,b) }
  !x(Q[1,],Q[2,],Q[3,],Q[4,]) && !x(Q[2,],Q[3,],Q[4,],Q[1,])
}
for (i in 1:200000) {
  Q <- rbind(c(0,0), c(1,0), c(1,1), c(0,1)) + matrix(runif(8, -0.9, 0.9), 4)
  if (sarea(Q) <= 0 || !simple(Q)) next
  conv <- all(sapply(1:4, function(k) tri_area(Q[k,], Q[k%%4+1,], Q[(k+1)%%4+1,]) > 0))
  if (conv) next
  n <- n + 1; sp <- split_point(Q)
  if (!sp$hits) { miss <- miss + 1; if (is.null(ex)) ex <- Q }
}
cat("non-convex simple:", n, " line misses kernel:", miss, "\n"); if (!is.null(ex)) print(round(ex, 3))
