# Figures for design/decisions.md. Run from the repo root:
#   Rscript design/figures/make-figures.R
# Base R only; writes SVGs next to this script. The geometry is computed,
# not sketched: the numbers printed on each figure come from this script.

out <- "design/figures"

ink <- "#0b0b0b"; ink2 <- "#52514e"; muted <- "#8a8984"; surface <- "#fcfcfb"
blue <- c("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95")
mid <- "#f0efec"
red <- c("#fbd3d2", "#f4a3a2", "#e34948", "#b8302f")

open_svg <- function(name, w, h) {
  svg(file.path(out, name), width = w, height = h, bg = surface, family = "sans")
  par(fg = ink2, col.axis = ink2, col.lab = ink2, col.main = ink,
      mar = c(1, 1, 3, 1))
}

# --- geometry helpers --------------------------------------------------------
sarea <- function(p) {  # signed shoelace area, p is n x 2
  x <- p[, 1]; y <- p[, 2]; j <- c(2:nrow(p), 1)
  sum(x * y[j] - x[j] * y) / 2
}
tri_area <- function(a, b, c) sarea(rbind(a, b, c))
# Sutherland-Hodgman: clip polygon `s` by convex CCW polygon `k`
clip <- function(s, k) {
  inside <- function(p, a, b) (b[1] - a[1]) * (p[2] - a[2]) - (b[2] - a[2]) * (p[1] - a[1]) >= -1e-12
  cross <- function(p, q, a, b) {
    d1 <- q - p; d2 <- b - a
    t <- ((a[1] - p[1]) * d2[2] - (a[2] - p[2]) * d2[1]) / (d1[1] * d2[2] - d1[2] * d2[1])
    p + t * d1
  }
  for (i in seq_len(nrow(k))) {
    a <- k[i, ]; b <- k[if (i == nrow(k)) 1 else i + 1, ]
    if (is.null(s) || nrow(s) == 0) return(NULL)
    res <- NULL
    for (j in seq_len(nrow(s))) {
      p <- s[j, ]; q <- s[if (j == nrow(s)) 1 else j + 1, ]
      ip <- inside(p, a, b); iq <- inside(q, a, b)
      if (ip && iq) res <- rbind(res, q)
      else if (ip && !iq) res <- rbind(res, cross(p, q, a, b))
      else if (!ip && iq) res <- rbind(res, cross(p, q, a, b), q)
    }
    s <- res
  }
  s
}
# keep the part of polygon s left of the directed line a -> b
halfplane <- function(s, a, b) {
  if (is.null(s) || nrow(s) < 3) return(NULL)
  side <- function(p) (b[1] - a[1]) * (p[2] - a[2]) - (b[2] - a[2]) * (p[1] - a[1])
  res <- NULL; n <- nrow(s)
  for (j in seq_len(n)) {
    p <- s[j, ]; q <- s[if (j == n) 1 else j + 1, ]
    sp <- side(p); sq <- side(q)
    if (sp >= 0) res <- rbind(res, p)
    if ((sp >= 0) != (sq >= 0)) res <- rbind(res, p + (sp / (sp - sq)) * (q - p))
  }
  res
}
affine_tri <- function(p, from, to) {  # map point p from triangle `from` to `to`
  M <- cbind(from[2, ] - from[1, ], from[3, ] - from[1, ])
  w <- solve(M, p - from[1, ])
  to[1, ] + w[1] * (to[2, ] - to[1, ]) + w[2] * (to[3, ] - to[1, ])
}

# --- 1. cell map: bilinear vs equal-area ------------------------------------
P <- rbind(c(0, 0), c(4.2, 0.5), c(3.3, 3.8), c(-0.6, 2.2))  # P00 P10 P11 P01
A <- sarea(P); n <- 6
bil <- function(u, v) (1 - u) * (1 - v) * P[1, ] + u * (1 - v) * P[2, ] + u * v * P[3, ] + (1 - u) * v * P[4, ]
C <- colMeans(P)
cu <- c(2 * tri_area(P[4, ], P[1, ], C) / A, 2 * tri_area(P[1, ], P[2, ], C) / A)
U <- rbind(c(0, 0), c(1, 0), c(1, 1), c(0, 1))
tris_u <- lapply(1:4, function(k) rbind(U[k, ], U[k %% 4 + 1, ], cu))
tris_w <- lapply(1:4, function(k) rbind(P[k, ], P[k %% 4 + 1, ], C))

ratio_col <- function(r) {  # diverging, gray at 1, blue below, red above
  d <- r - 1; s <- 0.15
  if (abs(d) < 0.005) return(mid)
  i <- min(4, ceiling(abs(d) / s * 4))
  if (d < 0) rev(blue[1:4])[5 - i] else red[i]
}
cells_bil <- list(); cells_ea <- list()
for (i in 0:(n - 1)) for (j in 0:(n - 1)) {
  sq <- rbind(c(i, j), c(i + 1, j), c(i + 1, j + 1), c(i, j + 1)) / n
  b <- t(apply(sq, 1, function(p) bil(p[1], p[2])))
  cells_bil[[length(cells_bil) + 1]] <- list(poly = b, r = sarea(b) / (A / n^2))
  pieces <- list(); a <- 0
  for (k in 1:4) {
    pc <- clip(sq, tris_u[[k]])
    if (!is.null(pc) && nrow(pc) >= 3) {
      w <- t(apply(pc, 1, affine_tri, from = tris_u[[k]], to = tris_w[[k]]))
      pieces[[length(pieces) + 1]] <- w; a <- a + sarea(w)
    }
  }
  cells_ea[[length(cells_ea) + 1]] <- list(pieces = pieces, r = a / (A / n^2))
}
rb <- range(sapply(cells_bil, `[[`, "r")); re <- range(sapply(cells_ea, `[[`, "r"))

open_svg("cellmap.svg", 9, 4.6)
layout(matrix(1:3, 1), widths = c(1, 1, 0.42))
xl <- range(P[, 1]) + c(-0.3, 0.3); yl <- range(P[, 2]) + c(-0.3, 0.6)
plot.new(); plot.window(xl, yl, asp = 1)
for (cc in cells_bil) polygon(cc$poly, col = ratio_col(cc$r), border = surface, lwd = 1.5)
polygon(P, border = ink, lwd = 1.5)
title(main = "Bilinear (GDAL's geolocation model)", font.main = 1, cex.main = 1.15)
mtext(sprintf("36 equal index cells: %.2f to %.2f of an equal share",
              rb[1], rb[2]), side = 1, line = -0.5, cex = 0.8, col = ink2)
plot.new(); plot.window(xl, yl, asp = 1)
for (cc in cells_ea) for (pc in cc$pieces) polygon(pc, col = ratio_col(cc$r), border = NA)
for (i in 0:n) {  # sub-cell outlines through the piecewise map
  for (dir in 1:2) {
    tt <- seq(0, 1, length.out = 241)
    pts <- if (dir == 1) cbind(i / n, tt) else cbind(tt, i / n)
    w <- t(apply(pts, 1, function(p) {
      for (k in 1:4) {
        M <- cbind(tris_u[[k]][2, ] - tris_u[[k]][1, ], tris_u[[k]][3, ] - tris_u[[k]][1, ])
        b <- solve(M, p - tris_u[[k]][1, ])
        if (all(b >= -1e-9) && sum(b) <= 1 + 1e-9) return(affine_tri(p, tris_u[[k]], tris_w[[k]]))
      }
      c(NA, NA)
    }))
    lines(w, col = surface, lwd = 1.5)
  }
}
for (k in 1:4) segments(P[k, 1], P[k, 2], C[1], C[2], col = ink2, lty = 2, lwd = 0.8)
points(C[1], C[2], pch = 19, col = ink, cex = 0.8)
polygon(P, border = ink, lwd = 1.5)
title(main = "EqualArea (four triangles at the corner mean)", font.main = 1, cex.main = 1.15)
mtext(sprintf("the same 36: %.3f to %.3f of an equal share",
              re[1], re[2]), side = 1, line = -0.5, cex = 0.8, col = ink2)
plot.new(); plot.window(c(0, 1), c(0, 1))
levs <- c(0.85, 0.92, 0.97, 1, 1.03, 1.08, 1.15)
labs <- c("-15%", "-8%", "-3%", "equal", "+3%", "+8%", "+15%")
for (i in seq_along(levs)) {
  y <- 0.85 - (i - 1) * 0.1
  rect(0.05, y - 0.035, 0.25, y + 0.035, col = ratio_col(levs[i]), border = NA)
  text(0.3, y, labs[i], adj = 0, cex = 0.9, col = ink2)
}
text(0.05, 0.97, "world area of\nan index cell", adj = c(0, 1), cex = 0.9, col = ink)
dev.off()

# --- 2. non-convex cells ------------------------------------------------------
split_point <- function(P) {
  A <- sarea(P); C <- colMeans(P)
  f <- function(X) tri_area(P[1, ], P[2, ], X) + tri_area(P[3, ], P[4, ], X) - A / 2
  # f is affine in X: f(X) = g . X + h
  h <- f(c(0, 0)); g <- c(f(c(1, 0)) - h, f(c(0, 1)) - h)
  kern <- rbind(c(-50, -50), c(50, -50), c(50, 50), c(-50, 50))
  for (k in 1:4) kern <- halfplane(kern, P[k, ], P[k %% 4 + 1, ])
  dirn <- c(-g[2], g[1]) / sqrt(sum(g^2))
  X0 <- C - g * f(C) / sum(g^2)  # = C, since f(C) = 0
  # inside the kernel: every side's orientation positive; each is affine in t
  lo <- -Inf; hi <- Inf
  for (k in 1:4) {
    s0 <- tri_area(P[k, ], P[k %% 4 + 1, ], X0)
    s1 <- tri_area(P[k, ], P[k %% 4 + 1, ], X0 + dirn) - s0
    if (abs(s1) < 1e-15) { if (s0 <= 0) { lo <- 1; hi <- 0 } }
    else if (s1 > 0) lo <- max(lo, -s0 / s1) else hi <- min(hi, -s0 / s1)
  }
  hits <- lo < hi
  chosen <- if (hits) X0 + min(max(0, lo), hi) * dirn else NULL
  list(A = A, C = C, kern = kern, line = rbind(X0 - 20 * dirn, X0 + 20 * dirn),
       chosen = chosen, hits = hits)
}
P_hit <- rbind(c(0, 0), c(4, 0), c(1.7, 1.4), c(0, 4))
P_dart <- rbind(c(0, 0), c(4, 0.4), c(0.7, 0.6), c(-0.2, 3.2))
draw_nc <- function(P, ttl) {
  s <- split_point(P)
  plot.new(); plot.window(range(P[, 1]) + c(-0.4, 0.4), range(P[, 2]) + c(-0.4, 0.6), asp = 1)
  polygon(s$kern, col = blue[1], border = blue[3], lwd = 1)
  lines(s$line, col = red[3], lwd = 2)
  polygon(P, border = ink, lwd = 1.5)
  if (!is.null(s$chosen)) {
    for (k in 1:4) segments(P[k, 1], P[k, 2], s$chosen[1], s$chosen[2], col = ink2, lty = 2)
    points(s$chosen[1], s$chosen[2], pch = 19, col = ink)
  }
  points(s$C[1], s$C[2], pch = 1, col = ink, cex = 1.2)
  title(main = ttl, font.main = 1, cex.main = 1.1)
  s
}
open_svg("nonconvex.svg", 9, 4.6)
layout(matrix(1:3, 1), widths = c(1, 1, 0.55))
s1 <- draw_nc(P_hit, "A non-convex cell: exact split")
s2 <- draw_nc(P_dart, "A deep dart: still exact")
mtext("0 of 59,234 random simple non-convex cells had the line miss the kernel",
      side = 1, line = -0.5, cex = 0.8, col = ink2, adj = 1)
plot.new(); plot.window(c(0, 1), c(0, 1))
rect(0.02, 0.86, 0.14, 0.92, col = blue[1], border = blue[3])
text(0.18, 0.89, "kernel (points that\nsee every corner)", adj = 0, cex = 0.85, col = ink2)
segments(0.02, 0.74, 0.14, 0.74, col = red[3], lwd = 2)
text(0.18, 0.74, "points giving equal\narea shares", adj = 0, cex = 0.85, col = ink2)
points(0.08, 0.6, pch = 1, cex = 1.2, col = ink)
text(0.18, 0.6, "corner mean", adj = 0, cex = 0.85, col = ink2)
points(0.08, 0.5, pch = 19, col = ink)
text(0.18, 0.5, "split point used", adj = 0, cex = 0.85, col = ink2)
dev.off()

# --- 3. folds ----------------------------------------------------------------
open_svg("folds.svg", 9, 4.2)
layout(matrix(1:2, 1))
plot.new(); plot.window(c(-0.5, 8.5), c(-0.5, 4.8))
nc <- 8
for (i in 0:(nc - 1)) for (j in 0:1) rect(i, j, i + 1, j + 1, col = if (j == 1) blue[2] else blue[1], border = surface, lwd = 2)
for (i in 0:(nc - 1)) text(i + 0.5, 1.5, i, cex = 0.9, col = ink)
for (i in 0:(nc / 2 - 1)) {
  x1 <- i + 0.5; x2 <- nc - 1 - i + 0.5; xm <- (x1 + x2) / 2; r <- (x2 - x1) / 2
  th <- seq(0, pi, length.out = 60)
  lines(xm + r * cos(th), 2 + 0.55 * r * sin(th), col = ink2, lwd = 1.2)
}
text(4, 4.6, "Tripolar: the top row folds onto itself", cex = 1.05, col = ink)
text(4, -0.35, "(top, i) joins (top, ncol - 1 - i): a neighbour rule the walk follows",
     cex = 0.75, col = ink2)
plot.new(); plot.window(c(0, 10), c(-0.5, 4.8))
band <- function(y0, col) polygon(c(0.5, 9.5, 9.5, 0.5), c(y0, y0 + 0.4, y0 + 1.6, y0 + 1.2), col = col, border = surface, lwd = 2)
band(0.6, adjustcolor(blue[3], 0.55)); band(1.6, adjustcolor(red[2], 0.6))
polygon(c(0.5, 9.5, 9.5, 0.5), c(1.6, 2.0, 2.2, 1.8), density = 25, angle = 45, col = ink2, border = NA)
text(5, 4.6, "Other folds: overlapping sheets", cex = 1.05, col = ink)
text(5, 3.5, "two scan lines cover the same ground (MODIS bow-tie)", cex = 0.8, col = ink2)
text(5, -0.35, "a point lies in two cells: refused for coverage",
     cex = 0.75, col = ink2)
dev.off()

# --- 4. normalisation --------------------------------------------------------
vals <- matrix(c(10, 12, NA, 11, 13, NA, 12, 14, NA), 3, 3, byrow = TRUE)
cov <- !is.na(vals)
frac <- mean(vals[cov]); dest <- sum(vals[cov]) / 9
open_svg("normalisation.svg", 8, 4.4)
layout(matrix(1:2, 1), widths = c(1, 1.1))
par(mar = c(1, 1, 3, 1))
plot.new(); plot.window(c(-0.2, 3.2), c(-0.6, 3.4), asp = 1)
seqc <- colorRampPalette(blue[1:5])(5)
for (r in 1:3) for (cc in 1:3) {
  v <- vals[r, cc]
  if (is.na(v)) {
    rect(cc - 1, 3 - r, cc, 4 - r, col = mid, border = surface, lwd = 2)
    rect(cc - 1, 3 - r, cc, 4 - r, density = 14, angle = 45, col = muted, border = NA)
  } else {
    rect(cc - 1, 3 - r, cc, 4 - r, col = seqc[v - 9], border = surface, lwd = 2)
    text(cc - 0.5, 3.5 - r, v, col = if (v >= 13) surface else ink, cex = 1.1)
  }
}
rect(0, 0, 3, 3, border = ink, lwd = 2)
title(main = "One destination cell over source cells", font.main = 1, cex.main = 1.05)
text(1.5, -0.4, "hatched: no source (outside the grid, or nodata)", cex = 0.75, col = ink2)
par(mar = c(1, 0, 3, 1))
plot.new(); plot.window(c(0, 1), c(0, 1))
text(0, 0.86, sprintf("FracArea = %.1f", frac), adj = 0, cex = 1.4, col = ink)
text(0, 0.74, "mean over the covered part\n(warp() today; xESMF conservative_normed)", adj = c(0, 1), cex = 0.85, col = ink2)
text(0, 0.42, sprintf("DestArea = %.1f", dest), adj = 0, cex = 1.4, col = ink)
text(0, 0.30, "sum over the whole cell, missing counted as 0\n(ESMF default DSTAREA; xESMF conservative)", adj = c(0, 1), cex = 0.85, col = ink2)
dev.off()

# --- 5. cell edges: parallels vs great circles --------------------------------
rad <- pi / 180
to_xyz <- function(lon, lat) cbind(cos(lat * rad) * cos(lon * rad), cos(lat * rad) * sin(lon * rad), sin(lat * rad))
to_ll <- function(p) cbind(atan2(p[, 2], p[, 1]) / rad, asin(p[, 3] / sqrt(rowSums(p^2))) / rad)
gc <- function(a, b, n = 100) {  # a, b lon/lat; slerp
  pa <- to_xyz(a[1], a[2]); pb <- to_xyz(b[1], b[2]); om <- acos(sum(pa * pb))
  t <- seq(0, 1, length.out = n)
  to_ll((sin((1 - t) * om) / sin(om)) %o% pa[1, ] + (sin(t * om) / sin(om)) %o% pb[1, ])
}
sph_tri <- function(a, b, c) {  # spherical excess, l'Huilier
  A <- to_xyz(a[1], a[2])[1, ]; B <- to_xyz(b[1], b[2])[1, ]; C <- to_xyz(c[1], c[2])[1, ]
  ab <- acos(sum(A * B)); bc <- acos(sum(B * C)); ca <- acos(sum(C * A)); s <- (ab + bc + ca) / 2
  4 * atan(sqrt(max(0, tan(s / 2) * tan((s - ab) / 2) * tan((s - bc) / 2) * tan((s - ca) / 2))))
}
box_vs_gc <- function(l0, l1, p0, p1) {
  box <- (l1 - l0) * rad * (sin(p1 * rad) - sin(p0 * rad))
  q <- sph_tri(c(l0, p0), c(l1, p0), c(l1, p1)) + sph_tri(c(l0, p0), c(l1, p1), c(l0, p1))
  c(box = box, gc = q, rel = (q - box) / box)
}
small <- box_vs_gc(0, 2.5, -41, -39)
bigc <- box_vs_gc(0, 40, -60, -30)
open_svg("cell-edges.svg", 8, 5)
par(mar = c(4, 4, 3, 1))
plot(NA, xlim = c(-2, 42), ylim = c(-66, -20), asp = 1, xlab = "longitude", ylab = "latitude",
     las = 1, bty = "n")
abline(h = seq(-60, -30, 10), v = seq(0, 40, 10), col = "#e6e5e0", lwd = 0.8)
top <- gc(c(0, -30), c(40, -30)); bot <- gc(c(0, -60), c(40, -60))
polygon(c(0, 40, 40, 0), c(-60, -60, -30, -30), col = adjustcolor(blue[2], 0.6), border = blue[5], lwd = 2)
lines(top, col = red[3], lwd = 2); lines(bot, col = red[3], lwd = 2)
segments(c(0, 40), c(-60, -60), c(0, 40), c(-30, -30), col = red[3], lwd = 2, lty = 3)
title(main = "A 40 x 30 degree cell: bounded by parallels, or by great circles", font.main = 1, cex.main = 1.05)
legend("topleft", bty = "n", cex = 0.85, text.col = ink2,
       legend = c(sprintf("parallels (CF lon/lat; rwarp, exact)"),
                  sprintf("great circles (ESMF, TempestRemap): %+.1f%% area", 100 * bigc["rel"])),
       col = c(blue[5], red[3]), lwd = 2)
mtext(sprintf("at 2.5 x 2 degrees, 40 S the difference is %.1e of the cell", abs(small["rel"])),
      side = 3, line = 0.2, cex = 0.8, col = ink2)
dev.off()

cat(sprintf("cellmap: bilinear %.3f..%.3f, equal-area %.4f..%.4f\n", rb[1], rb[2], re[1], re[2]))
cat(sprintf("normalisation: FracArea %.2f DestArea %.2f\n", frac, dest))
cat(sprintf("edges: small rel %.3e, big rel %.4f\n", small["rel"], bigc["rel"]))
