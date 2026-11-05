#!/usr/bin/env bash
# Vendor the Rust crates into src/rust/vendor.tar.xz, for an offline build
# (CRAN). Run before R CMD build for a submission; CI runs it before the check.
#
# Trimmed to what the build compiles, which takes it from about 17 MB to about
# 5 MB compressed:
#  - tests, benches, examples, docs of every crate (rwarp's own test fixtures
#    are 12 MB);
#  - libsqlite3-sys's bundled SQLite and SQLCipher sources: it is built
#    without its `bundled` feature and links the system SQLite;
#  - proj-sys's PROJ source tarball, used only to build PROJ from source,
#    which tools/config.R refuses (it requires the system PROJ >= 9.6.2).
# Cargo checks vendored files against .cargo-checksum.json, so the per-file
# checksums are emptied (the package checksum stays).
set -euo pipefail

cd "$(dirname "$0")/../src/rust"
rm -rf vendor vendor.tar.xz
cargo vendor --locked --versioned-dirs vendor > /dev/null 2>&1

for d in vendor/*/; do
  rm -rf "${d}tests" "${d}benches" "${d}examples" "${d}docs" "${d}doc" "${d}.github"
done
rm -rf vendor/libsqlite3-sys-*/sqlite3 vendor/libsqlite3-sys-*/sqlcipher
rm -f vendor/proj-sys-*/PROJSRC/*.tar.gz

for f in vendor/*/.cargo-checksum.json; do
  sed -E -i.bak 's/"files":\{[^}]*\}/"files":{}/' "$f" && rm -f "$f.bak"
done

tar -cJf vendor.tar.xz vendor
rm -rf vendor
echo "src/rust/vendor.tar.xz: $(du -h vendor.tar.xz | cut -f1)"
