#!/bin/sh
# Fetch tests/fixtures, which the packaged crate leaves out, from GitHub.
#
#   sh tests/get-fixtures.sh          # the tag for this crate's version
#   sh tests/get-fixtures.sh main     # or any branch, tag or commit
#
# Run from the crate root. Needs curl and tar.
set -eu
version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
ref=${1:-v$version}
mkdir -p tests
curl -fsSL "https://github.com/hypertidy/rwarp/archive/$ref.tar.gz" |
  tar -xz -C tests --strip-components=2 --wildcards "*/tests/fixtures"
echo "tests/fixtures from $ref"
