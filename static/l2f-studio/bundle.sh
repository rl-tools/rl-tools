#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

npm ci
npx --no-install esbuild dependencies/dependencies.js --bundle --minify --format=esm --outfile=external/blob/lib/dependencies.js
npx --no-install esbuild dependencies/three.js --bundle --minify --format=esm --outfile=external/blob/lib/three.js
npx --no-install esbuild dependencies/stats.js --bundle --minify --format=esm --outfile=external/blob/lib/stats.js
