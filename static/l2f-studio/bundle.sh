npm install esbuild --save-dev
npx esbuild dependencies/dependencies.js --bundle --minify --format=esm --outfile=blob/lib/dependencies.js
npx esbuild dependencies/three.js --bundle --minify --format=esm --outfile=blob/lib/three.js
npx esbuild dependencies/stats.js --bundle --minify --format=esm --outfile=blob/lib/stats.js
