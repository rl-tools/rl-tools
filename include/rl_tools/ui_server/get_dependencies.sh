VERSION=r156
mkdir static/lib
wget https://github.com/mrdoob/three.js/raw/$VERSION/build/three.module.js -O static/lib/three.module.js
wget https://github.com/mrdoob/three.js/raw/$VERSION/examples/jsm/controls/OrbitControls.js -O static/lib/OrbitControls.js

