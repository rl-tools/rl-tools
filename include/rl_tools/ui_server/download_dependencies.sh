VERSION=r156
mkdir static/multirotor/lib
wget https://github.com/mrdoob/three.js/raw/$VERSION/build/three.module.js -O static/multirotor/lib/three.module.js
wget https://github.com/mrdoob/three.js/raw/$VERSION/examples/jsm/controls/OrbitControls.js -O static/multirotor/lib/OrbitControls.js

