set -e
(cd tools && ./index_experiments_static.sh)
python3 -m http.server
