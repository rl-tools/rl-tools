set -e
watch -n10 ./tools/index_experiments_static.sh experiments &
python3 -m http.server $@
