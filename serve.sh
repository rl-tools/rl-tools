set -e
bash -c "cd tools && watch -n10 ./index_experiments_static.sh" &
python3 -m http.server $@
