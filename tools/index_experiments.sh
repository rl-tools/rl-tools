#!/bin/bash

create_index_files() {
  local dir="$1"
  echo "Indexing $dir."
  local files_index="${dir}/index_files.txt"
  local directories_index="${dir}/index_directories.txt"
  find "$dir" -maxdepth 1 -type f ! -name 'index_files.txt' ! -name 'index_directories.txt' ! -name 'index_full.txt' !  -name '.*' -exec basename {} \; | grep -v '^$' > "$files_index"
  find "$dir" -maxdepth 1 -type d ! -path "$dir" ! -name 'index_files.txt' ! -name 'index_directories.txt' ! -name 'index_full.txt' ! -name '.*' -exec basename {} \; | grep -v '^$' > "$directories_index"
}

export -f create_index_files
start_dir="${1:-../experiments}"
find "$start_dir" -type d -exec bash -c 'create_index_files "$0"' {} \;
echo "Index files created successfully."
