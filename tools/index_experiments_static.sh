set -e
cd ../experiments
find . -type f -not -path '\./\.git/*' | grep -v index.txt\$ | grep -v index_files.txt\$ | grep -v index_directories.txt\$ | grep -v index_static.txt\$ | sort > index_static.txt
cat index_static.txt
