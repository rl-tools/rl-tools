set -e
cd ../experiments
find . -type f -not -path '\./\.git/*' | grep -v index.txt\$ | grep -v index_files.txt\$ | grep -v index_directories.txt\$ | grep -v index_full.txt\$ | sort > index_full.txt
cat index_full.txt
