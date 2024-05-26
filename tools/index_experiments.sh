set -e
cd ../experiments
find . -type f -not -path '\./\.git/*' -not -path \./index.txt | sort > index.txt
cat index.txt
