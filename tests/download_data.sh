SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
git clone -b master https://huggingface.co/datasets/rl-tools/test-data $SCRIPT_DIR/data
cd $SCRIPT_DIR/data
git checkout 8dc80c4f07c00cc5b56f8b12fd4b15b80f0ae5f9
