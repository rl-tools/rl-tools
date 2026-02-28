SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
git clone -b master https://huggingface.co/datasets/rl-tools/test-data $SCRIPT_DIR/data
cd $SCRIPT_DIR/data
git checkout 6188b2b770ec8415f00257ca5f822c883aebe24a
