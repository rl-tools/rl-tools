SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
git clone -b master https://huggingface.co/datasets/rl-tools/test-data $SCRIPT_DIR/data
cd $SCRIPT_DIR/data
git checkout 3a27cba4574d443845dbb1b8ceebf42898d4f50c
