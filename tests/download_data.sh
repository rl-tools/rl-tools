SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
git clone -b master https://huggingface.co/datasets/rl-tools/test-data $SCRIPT_DIR/data
cd $SCRIPT_DIR/data
git checkout ba81c1f298c68c8c5bfe6b61a7017a0b482f1c45