SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
git clone -b master https://huggingface.co/datasets/rl-tools/test-data $SCRIPT_DIR/data
cd $SCRIPT_DIR/data
git checkout ea997d044860f67e852ed92217be95fde00be165
