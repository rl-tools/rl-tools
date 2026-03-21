SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
git clone -b master "${RL_TOOLS_TEST_DATA_SOURCE:-https://huggingface.co/datasets/rl-tools/test-data}" $SCRIPT_DIR/data
cd $SCRIPT_DIR/data
git pull
git checkout 517ef18fc7876d04b7ac5815ee69ce17f3d1f50f
