SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
git clone -b master "${RL_TOOLS_TEST_DATA_SOURCE:-https://huggingface.co/datasets/rl-tools/test-data}" $SCRIPT_DIR/data
cd $SCRIPT_DIR/data
git pull
git checkout d39a1ede85a505af01afd897aa51779f87db3cc1
