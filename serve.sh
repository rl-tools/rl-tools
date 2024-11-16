set -e

EXPERIMENTS_DIR=$1
if [ -z "$1" ]; then
    EXPERIMENTS_DIR=experiments
    echo "No path to the experiments directory provided. Using default: $EXPERIMENTS_DIR"
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
watch -n10 $SCRIPT_DIR/tools/index_experiments.sh experiments &
python3 -m http.server $@
