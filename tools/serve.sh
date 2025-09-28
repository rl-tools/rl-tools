# serve the main folder
set -e
(cd )
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
bash -c "while true; do $SCRIPT_DIR/index_experiments.sh $PARENT_DIR/experiments; sleep 10; done" &
python3 -m http.server -d "$PARENT_DIR"