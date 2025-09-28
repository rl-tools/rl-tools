SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

rm -rf $SCRIPT_DIR/lib
git clone https://github.com/rl-tools/extrack-ui-lib.git $SCRIPT_DIR/lib
