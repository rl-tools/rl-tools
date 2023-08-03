set -e
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
BACKPROP_TOOLS_ROOT=$(cd $SCRIPTPATH/../..; pwd -P)
echo $BACKPROP_TOOLS_ROOT

(cd /tmp && rm -rf backprop_tools_build_bare; mkdir backprop_tools_build_bare && cd backprop_tools_build_bare && cmake $BACKPROP_TOOLS_ROOT $ARGS && cmake --build . -j16)

echo "Success!"