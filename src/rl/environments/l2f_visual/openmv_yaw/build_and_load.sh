#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <h5_checkpoint> <openmv_mount>" >&2
    exit 1
fi

CKPT="$1"
MOUNT="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python3"
VELA="${VELA:-$REPO_ROOT/.venv/bin/vela}"
VELA_INI="${VELA_INI:-$HOME/.config/OpenMV/openmvide/firmware/OPENMV_AE3/vela.ini}"
BIN_LIMIT="${BIN_LIMIT:-1}"

if [ ! -x "$PY" ]; then
    echo "missing python in virtualenv: $PY" >&2
    echo "create it with: python3 -m venv .venv" >&2
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "missing checkpoint: $CKPT" >&2
    exit 1
fi
if [ ! -d "$MOUNT" ]; then
    echo "openmv mount not found: $MOUNT" >&2
    exit 1
fi
if [ ! -f "$SCRIPT_DIR/yaw.py" ]; then
    echo "missing OpenMV yaw script: $SCRIPT_DIR/yaw.py" >&2
    exit 1
fi
if [ ! -x "$VELA" ]; then
    echo "missing vela executable: $VELA" >&2
    echo "Set VELA=/path/to/vela or install ethos-u-vela in .venv." >&2
    exit 1
fi
if [ ! -f "$VELA_INI" ]; then
    echo "missing Vela ini: $VELA_INI" >&2
    echo "Set VELA_INI=/path/to/vela.ini if your OpenMV firmware path differs." >&2
    exit 1
fi

CKPT_DIR="$(cd "$(dirname "$CKPT")" && pwd)"
CKPT_BASE="$CKPT_DIR/$(basename "${CKPT%.h5}")"

echo "==> wiping existing checkpoint/data from $MOUNT"
find "$MOUNT" -maxdepth 1 -type f \
    \( -name "*.tflite" -o -name "*.bin" -o -name "*.example_meta.json" \) \
    -print -delete

echo "==> converting $CKPT (int8, bin samples: first $BIN_LIMIT)"
"$PY" "$REPO_ROOT/tools/hdf5_to_tflite.py" \
    --quantize int8 \
    --example-bin-limit "$BIN_LIMIT" \
    "$CKPT"

TFLITE_IN="${CKPT_BASE}.int8.tflite"
if [ ! -f "$TFLITE_IN" ]; then
    echo "int8 tflite not found: $TFLITE_IN" >&2
    exit 1
fi

echo "==> running vela on $TFLITE_IN"
"$VELA" \
    --accelerator-config ethos-u55-256 \
    --config "$VELA_INI" \
    --system-config RTSS_HP_SRAM_OSPI \
    --memory-mode Shared_Sram \
    --output-dir "$CKPT_DIR" \
    "$TFLITE_IN"

VELA_OUT="${CKPT_BASE}.int8_vela.tflite"
if [ ! -f "$VELA_OUT" ]; then
    echo "vela output not found: $VELA_OUT" >&2
    exit 1
fi

echo "==> copying artifacts to $MOUNT"
cp -v "$SCRIPT_DIR/yaw.py" "$MOUNT/yaw.py"
cp -v "$SCRIPT_DIR/yaw.py" "$MOUNT/main.py"
cp -v "$VELA_OUT" "$MOUNT/"
cp -v "${CKPT_BASE}".example_input.*.bin "$MOUNT/"
cp -v "${CKPT_BASE}.example_output.bin" "$MOUNT/"
cp -v "${CKPT_BASE}.example_int8_output.bin" "$MOUNT/"
if [ -f "${CKPT_BASE}.example_int8_output_raw.bin" ]; then
    cp -v "${CKPT_BASE}.example_int8_output_raw.bin" "$MOUNT/"
fi
cp -v "${CKPT_BASE}.example_meta.json" "$MOUNT/"

sync
echo "==> done"
