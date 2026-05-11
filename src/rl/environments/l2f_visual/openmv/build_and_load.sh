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
INFERENCE_PY="$SCRIPT_DIR/inference.py"
BIN_LIMIT="${BIN_LIMIT:-2}"
VELA_INI="${VELA_INI:-$HOME/.config/OpenMV/openmvide/firmware/OPENMV_AE3/vela.ini}"

if [ ! -f "$CKPT" ]; then
    echo "missing checkpoint: $CKPT" >&2
    exit 1
fi

if [ ! -d "$MOUNT" ]; then
    echo "openmv mount not found: $MOUNT" >&2
    exit 1
fi

if [ ! -f "$INFERENCE_PY" ]; then
    echo "missing OpenMV inference script: $INFERENCE_PY" >&2
    exit 1
fi

CKPT_DIR="$(cd "$(dirname "$CKPT")" && pwd)"
CKPT_BASE="$CKPT_DIR/$(basename "${CKPT%.h5}")"

echo "==> wiping existing checkpoint/data from $MOUNT"
find "$MOUNT" -maxdepth 1 -type f \
    \( -name "*.tflite" -o -name "*.bin" -o -name "*.example_meta.json" \) \
    -print -delete

echo "==> converting $CKPT (bin samples: first $BIN_LIMIT)"
"$PY" "$REPO_ROOT/tools/hdf5_to_tflite.py" \
    --quantize int8 \
    --openmv-visual-split \
    --example-bin-limit "$BIN_LIMIT" \
    "$CKPT"

VISUAL_TFLITE="${CKPT_BASE}.visual.int8.tflite"
CONTROL_TFLITE="${CKPT_BASE}.control.int8.tflite"
if [ ! -f "$VISUAL_TFLITE" ]; then
    echo "visual int8 tflite not found: $VISUAL_TFLITE" >&2
    exit 1
fi
if [ ! -f "$CONTROL_TFLITE" ]; then
    echo "control int8 tflite not found: $CONTROL_TFLITE" >&2
    exit 1
fi
if [ ! -x "$VELA" ]; then
    echo "missing vela executable: $VELA" >&2
    exit 1
fi
if [ ! -f "$VELA_INI" ]; then
    echo "missing Vela ini: $VELA_INI" >&2
    exit 1
fi

echo "==> running vela on $VISUAL_TFLITE"
"$VELA" \
    --accelerator-config ethos-u55-256 \
    --config "$VELA_INI" \
    --system-config RTSS_HP_SRAM_OSPI \
    --memory-mode Shared_Sram \
    --output-dir "$CKPT_DIR" \
    "$VISUAL_TFLITE"

echo "==> running vela on $CONTROL_TFLITE"
"$VELA" \
    --accelerator-config ethos-u55-256 \
    --config "$VELA_INI" \
    --system-config RTSS_HP_SRAM_OSPI \
    --memory-mode Shared_Sram \
    --output-dir "$CKPT_DIR" \
    "$CONTROL_TFLITE"

VISUAL_VELA_OUT="${CKPT_BASE}.visual.int8_vela.tflite"
CONTROL_VELA_OUT="${CKPT_BASE}.control.int8_vela.tflite"
if [ ! -f "$VISUAL_VELA_OUT" ]; then
    echo "visual vela output not found: $VISUAL_VELA_OUT" >&2
    exit 1
fi
if [ ! -f "$CONTROL_VELA_OUT" ]; then
    echo "control vela output not found: $CONTROL_VELA_OUT" >&2
    exit 1
fi

echo "==> copying artifacts to $MOUNT"
cp -v "$VISUAL_VELA_OUT" "$MOUNT/"
cp -v "$CONTROL_VELA_OUT" "$MOUNT/"
cp -v "${CKPT_BASE}".example_input.*.bin "$MOUNT/"
cp -v "${CKPT_BASE}.example_output.bin" "$MOUNT/"
cp -v "${CKPT_BASE}.example_meta.json" "$MOUNT/"
cp -v "$INFERENCE_PY" "$MOUNT/main.py"
for extra in \
    example_int8_output.bin \
    example_int8_output_raw.bin \
    example_visual_embedding.bin \
    example_visual_embedding_raw.bin \
    example_split_int8_output.bin \
    example_split_int8_output_raw.bin
do
    src="${CKPT_BASE}.${extra}"
    if [ -f "$src" ]; then
        cp -v "$src" "$MOUNT/"
    fi
done

sync
echo "==> done"
