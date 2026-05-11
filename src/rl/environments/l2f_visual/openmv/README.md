# OpenMV deployment

Single-checkpoint flow. The large HDF5 (e.g. `checkpoint_512examples.h5`) is
used for both int8 calibration and for the deployed companion bins; the number
of samples written to the bins is capped by `--example-bin-limit` to fit the
OpenMV flash budget.

## End-to-end (script)

```sh
src/rl/environments/l2f_visual/openmv/build_and_load.sh \
    <path>/checkpoint_<N>examples.h5 \
    /run/media/<you>/OPENMV
```

The script runs `hdf5_to_tflite.py --quantize int8 --openmv-visual-split`.
The converter quantizes the fused actor once, then prunes that int8 graph into
two deployment models:

```text
<checkpoint>.visual.int8.tflite   image stack -> visual embedding
<checkpoint>.control.int8.tflite  visual embedding + dense state -> action
```

The visual-output/control-input embedding tensor keeps the fused graph's exact
int8 scale and zero point, so the runtime raw-copies the cached embedding bytes
into the 500 Hz control model. It then runs Vela on both split models, wipes
stale artifacts from the mount, and copies the Vela-compiled split tflites plus
`.example_input.*.bin`, `.example_meta.json`, and the split-chain companion
check files. The runtime checks the split chain on boot, then runs the dense
control loop at 500 Hz while refreshing the visual embedding at 100 Hz.

## Manual steps

```sh
python3 tools/hdf5_to_tflite.py \
    --quantize int8 \
    --openmv-visual-split \
    --example-bin-limit 16 \
    <path>/checkpoint_<N>examples.h5

python3 -m ethosu.vela \
    --optimise Performance \
    --system-config RTSS_HP_SRAM_OSPI \
    --accelerator-config ethos-u55-256 \
    --memory-mode Shared_Sram \
    --config $HOME/.config/OpenMV/openmvide/firmware/OPENMV_AE3/vela.ini \
    --verbose-performance --verbose-cycle-estimate \
    --output-dir <path>/ \
    <path>/checkpoint_<N>examples.visual.int8.tflite

python3 -m ethosu.vela \
    --optimise Performance \
    --system-config RTSS_HP_SRAM_OSPI \
    --accelerator-config ethos-u55-256 \
    --memory-mode Shared_Sram \
    --config $HOME/.config/OpenMV/openmvide/firmware/OPENMV_AE3/vela.ini \
    --verbose-performance --verbose-cycle-estimate \
    --output-dir <path>/ \
    <path>/checkpoint_<N>examples.control.int8.tflite
```

Copy the Vela tflite + bin/json artifacts to the OpenMV mount, then open
`inference.py` in the OpenMV IDE and run.

```sh
mpremote connect /dev/serial/by-id/usb-OpenMV_OpenMV_Camera_085ce70000000000-if00 \
    run src/rl/environments/l2f_visual/openmv/inference.py
```

`inference.py` auto-detects the number of samples present in the bins and
iterates the visual-embedding + split-chain strict-wiring check across all of
them before starting the live 500 Hz control / 100 Hz vision policy loop.
