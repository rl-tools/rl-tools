```sh
python3 tools/hdf5_to_tflite.py --quantize int8 /home/jonas/mono/rl-tools/experiments/2026-04-18_21-44-34/0240596_l2f_visual_imitation_cuda_default/default/0100/steps/000000512064000/checkpoint_2.h5
python3 tools/hdf5_to_tflite.py --quantize int8 /home/jonas/mono/rl-tools/experiments/2026-04-18_21-44-34/0240596_l2f_visual_imitation_cuda_default/default/0100/steps/000000512064000/checkpoint_512.h5
```

In the OpenMV IDE: Tools -> Machine Vision -> Compile model for NPU -> High performance core -> select checkpoint_512.int8.tflite -> save as `checkpoint_512_vela.tflite`. Transfer vela output tflite and `checkpoint_2.example_input.0.bin`, `checkpoint_2.example_input.1.bin`, `checkpoint_2.example_output.bin` and  `checkpoint_2.example_meta.json` to the OpenMV.
Open `inference.py` in the OpenMV IDE and run.


```
python3 -m ethosu.vela --optimise Performance --system-config RTSS_HP_SRAM_OSPI --accelerator-config ethos-u55-256 --memory-mode Shared_Sram --config /home/jonas/.config/OpenMV/openmvide/firmware/OPENMV_AE3/vela.ini --verbose-performance --verbose-cycle-estimate --output-dir /home/jonas/mono/rl-tools/experiments/2026-04-19_23-14-14/0240596_l2f_visual_imitation_cuda_default/default/0100/steps/000000000064000/ /home/jonas/mono/rl-tools/experiments/2026-04-19_23-14-14/0240596_l2f_visual_imitation_cuda_default/default/0100/steps/000000000064000/checkpoint_512examples.int8.tflite
```

/home/jonas/mono/rl-tools/experiments/2026-04-19_23-14-14/0240596_l2f_visual_imitation_cuda_default/default/0100/


```
mpremote connect /dev/serial/by-id/usb-OpenMV_OpenMV_Camera_085ce70000000000-if00 run src/rl/environments/l2f_visual/openmv/inference.py
```