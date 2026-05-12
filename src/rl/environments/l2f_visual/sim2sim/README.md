```sh
pip install -e ~/git/flysplat
```

```sh
FLYSPLAT_ROOT=~/git/flysplat
RL_TOOLS_ROOT=~/rl-tools3
CHECKPOINT=/home/jonas/mount/arpl-server/home/jonas/mnt/experiments/2026-05-11_14-43-37/05a5d1d_l2f_visual_training_cuda_default/default/0000/steps/000000157548544/checkpoint_512examples.h5

python3 -m flysplat.eval_l2f_tflite \
  --checkpoint "$CHECKPOINT" \
  --quantize int8 \
  --converter-python "$RL_TOOLS_ROOT/.venv/bin/python" \
  --converter-script "$RL_TOOLS_ROOT/tools/hdf5_to_tflite.py" \
  --scene "$FLYSPLAT_ROOT/data/room/point_cloud/iteration_30000/point_cloud.ply" \
  --episodes 1 \
  --steps 5000 \
  --sim-rate 100 \
  --dynamics-model crazyflie_openmv \
  --device cuda \
  --start 0 0 0 \
  --target 0 0 0 \
  --target-yaw 0 \
  --validate-examples 1 \
  --video-every 1 \
  --out-dir ./runs/test
```

Use `--dynamics-model rotorpy_crazyflie` to compare against the old RotorPY-backed Crazyflie dynamics.
