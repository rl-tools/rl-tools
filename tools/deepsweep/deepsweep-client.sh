#!/usr/bin/env bash
set -e
SEED=$(jq -e ".seed" <<< "$DEEPSWEEP_SPEC")
DMODEL=$(jq -e ".dmodel" <<< "$DEEPSWEEP_SPEC")
echo DModel $DMODEL, Seed $SEED

EXPERIMENT=deepsweep_runs/$DEEPSWEEP_JOB/$DMODEL/$SEED
mkdir -p $EXPERIMENT
MACOS_OPTS="-I /opt/homebrew/include -L /opt/homebrew/lib -DRL_TOOLS_BACKEND_ENABLE_ACCELERATE -framework Accelerate"
LINUX_OPTS="-I /usr/include/hdf5/serial/ -L/usr/lib/x86_64-linux-gnu/hdf5/serial -DRL_TOOLS_BACKEND_ENABLE_OPENBLAS -lopenblas"
# check os type
OS=$(uname -s)
OPTS=$LINUX_OPTS
if [ "$OS" == "Darwin" ]; then
  OPTS=$MACOS_OPTS
fi

g++ -std=c++17 -O3 -ffast-math -I include -I external/highfive/include -DRL_TOOLS_ENABLE_JSON -DRL_TOOLS_DISABLE_INTERMEDIATE_CHECKPOINTS -DRL_TOOLS_ENABLE_HDF5 -DDMODEL=$DMODEL src/foundation_policy/post_training/main.cpp $OPTS -lhdf5 -o $EXPERIMENT/a.out
CMD="MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RL_TOOLS_EXTRACK_EXPERIMENT=$JOB_ID RL_TOOLS_RUN_PATH=$EXPERIMENT ./$EXPERIMENT/a.out $SEED"
echo "Executing: $CMD"
eval $CMD

TEST_STATS_PATH=$EXPERIMENT/test_stats.csv
jq -n --rawfile csv "$TEST_STATS_PATH" '{test_stats: $csv}' | curl -sS -X POST -H 'Content-Type: application/json' --data @- "$DEEPSWEEP_SERVER/jobs/$DEEPSWEEP_JOB/tasks/$DEEPSWEEP_TASK_ID"
