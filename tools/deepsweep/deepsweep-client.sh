#!/usr/bin/env bash
set -e
SEED=$(jq -e ".seed" <<< "$DEEPSWEEP_SPEC")
DMODEL=$(jq -e ".dmodel" <<< "$DEEPSWEEP_SPEC")
echo DModel $DMODEL, Seed $SEED

EXPERIMENT=deepsweep_runs/$DEEPSWEEP_JOB/$DMODEL/$SEED
mkdir -p $EXPERIMENT
g++ -std=c++17 -O3 -ffast-math -I include -I external/highfive/include -I /opt/homebrew/include -L /opt/homebrew/lib -lhdf5 -framework Accelerate -DRL_TOOLS_ENABLE_JSON -DRL_TOOLS_ENABLE_HDF5 -DRL_TOOLS_BACKEND_ENABLE_ACCELERATE -DDMODEL=$DMODEL src/foundation_policy/post_training/main.cpp -o $EXPERIMENT/a.out
CMD="MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RL_TOOLS_EXTRACK_EXPERIMENT=$JOB_ID RL_TOOLS_RUN_PATH=$EXPERIMENT ./$EXPERIMENT/a.out $SEED"
echo "Executing: $CMD"
eval $CMD

TEST_STATS_PATH=$EXPERIMENT/test_stats.csv
jq -n --rawfile csv "$TEST_STATS_PATH" '{test_stats: $csv}' | curl -sS -X POST -H 'Content-Type: application/json' --data @- "$DEEPSWEEP_SERVER/jobs/$DEEPSWEEP_JOB/tasks/$DEEPSWEEP_TASK_ID"