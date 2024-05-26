if [ ! EXPERIMENTS ]; then
EXPERIMENTS=experiments
fi
find $EXPERIMENTS -type f | sort > $EXPERIMENTS/index.txt