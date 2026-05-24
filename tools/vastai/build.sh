set -e
git clone https://github.com/rl-tools/rl-tools-internal.git rl-tools || true
docker build . -t rltools/vast-snapshot:latest --progress=plain
docker push rltools/vast-snapshot:latest
