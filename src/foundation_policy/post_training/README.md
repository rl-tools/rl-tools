```
find experiments/2025-03-25_19-19-11 -type f | grep return.json\$ | xargs -I{} -P$(nproc) bash -c 'echo {} $(jq ".[length-1].episode_length_mean" {})' | p2s.sort 'x.split(" ")[1]' | awk '{print $1}' | xargs -I{} dirname {} | xargs -I{} dirname {} | xargs -I{} basename {} > src/foundation_policy/checkpoints.txt
```
