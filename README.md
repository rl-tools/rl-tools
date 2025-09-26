## Quick Start
Clone this repo, then build a Zoo example:
```
g++ -std=c++17 -O3 -ffast-math -I include src/rl/zoo/l2f/sac.cpp
```
Run it `./a.out 1337` (number = seed) then run `python3 -m http.server` to visualize the results. Open `http://localhost:8000` and navigate to the ExTrack UI to watch the quadrotor flying.

- **macOS**: Append `-framework Accelerate -DRL_TOOLS_BACKEND_ENABLE_ACCELERATE` for fast training (~4s on M3)
- **Ubuntu**: Use `apt install libopenblas-dev` and append `-lopenblas -DRL_TOOLS_BACKEND_ENABLE_OPENBLAS` (~6s on Zen 5).
