# Compilation & Numerics 
- Try disabling `-ffast-math` (`-DRL_TOOLS_ENABLE_FAST_MATH:BOOL=OFF`)
- Try disabling aligned malloc (`#define RL_TOOLS_DISABLE_ALIGNED_MEMORY_ALLOCATIONS`)
- Use different compilers and different versions to assess different error messages
# RL
- Check the action distribution of the actor after random initialization
# CUDA
- Make sure nothing is passed by reference into any kernel