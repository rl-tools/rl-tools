add_subdirectory(pendulum)
if(BACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO)
    add_subdirectory(mujoco)
endif()
