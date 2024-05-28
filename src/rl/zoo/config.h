#if defined(RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
#if defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
using LOOP_CORE_CONFIG = rlt::rl::zoo::sac::PendulumV1<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
#else
#error "RLtools Zoo SAC: Environment not defined"
#endif
#elif defined(RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
#if defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
using LOOP_CORE_CONFIG = rlt::rl::zoo::td3::PendulumV1<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
#else
#error "RLtools Zoo TD3: Environment not defined"
#endif
#elif defined(RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
#if defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
using LOOP_CORE_CONFIG = rlt::rl::zoo::ppo::PendulumV1<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_ANT_V4)
using LOOP_CORE_CONFIG = rlt::rl::zoo::ppo::AntV4<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
#else
#error "RLtools Zoo PPO: Environment not defined"
#endif
#else
#error "RLtools Zoo: Algorithm not defined"
#endif

constexpr TI NUM_CHECKPOINTS = 10;
constexpr TI NUM_EVALUATIONS = 100;
constexpr TI NUM_SAVE_TRAJECTORIES = 10;
using LOOP_EXTRACK_CONFIG = rlt::rl::loop::steps::extrack::Config<LOOP_CORE_CONFIG>;
struct LOOP_CHECKPOINT_PARAMETERS: rlt::rl::loop::steps::checkpoint::Parameters<T, TI>{
    static constexpr TI CHECKPOINT_INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_CHECKPOINTS;
    static constexpr TI CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL_TEMP == 0 ? 1 : CHECKPOINT_INTERVAL_TEMP;
};
using LOOP_CHECKPOINT_CONFIG = rlt::rl::loop::steps::checkpoint::Config<LOOP_EXTRACK_CONFIG, LOOP_CHECKPOINT_PARAMETERS>;
struct LOOP_EVALUATION_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>{
    static constexpr TI EVALUATION_INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_EVALUATIONS;
    static constexpr TI EVALUATION_INTERVAL = EVALUATION_INTERVAL_TEMP == 0 ? 1 : EVALUATION_INTERVAL_TEMP;
    static constexpr TI NUM_EVALUATION_EPISODES = 100;
    static constexpr TI N_EVALUATIONS = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};
using LOOP_EVALUATION_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CHECKPOINT_CONFIG, LOOP_EVALUATION_PARAMETERS>;
struct LOOP_SAVE_TRAJECTORIES_PARAMETERS: rlt::rl::loop::steps::save_trajectories::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>{
    static constexpr TI INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_SAVE_TRAJECTORIES;
    static constexpr TI INTERVAL = INTERVAL_TEMP == 0 ? 1 : INTERVAL_TEMP;
    static constexpr TI NUM_EPISODES = 10;
};
using LOOP_SAVE_TRAJECTORIES_CONFIG = rlt::rl::loop::steps::save_trajectories::Config<LOOP_EVALUATION_CONFIG, LOOP_SAVE_TRAJECTORIES_PARAMETERS>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_SAVE_TRAJECTORIES_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
