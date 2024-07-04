#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>


#if defined(RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
#if defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
using LOOP_CORE_CONFIG = rlt::rl::zoo::sac::pendulum_v1::PendulumV1<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{}; // no-op, this allows to have a different EPISODE_STEP_LIMIT for training and evaluation (on a per algorithm&environment baseis)
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_ACROBOT_SWINGUP_V0)
using LOOP_CORE_CONFIG = rlt::rl::zoo::sac::acrobot_swingup_v0::AcrobotSwingupV0<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
using LOOP_EVALUATION_PARAMETER_OVERWRITES = rlt::rl::zoo::sac::acrobot_swingup_v0::AcrobotSwingupV0<DEVICE, T, TI, RNG>::LOOP_EVALUATION_PARAMETER_OVERWRITES<BASE>;
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F)
using LOOP_CORE_CONFIG = rlt::rl::zoo::sac::l2f::LearningToFly<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{};
#else
#error "RLtools Zoo SAC: Environment not defined"
#endif
#elif defined(RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
#if defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
using LOOP_CORE_CONFIG = rlt::rl::zoo::td3::pendulum_v1::PendulumV1<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{}; // no-op
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F)
using LOOP_CORE_CONFIG = rlt::rl::zoo::td3::l2f::LearningToFly<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{}; // no-op
#else
#error "RLtools Zoo TD3: Environment not defined"
#endif
#elif defined(RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
#if defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
using LOOP_CORE_CONFIG = rlt::rl::zoo::ppo::pendulum_v1::PendulumV1<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{}; // no-op
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_BOTTLENECK_V0)
using LOOP_CORE_CONFIG = rlt::rl::zoo::ppo::bottleneck_v0::BottleneckV0<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
using LOOP_EVALUATION_PARAMETER_OVERWRITES = rlt::rl::zoo::ppo::bottleneck_v0::BottleneckV0<DEVICE, T, TI, RNG>::LOOP_EVALUATION_PARAMETER_OVERWRITES<BASE>;
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_ANT_V4)
using LOOP_CORE_CONFIG = rlt::rl::zoo::ppo::ant_v4::AntV4<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{}; // no-op
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
struct LOOP_EVALUATION_PARAMETERS: LOOP_EVALUATION_PARAMETER_OVERWRITES<rlt::rl::loop::steps::evaluation::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>>{
    static constexpr TI EVALUATION_INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_EVALUATIONS;
    static constexpr TI EVALUATION_INTERVAL = EVALUATION_INTERVAL_TEMP == 0 ? 1 : EVALUATION_INTERVAL_TEMP;
    static constexpr TI NUM_EVALUATION_EPISODES = 10;
    static constexpr TI N_EVALUATIONS = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};
using LOOP_EVALUATION_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CHECKPOINT_CONFIG, LOOP_EVALUATION_PARAMETERS>;
struct LOOP_SAVE_TRAJECTORIES_PARAMETERS: LOOP_EVALUATION_PARAMETER_OVERWRITES<rlt::rl::loop::steps::save_trajectories::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>>{
    static constexpr TI INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_SAVE_TRAJECTORIES;
    static constexpr TI INTERVAL = INTERVAL_TEMP == 0 ? 1 : INTERVAL_TEMP;
    static constexpr TI NUM_EPISODES = 10;
};
using LOOP_SAVE_TRAJECTORIES_CONFIG = rlt::rl::loop::steps::save_trajectories::Config<LOOP_EVALUATION_CONFIG, LOOP_SAVE_TRAJECTORIES_PARAMETERS>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_SAVE_TRAJECTORIES_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
