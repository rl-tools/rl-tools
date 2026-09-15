#pragma once

#include "training_mixed_parameters.h"

namespace rl_tools::rl::environments::l2f_visual::training_mixed{
    using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<rlt::devices::DefaultCUDASpecification>;
    using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;

    using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, STATIC_PARAMETERS, N_ENVIRONMENTS_PER_SCENE, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES, RENDER_SHADING, RENDER_ENABLE_MOTION_BLUR, RENDER_MOTION_BLUR_SAMPLES, RENDER_ENABLE_ANTI_ALIASING, RENDER_ANTI_ALIASING_GRID_SIZE>;
    static_assert(VISUAL_SPEC::RENDERER_SPEC::SHADING::PBR_SHADING, "l2f visual training must use the High renderer profile");
    using ENVIRONMENT = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;
    using CAMERA_DATA = rlt::rendering::raytracing::Camera<T>;
    static constexpr bool RENDER_MOTION_BLUR_ACTIVE = ENVIRONMENT::SPEC::RENDERER_SPEC::ENABLE_MOTION_BLUR;
    static constexpr bool RENDER_ANTI_ALIASING_ACTIVE = ENVIRONMENT::SPEC::RENDERER_SPEC::ENABLE_ANTI_ALIASING;
    static_assert(ENVIRONMENT::Observation::HEIGHT == CAM_HEIGHT && ENVIRONMENT::Observation::WIDTH == CAM_WIDTH && ENVIRONMENT::Observation::CHANNELS == IMAGE_CHANNELS, "The visual environment must produce the configured camera frames");

    using LOOP_CORE_PARAMETERS = LoopParameters<ENVIRONMENT>;

    using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, ConfigApproximators>;

    using PPO_SPEC = typename LOOP_CORE_CONFIG::PPO_SPEC;
    using PPO_TYPE = typename LOOP_CORE_CONFIG::PPO_TYPE;
    using PPO_BUFFERS_TYPE = typename LOOP_CORE_CONFIG::PPO_BUFFERS_TYPE;
    using ON_POLICY_RUNNER_SPEC = typename LOOP_CORE_CONFIG::ON_POLICY_RUNNER_SPEC;
    using ON_POLICY_RUNNER_DATASET_SPEC = typename LOOP_CORE_CONFIG::ON_POLICY_RUNNER_DATASET_SPEC;
    using ON_POLICY_RUNNER_DATASET_TYPE = typename LOOP_CORE_CONFIG::ON_POLICY_RUNNER_DATASET_TYPE;
    using ACTOR_OPTIMIZER = typename LOOP_CORE_CONFIG::NN::ACTOR_OPTIMIZER;
    using CRITIC_OPTIMIZER = typename LOOP_CORE_CONFIG::NN::CRITIC_OPTIMIZER;
    using ACTOR_BUFFERS = typename LOOP_CORE_CONFIG::ACTOR_BUFFERS;
    using CRITIC_BUFFERS = typename LOOP_CORE_CONFIG::CRITIC_BUFFERS;
    using ACTOR_TYPE = typename LOOP_CORE_CONFIG::NN::ACTOR_TYPE;

    using CAPABILITY_ROLLOUT = rlt::nn::capability::Forward<true>;
    using ROLLOUT_ACTOR_TYPE = typename ACTOR_TYPE::template CHANGE_CAPABILITY<CAPABILITY_ROLLOUT>::template CHANGE_BATCH_SIZE<TI, N_ENVIRONMENTS>;
    using ROLLOUT_ACTOR_BUFFERS = typename ROLLOUT_ACTOR_TYPE::template Buffer<true>;
    using CHECKPOINT_ACTOR_TYPE = typename ACTOR_TYPE::template CHANGE_CAPABILITY<CAPABILITY_ROLLOUT>::template CHANGE_BATCH_SIZE<TI, N_EXAMPLES>;
    static constexpr TI STEPS_PER_ENV = LOOP_CORE_PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV;
    static constexpr TI BATCH_SIZE = LOOP_CORE_PARAMETERS::BATCH_SIZE;
    static constexpr TI OBSERVATION_DIM = ENVIRONMENT::OBSERVATION_DIM;
    static constexpr TI OBS_PRIV_DIM = ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED;
    static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
    static constexpr TI STEPS_TOTAL = ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL;
    static constexpr TI STEPS_TOTAL_ALL = ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL_ALL;
    static constexpr TI N_EPOCHS = LOOP_CORE_PARAMETERS::PPO_PARAMETERS::N_EPOCHS;
    static constexpr TI N_BATCHES = STEPS_TOTAL / BATCH_SIZE;
    using MIXED_SPEC = mixed_imitation::Specification<TYPE_POLICY, TI, BATCH_SIZE, N_ENVIRONMENTS_PER_SCENE, MIXED_TRAINING ? MIXED_PARAMETERS::IMITATION_ENVIRONMENTS_PER_SCENE : 0, MIXED_PARAMETERS>;
    static constexpr TI RL_BATCH_SIZE = MIXED_SPEC::RL_BATCH_SIZE;
    static constexpr TI RL_STEPS_TOTAL = N_BATCHES * RL_BATCH_SIZE;
    static constexpr TI RL_UPDATE_SAMPLES = RL_STEPS_TOTAL * GRADIENT_ACCUMULATION_ROLLOUTS;
    static constexpr TI UPDATE_SAMPLES = STEPS_TOTAL * GRADIENT_ACCUMULATION_ROLLOUTS;
    static constexpr T GRADIENT_SCALE = static_cast<T>(BATCH_SIZE) / static_cast<T>(UPDATE_SAMPLES);

    static_assert(N_BATCHES > 0, "STEPS_TOTAL must be >= BATCH_SIZE");
    static_assert(STEPS_TOTAL % BATCH_SIZE == 0, "Every rollout sample must be consumed exactly once");
    static_assert(N_EPOCHS == 1, "On-policy accumulation does not reuse rollout samples");
    static_assert(GRADIENT_ACCUMULATION_ROLLOUTS == 1 || ROLLOUTS_PER_SCENE_SET == 1, "Multiple accumulation groups require resets at each rollout boundary");
    static_assert(N_ENVIRONMENTS * GRADIENT_ACCUMULATION_ROLLOUTS <= RNG_GPU::NUM_RNGS, "Each logical environment needs an independent RNG stream");
    static_assert(N_EXAMPLES <= BATCH_SIZE, "N_EXAMPLES must fit the reusable combined-observation batch buffer");
    static_assert(N_EXAMPLES <= STEPS_TOTAL, "N_EXAMPLES must fit one PPO rollout dataset");
    static_assert(!MIXED_TRAINING || BATCH_SIZE % N_ENVIRONMENTS == 0, "Every mixed minibatch must cover all scenes with the same IL/RL split");

}
