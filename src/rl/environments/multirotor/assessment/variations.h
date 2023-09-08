#include <backprop_tools/rl/environments/multirotor/multirotor.h>

namespace variations {
    namespace bpt = BACKPROP_TOOLS_NAMESPACE;
    template <typename TI, typename T_ENVIRONMENT, typename T_RNG>
    struct Specification{
        using ENVIRONMENT = T_ENVIRONMENT;
        using T = typename ENVIRONMENT::T;
        using RNG = T_RNG;
        static constexpr TI MAX_EPISODE_LENGTH = 1000;
    };
    namespace init{
        template <typename DEVICE, typename SPEC>
        void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::simple<typename SPEC::ENVIRONMENT::T, typename DEVICE::index_t, 4, typename SPEC::ENVIRONMENT::REWARD_FUNCTION>;
        }
        template <typename DEVICE, typename SPEC>
        void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::orientation_small_angle<typename SPEC::ENVIRONMENT::T, typename DEVICE::index_t, 4, typename SPEC::ENVIRONMENT::REWARD_FUNCTION>;
        }
        template <typename DEVICE, typename SPEC>
        void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::orientation_big_angle<typename SPEC::ENVIRONMENT::T, typename DEVICE::index_t, 4, typename SPEC::ENVIRONMENT::REWARD_FUNCTION>;
        }
        template <typename DEVICE, typename SPEC>
        void variation_3(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::orientation_bigger_angle<typename SPEC::ENVIRONMENT::T, typename DEVICE::index_t, 4, typename SPEC::ENVIRONMENT::REWARD_FUNCTION>;
        }
        template <typename DEVICE, typename SPEC>
        void variation_4(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::orientation_biggest_angle<typename SPEC::ENVIRONMENT::T, typename DEVICE::index_t, 4, typename SPEC::ENVIRONMENT::REWARD_FUNCTION>;
        }
        template <typename DEVICE, typename SPEC>
        void variation_5(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::orientation_all_around<typename SPEC::ENVIRONMENT::T, typename DEVICE::index_t, 4, typename SPEC::ENVIRONMENT::REWARD_FUNCTION>;
        }
    }
    namespace observation_noise{
        namespace position{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.position = 0.05;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.position = 0.1;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.position = 0.2;
            }
        }
        namespace orientation{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.orientation = 0.05;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.orientation = 0.1;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.orientation = 0.2;
            }
        }
        namespace linear_velocity{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.linear_velocity = 0.05;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.linear_velocity = 0.1;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.linear_velocity = 0.2;
            }
        }
        namespace angular_velocity{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.angular_velocity = 0.1;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.angular_velocity = 0.5;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.angular_velocity = 1;
            }
        }
    }
    namespace action_noise{
        template <typename DEVICE, typename SPEC>
        void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.action_noise.normalized_rpm = 0.1;
        }
        template <typename DEVICE, typename SPEC>
        void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.action_noise.normalized_rpm = 0.2;
        }
        template <typename DEVICE, typename SPEC>
        void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.action_noise.normalized_rpm = 0.5;
        }
    }
    namespace dynamics{
        namespace inertia{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T J_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.8, (T)1.2, rng);
                env.parameters.dynamics.J[0][0] *= J_factor;
                env.parameters.dynamics.J[1][1] *= J_factor;
                env.parameters.dynamics.J[2][2] *= J_factor;
                env.parameters.dynamics.J_inv[0][0] /= J_factor;
                env.parameters.dynamics.J_inv[1][1] /= J_factor;
                env.parameters.dynamics.J_inv[2][2] /= J_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T J_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.5, (T)2, rng);
                env.parameters.dynamics.J[0][0] *= J_factor;
                env.parameters.dynamics.J[1][1] *= J_factor;
                env.parameters.dynamics.J[2][2] *= J_factor;
                env.parameters.dynamics.J_inv[0][0] /= J_factor;
                env.parameters.dynamics.J_inv[1][1] /= J_factor;
                env.parameters.dynamics.J_inv[2][2] /= J_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T J_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.2, (T)5, rng);
                env.parameters.dynamics.J[0][0] *= J_factor;
                env.parameters.dynamics.J[1][1] *= J_factor;
                env.parameters.dynamics.J[2][2] *= J_factor;
                env.parameters.dynamics.J_inv[0][0] /= J_factor;
                env.parameters.dynamics.J_inv[1][1] /= J_factor;
                env.parameters.dynamics.J_inv[2][2] /= J_factor;
            }
        }
        namespace mass{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T mass_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.8, (T)1.2, rng);
                env.parameters.dynamics.mass *= mass_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T mass_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.7, (T)1.5, rng);
                env.parameters.dynamics.mass *= mass_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T mass_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.5, (T)2, rng);
                env.parameters.dynamics.mass *= mass_factor;
            }
        }
        namespace max_rpm{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T max_rpm_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.8, (T)1.2, rng);
                env.parameters.dynamics.action_limit.max *= max_rpm_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T max_rpm_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.7, (T)1.5, rng);
                env.parameters.dynamics.action_limit.max *= max_rpm_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T max_rpm_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.5, (T)2, rng);
                env.parameters.dynamics.action_limit.max *= max_rpm_factor;
            }
        }
        namespace rpm_time_constant{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T rpm_time_constant_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.8, (T)1.2, rng);
                env.parameters.dynamics.rpm_time_constant *= rpm_time_constant_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T rpm_time_constant_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.7, (T)1.5, rng);
                env.parameters.dynamics.rpm_time_constant *= rpm_time_constant_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T rpm_time_constant_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.5, (T)2, rng);
                env.parameters.dynamics.rpm_time_constant *= rpm_time_constant_factor;
            }
        }


    }

    template <typename DEVICE, typename SPEC>
    void base(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
        action_noise::variation_0<DEVICE, SPEC>(env, rng);
        init::variation_1<DEVICE, SPEC>(env, rng);
    }

}
