struct OPTIONS_POST_TRAINING: OPTIONS_PRE_TRAINING{
    static constexpr bool OBSERVE_THRASH_MARKOV = false;
    static constexpr bool MOTOR_DELAY = true;
    static constexpr bool ACTION_HISTORY = true;
    static constexpr TI ACTION_HISTORY_LENGTH = 1;
    static constexpr bool OBSERVATION_NOISE = true;
};


static_assert(sizeof(TI) == 8);
// constants parameters
constexpr TI NUM_EPISODES = 10;
constexpr TI NUM_EPISODES_EVAL = 100;
constexpr TI N_EPOCH = 1000;
constexpr TI N_PRE_TRAINING_SEEDS = 1;
constexpr TI SEQUENCE_LENGTH = 500;
constexpr TI BATCH_SIZE = 64;
constexpr T SOLVED_RETURN = 300;
constexpr TI HIDDEN_DIM = 32;
constexpr TI NUM_TEACHERS = 1000;
constexpr TI NUM_ACTIVE_TEACHERS = NUM_TEACHERS;
constexpr TI EPOCH_DAGGER = 10;
constexpr bool DYNAMIC_ALLOCATION = true;
constexpr bool SHUFFLE = true;
constexpr bool TEACHER_DETERMINISTIC = true;
constexpr bool ON_POLICY = true;
constexpr TI TEACHER_STUDENT_MIX = 0; // added teacher epochs in DAgger epochs

// typedefs
using ENVIRONMENT = typename builder::ENVIRONMENT_FACTORY_POST_TRAINING<DEVICE, T, TI, OPTIONS_POST_TRAINING>::ENVIRONMENT;
#ifdef RL_TOOLS_POST_TRAINING
struct ENVIRONMENT_TEACHER_STATIC_PARAMETERS: ENVIRONMENT::SPEC::STATIC_PARAMETERS{
    using LOOP_CORE_CONFIG_PRE_TRAINING = builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS_PRE_TRAINING, DYNAMIC_ALLOCATION>::LOOP_CORE_CONFIG;
    using ENV = LOOP_CORE_CONFIG_PRE_TRAINING::ENVIRONMENT;
    using OBSERVATION_TYPE = ENV::Observation;
    using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
    static constexpr auto PARAMETER_VALUES = [](){
        auto params = ENVIRONMENT::SPEC::STATIC_PARAMETERS::PARAMETER_VALUES;
        params.mdp.observation_noise.position = 0;
        params.mdp.observation_noise.orientation = 0;
        params.mdp.observation_noise.linear_velocity = 0;
        params.mdp.observation_noise.angular_velocity = 0;
        params.mdp.observation_noise.imu_acceleration = 0;
        return params;
    }();
};
using ENVIRONMENT_TEACHER_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, ENVIRONMENT_TEACHER_STATIC_PARAMETERS>;
using ENVIRONMENT_TEACHER = rl_tools::rl::environments::Multirotor<ENVIRONMENT_TEACHER_SPEC>;
#endif

template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential::OutputModule>
using Module = typename rlt::nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;

// using MLP_CONFIG = rlt::nn_models::mlp::Configuration<T, TI, ENVIRONMENT::ACTION_DIM, 3, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::FAST_TANH, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
// using MLP = rlt::nn_models::mlp::BindConfiguration<MLP_CONFIG>;
// using MODULE_CHAIN = Module<MLP>;

using INPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU>;
using INPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
using GRU_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, HIDDEN_DIM>;
using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;
using OUTPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, ENVIRONMENT::ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
using OUTPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<OUTPUT_LAYER_CONFIG>;
using MODULE_CHAIN = Module<INPUT_LAYER, Module<GRU, Module<OUTPUT_LAYER>>>;


using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, DYNAMIC_ALLOCATION>;
using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
using ACTOR = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
    static constexpr T ALPHA = 0.0001;
};
using OPTIMIZER = rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<T, TI, ADAM_PARAMETERS>>;
using OUTPUT_SHAPE = ACTOR::OUTPUT_SHAPE;
using RESULT = rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES, ENVIRONMENT::EPISODE_STEP_LIMIT>>;
using RESULT_EVAL = rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES_EVAL, ENVIRONMENT::EPISODE_STEP_LIMIT>>;
using DATA = rlt::rl::utils::evaluation::Data<rlt::rl::utils::evaluation::DataSpecification<RESULT::SPEC>>;
using DATA_EVAL = rlt::rl::utils::evaluation::NoData<rlt::rl::utils::evaluation::DataSpecification<RESULT_EVAL::SPEC>>;
