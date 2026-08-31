struct OPTIONS_POST_TRAINING: OPTIONS_PRE_TRAINING{
    static constexpr bool OBSERVE_THRASH_MARKOV = false;
    static constexpr bool MOTOR_DELAY = true;
    static constexpr bool ACTION_HISTORY = true;
    static constexpr TI ACTION_HISTORY_LENGTH = 1;
    static constexpr bool OBSERVATION_NOISE = true;
};


static_assert(sizeof(TI) == 8);
// constants parameters
#ifdef RL_TOOLS_NUM_EPISODES
#warning "Using RL_TOOLS_NUM_EPISODES for number of episodes"
constexpr TI NUM_EPISODES = RL_TOOLS_NUM_EPISODES;
#else
constexpr TI NUM_EPISODES = 10;
#endif
constexpr TI NUM_EPISODES_EVAL = 100;
constexpr TI N_EPOCH = 1000;
constexpr TI N_PRE_TRAINING_SEEDS = 1;
constexpr TI SEQUENCE_LENGTH = 500;
constexpr TI BATCH_SIZE = 64;
constexpr T SOLVED_RETURN = 300;
#ifdef RL_TOOLS_DMODEL
#warning "Using RL_TOOLS_DMODEL for hidden dimension"
constexpr TI HIDDEN_DIM = RL_TOOLS_DMODEL;
#else
constexpr TI HIDDEN_DIM = 16;
#endif
#ifdef RL_TOOLS_NUM_TEACHERS
#warning "Using RL_TOOLS_NUM_TEACHERS for number of teachers"
constexpr TI NUM_TEACHERS = RL_TOOLS_NUM_TEACHERS;
#else
constexpr TI NUM_TEACHERS = 1000;
#endif
constexpr TI NUM_ACTIVE_TEACHERS = NUM_TEACHERS;
enum class TEACHER_SELECTION_MODE {
    ALL,
    BEST,
    WORST,
    RANDOM,
};
#if !defined(RL_TOOLS_TEACHER_SELECTION_MODE_ALL) or !defined(RL_TOOLS_TEACHER_SELECTION_MODE_BEST) or !defined(RL_TOOLS_TEACHER_SELECTION_MODE_WORST) or !defined(RL_TOOLS_TEACHER_SELECTION_MODE_RANDOM)
constexpr TEACHER_SELECTION_MODE TEACHER_SELECTION = TEACHER_SELECTION_MODE::ALL;
#elif defined(RL_TOOLS_TEACHER_SELECTION_MODE_ALL)
#warning "Using RL_TOOLS_TEACHER_SELECTION_MODE_ALL for teacher selection mode"
constexpr TEACHER_SELECTION_MODE TEACHER_SELECTION = TEACHER_SELECTION_MODE::ALL;
#elif defined(RL_TOOLS_TEACHER_SELECTION_MODE_BEST)
#warning "Using RL_TOOLS_TEACHER_SELECTION_MODE_BEST for teacher selection mode"
constexpr TEACHER_SELECTION_MODE TEACHER_SELECTION = TEACHER_SELECTION_MODE::BEST;
#elif defined(RL_TOOLS_TEACHER_SELECTION_MODE_WORST)
#warning "Using RL_TOOLS_TEACHER_SELECTION_MODE_WORST for teacher selection mode"
constexpr TEACHER_SELECTION_MODE TEACHER_SELECTION = TEACHER_SELECTION_MODE::WORST;
#elif defined(RL_TOOLS_TEACHER_SELECTION_MODE_RANDOM)
#warning "Using RL_TOOLS_TEACHER_SELECTION_MODE_RANDOM for teacher selection mode"
constexpr TEACHER_SELECTION_MODE TEACHER_SELECTION = TEACHER_SELECTION_MODE::RANDOM;
#endif
constexpr TI EPOCH_TEACHER_FORCING = 10;
constexpr bool DYNAMIC_ALLOCATION = true;
constexpr bool SHUFFLE = true;
constexpr bool TEACHER_DETERMINISTIC = true;
constexpr bool ON_POLICY = true;
constexpr TI TEACHER_STUDENT_MIX = 0; // added teacher epochs in DAgger epochs
constexpr bool STEADY_STATE_POSITION_CORRECTION = true;
constexpr TI STEADY_STATE_POSITION_OFFSET_ESTIMATION_START = 250;


// typedefs
using ENVIRONMENT = typename builder::ENVIRONMENT_FACTORY_POST_TRAINING<DEVICE, T, TI, OPTIONS_POST_TRAINING>::ENVIRONMENT;
constexpr TI STEADY_STATE_POSITION_OFFSET_ESTIMATION_END = ENVIRONMENT::EPISODE_STEP_LIMIT;
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

using INPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Input>;
using INPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
using GRU_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, HIDDEN_DIM, rlt::nn::parameters::groups::Normal>;
using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;
using OUTPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, ENVIRONMENT::ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Output>;
using OUTPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<OUTPUT_LAYER_CONFIG>;
using MODULE_CHAIN = Module<INPUT_LAYER, Module<GRU, Module<OUTPUT_LAYER>>>;


using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, DYNAMIC_ALLOCATION>;
using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
using ACTOR = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
    static constexpr T ALPHA = 0.0001;
    static constexpr T WEIGHT_DECAY = 0;
    static constexpr T WEIGHT_DECAY_INPUT = 0;
    static constexpr T WEIGHT_DECAY_OUTPUT = 0;
};
using OPTIMIZER = rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<T, TI, ADAM_PARAMETERS>>;
using OUTPUT_SHAPE = ACTOR::OUTPUT_SHAPE;
using RESULT = rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES, ENVIRONMENT::EPISODE_STEP_LIMIT>>;
using RESULT_EVAL = rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES_EVAL, ENVIRONMENT::EPISODE_STEP_LIMIT>>;
using DATA = rlt::rl::utils::evaluation::Data<rlt::rl::utils::evaluation::DataSpecification<RESULT::SPEC>>;
using DATA_EVAL = rlt::rl::utils::evaluation::NoData<rlt::rl::utils::evaluation::DataSpecification<RESULT_EVAL::SPEC>>;
