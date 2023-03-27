#include <layer_in_c/operations/cpu_mux.h>
#include <layer_in_c/rl/components/running_normalizer/operations_generic.h>
namespace lic = layer_in_c;
#include <gtest/gtest.h>


using DEVICE = lic::DEVICE_FACTORY<lic::devices::DefaultCPUSpecification>;
using TI = typename DEVICE::index_t;

template <typename T, TI ROWS, TI COLS, TI BATCH_SIZE>
void test(){
    auto threshold = lic::utils::typing::is_same_v<T, float> ? 1e-5 : 1e-10;
    static_assert((ROWS % BATCH_SIZE) == 0);
    DEVICE device;
    lic::Matrix<lic::matrix::Specification<T, TI, ROWS, COLS>> data;
    lic::rl::components::RunningNormalizer<lic::rl::components::running_normalizer::Specification<T, TI, COLS>> running_normalizer;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    lic::malloc(device, data);
    lic::malloc(device, running_normalizer);
    lic::init(device, running_normalizer);
    lic::randn(device, data, rng);
    for(TI batch_start = 0; batch_start < ROWS; batch_start += BATCH_SIZE){
        auto batch = lic::view(device, data, lic::matrix::ViewSpec<BATCH_SIZE, COLS>{}, batch_start, 0);
        lic::update(device, running_normalizer, batch);
        if(batch_start == 0){
            for(TI col_i = 0; col_i < COLS; col_i++){
                auto col = lic::col(device, batch, col_i);
                auto mean = lic::mean(device, col);
                auto std = lic::std(device, col);
                auto mean_diff = mean - get(running_normalizer.mean, 0, col_i);
                auto std_diff = std - get(running_normalizer.std, 0, col_i);
                if(mean_diff >= threshold || std_diff >= threshold){
                    std::cout << "mean_diff: " << mean_diff << std::endl;
                    std::cout << "std_diff: " << std_diff << std::endl;
                }
                ASSERT_LT(mean_diff, threshold);
                ASSERT_LT(std_diff, threshold);
            }
        }
    }
    for(TI col_i = 0; col_i < COLS; col_i++){
        auto col = lic::col(device, data, col_i);
        auto mean = lic::mean(device, col);
        auto std = lic::std(device, col);
        auto mean_diff = mean - get(running_normalizer.mean, 0, col_i);
        auto std_diff = std - get(running_normalizer.std, 0, col_i);
        if(mean_diff >= threshold || std_diff >= threshold){
            std::cout << "mean_diff: " << mean_diff << std::endl;
            std::cout << "std_diff: " << std_diff << std::endl;
        }
        ASSERT_LT(mean_diff, threshold);
        ASSERT_LT(std_diff, threshold);
    }
}

TEST(LAYER_IN_C_RL_COMPONENTS_RUNNING_NORMALIZER, TEST){
    test<double, 100, 10, 10>();
    test<double, 32, 1, 1>();
    test<double, 32, 1, 2>();
    test<double, 32, 10, 1>();
    test<double, 32, 10, 2>();
    test<double, 32, 10, 32>();
    test<double, 32, 1, 32>();
    test<double, 100, 1, 50>();
    test<double, 100, 10, 50>();
    test<double, 10000, 10, 50>();
}
