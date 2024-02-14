// Group 1
#include <rl_tools/operations/cpu/group_1.h>
#include <rl_tools/operations/cuda/group_1.h>

// Group 2
#include <rl_tools/operations/cpu/group_2.h>
#include <rl_tools/operations/cuda/group_2.h>

// Group 3
#include <rl_tools/operations/cpu/group_3.h>
#include <rl_tools/operations/cuda/group_3.h>

#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn/loss_functions/mse/operations_cuda.h>
#include <rl_tools/nn_models/operations_generic.h>
#include <rl_tools/nn_models/operations_cpu.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <gtest/gtest.h>

template <typename T, typename TI, TI DIM_1, TI DIM_2, TI OFFSET_1, TI OFFSET_2, TI ALIGNMENT_1, TI ALIGNMENT_2, TI DIM_3, TI DIM_4, TI OFFSET_3, TI OFFSET_4, TI ALIGNMENT_3, TI ALIGNMENT_4>
void COPY_CONTAINER() {
    using DEVICE_CPU = rlt::devices::DefaultCPU;
    using DEVICE_CUDA = rlt::devices::DefaultCUDA;

    DEVICE_CUDA device_cuda;
    DEVICE_CPU device_cpu;

    {
        rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, DIM_1, DIM_2>> matrix_cpu;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CUDA::index_t, DIM_1, DIM_2>> matrix_cuda;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, DIM_1, DIM_2>> matrix_cpu2;
        rlt::malloc(device_cpu, matrix_cpu);
        rlt::malloc(device_cuda, matrix_cuda);
        rlt::malloc(device_cpu, matrix_cpu2);

        rlt::set_all(device_cpu, matrix_cpu, 1337.0f);

        rlt::copy(device_cpu, device_cuda, matrix_cpu, matrix_cuda);
        rlt::copy(device_cuda, device_cpu, matrix_cuda, matrix_cpu2);
        auto diff = rlt::abs_diff(device_cpu, matrix_cpu, matrix_cpu2);
        ASSERT_FLOAT_EQ(diff, 0.0f);
        rlt::free(device_cpu, matrix_cpu);
        rlt::free(device_cuda, matrix_cuda);
        rlt::free(device_cpu, matrix_cpu2);
    }
    {
        rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, DIM_1, DIM_2, rlt::matrix::layouts::RowMajorAlignment<DEVICE_CPU::index_t, ALIGNMENT_1>>> matrix_cpu;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CUDA::index_t, DIM_1, DIM_2>> matrix_cuda;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, DIM_1, DIM_2>> matrix_cpu2;
        rlt::malloc(device_cpu, matrix_cpu);
        rlt::malloc(device_cuda, matrix_cuda);
        rlt::malloc(device_cpu, matrix_cpu2);

        rlt::set_all(device_cpu, matrix_cpu, 1337.0f);

        rlt::copy(device_cpu, device_cuda, matrix_cpu, matrix_cuda);
        rlt::copy(device_cuda, device_cpu, matrix_cuda, matrix_cpu2);
        auto diff = rlt::abs_diff(device_cpu, matrix_cpu, matrix_cpu2);
        ASSERT_FLOAT_EQ(diff, 0.0f);
        rlt::free(device_cpu, matrix_cpu);
        rlt::free(device_cuda, matrix_cuda);
        rlt::free(device_cpu, matrix_cpu2);
    }

    {
        rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, DIM_1, DIM_2>> matrix_cpu;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CUDA::index_t, DIM_1, DIM_2, rlt::matrix::layouts::RowMajorAlignment<DEVICE_CPU::index_t, ALIGNMENT_2>>> matrix_cuda;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, DIM_1, DIM_2>> matrix_cpu2;
        rlt::malloc(device_cpu, matrix_cpu);
        rlt::malloc(device_cuda, matrix_cuda);
        rlt::malloc(device_cpu, matrix_cpu2);

        rlt::set_all(device_cpu, matrix_cpu, 1337.0f);

        rlt::copy(device_cpu, device_cuda, matrix_cpu, matrix_cuda);
        static_assert(DIM_1 > OFFSET_1);
        static_assert(DIM_2 > OFFSET_2);
        increment(matrix_cpu, OFFSET_1, OFFSET_2, 17);
        rlt::copy(device_cuda, device_cpu, matrix_cuda, matrix_cpu2);
        auto diff = rlt::abs_diff(device_cpu, matrix_cpu, matrix_cpu2);
        ASSERT_FLOAT_EQ(diff, 17.0f);
        rlt::free(device_cpu, matrix_cpu);
        rlt::free(device_cuda, matrix_cuda);
        rlt::free(device_cpu, matrix_cpu2);
    }
    {
        static_assert(DIM_3 >= DIM_1);
        static_assert(DIM_4 >= DIM_2);
        rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, DIM_1, DIM_2>> matrix_cpu;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CUDA::index_t, DIM_3, DIM_4, rlt::matrix::layouts::RowMajorAlignment<DEVICE_CPU::index_t, ALIGNMENT_3>>> matrix_cuda_data;
        static_assert(OFFSET_3 < DIM_3);
        static_assert(OFFSET_4 < DIM_4);
        auto matrix_cuda = rlt::view<DEVICE_CUDA, typename decltype(matrix_cuda_data)::SPEC, DIM_1, DIM_2>(device_cuda, matrix_cuda_data, OFFSET_3, OFFSET_4);
        rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CUDA::index_t, DIM_1, DIM_2, rlt::matrix::layouts::RowMajorAlignment<DEVICE_CPU::index_t, ALIGNMENT_4>>> matrix_cuda2;

        rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, DIM_1, DIM_2>> matrix_cpu2;
        rlt::malloc(device_cpu, matrix_cpu);
        rlt::malloc(device_cuda, matrix_cuda);
        rlt::malloc(device_cuda, matrix_cuda2);
        rlt::malloc(device_cpu, matrix_cpu2);

        auto rng = rlt::random::default_engine(decltype(device_cpu)::SPEC::RANDOM());

        for(DEVICE_CPU::index_t row_i = 0; row_i < decltype(matrix_cpu)::SPEC::ROWS; row_i++){
            for(DEVICE_CPU::index_t col_i = 0; col_i < decltype(matrix_cpu)::SPEC::COLS; col_i++){
                set(matrix_cpu, row_i, col_i, rlt::random::normal_distribution::sample(decltype(device_cpu)::SPEC::RANDOM(), (T)0, (T)1, rng));
            }
        }

        rlt::copy(device_cpu, device_cuda, matrix_cpu, matrix_cuda);
        increment(matrix_cpu, OFFSET_1, OFFSET_2, 17);
        rlt::copy(device_cuda, device_cuda, matrix_cuda, matrix_cuda2);
        rlt::copy(device_cuda, device_cpu, matrix_cuda2, matrix_cpu2);
        auto diff = rlt::abs_diff(device_cpu, matrix_cpu, matrix_cpu2);
        ASSERT_FLOAT_EQ(diff, 17.0f);
        rlt::free(device_cpu, matrix_cpu);
        rlt::free(device_cuda, matrix_cuda);
        rlt::free(device_cuda, matrix_cuda2);
        rlt::free(device_cpu, matrix_cpu2);
    }
}

TEST(RL_TOOLS_NN_CUDA, COPY_CONTAINER){
/*
template <typename T, typename TI, TI DIM_1, TI DIM_2, TI OFFSET_1, TI OFFSET_2, TI ALIGNMENT_1, TI ALIGNMENT_2, TI DIM_3, TI DIM_4, TI OFFSET_3, TI OFFSET_4, TI ALIGNMENT_3, TI ALIGNMENT_4>
    julia code to generate fuzzing calls
    s(dtype, dim_1, dim_2, alignment_1, alignment_2, dim_3, dim_4, alignment_3, alignment_4) = "COPY_CONTAINER<$dtype, unsigned int, $dim_1, $dim_2, $(rand(0:(dim_1-1))), $(rand(0:(dim_2-1))), $alignment_1, $alignment_2, $dim_3, $dim_4, $(rand(0:(dim_3-1))), $(rand(0:(dim_4-1))), $alignment_3, $alignment_4>();\n"
    t(dtype, dim_1, dim_2, alignment_1, alignment_2, alignment_3, alignment_4) = s(dtype, dim_1, dim_2, alignment_1, alignment_2, dim_1 + rand(0:1000), dim_2 + rand(0:1000), alignment_3, alignment_4)
    print(reduce((a,c)->a * t((rand(0:1) == 0 ? "float" : "double"), rand(1:1000), rand(1:1000), rand(1:1000), rand(1:1000), rand(1:1000), rand(1:1000)), 1:50, init=""))
*/
//    COPY_CONTAINER<float, unsigned int, 10, 10, 9, 1, 13, 13, 30, 63, 5, 7, 13, 563>();
//    COPY_CONTAINER<double, unsigned int, 77, 809, 26, 582, 598, 856, 87, 904, 61, 72, 96, 908>();
    COPY_CONTAINER<float, unsigned int, 368, 885, 60, 766, 968, 990, 472, 1676, 47, 1111, 160, 359>();
    COPY_CONTAINER<double, unsigned int, 87, 986, 22, 592, 209, 41, 771, 1635, 409, 1304, 937, 692>();
    COPY_CONTAINER<float, unsigned int, 764, 121, 28, 108, 156, 614, 1175, 496, 1048, 196, 596, 537>();
    COPY_CONTAINER<float, unsigned int, 920, 444, 479, 355, 552, 723, 1189, 698, 336, 339, 267, 172>();
    COPY_CONTAINER<double, unsigned int, 982, 400, 515, 93, 641, 808, 1844, 782, 1457, 87, 821, 883>();
    COPY_CONTAINER<double, unsigned int, 912, 613, 250, 271, 287, 235, 927, 697, 603, 207, 233, 793>();
    COPY_CONTAINER<double, unsigned int, 693, 342, 99, 100, 399, 603, 1338, 846, 591, 405, 649, 885>();
    COPY_CONTAINER<float, unsigned int, 852, 894, 635, 673, 171, 72, 1202, 1513, 843, 241, 135, 959>();
    COPY_CONTAINER<double, unsigned int, 948, 611, 172, 570, 652, 83, 1176, 1111, 260, 418, 536, 572>();
    COPY_CONTAINER<float, unsigned int, 368, 885, 60, 766, 968, 990, 472, 1676, 47, 1111, 160, 359>();
    COPY_CONTAINER<double, unsigned int, 87, 986, 22, 592, 209, 41, 771, 1635, 409, 1304, 937, 692>();
    COPY_CONTAINER<float, unsigned int, 764, 121, 28, 108, 156, 614, 1175, 496, 1048, 196, 596, 537>();
    COPY_CONTAINER<float, unsigned int, 920, 444, 479, 355, 552, 723, 1189, 698, 336, 339, 267, 172>();
    COPY_CONTAINER<double, unsigned int, 982, 400, 515, 93, 641, 808, 1844, 782, 1457, 87, 821, 883>();
    COPY_CONTAINER<double, unsigned int, 912, 613, 250, 271, 287, 235, 927, 697, 603, 207, 233, 793>();
    COPY_CONTAINER<double, unsigned int, 693, 342, 99, 100, 399, 603, 1338, 846, 591, 405, 649, 885>();
    COPY_CONTAINER<float, unsigned int, 852, 894, 635, 673, 171, 72, 1202, 1513, 843, 241, 135, 959>();
    COPY_CONTAINER<double, unsigned int, 948, 611, 172, 570, 652, 83, 1176, 1111, 260, 418, 536, 572>();
    COPY_CONTAINER<double, unsigned int, 660, 152, 317, 87, 621, 458, 823, 457, 712, 51, 516, 568>();
    COPY_CONTAINER<double, unsigned int, 660, 466, 42, 13, 789, 704, 1495, 1466, 754, 899, 589, 426>();
    COPY_CONTAINER<float, unsigned int, 181, 83, 81, 26, 276, 84, 638, 175, 302, 136, 339, 553>();
    COPY_CONTAINER<float, unsigned int, 664, 993, 84, 607, 670, 613, 1092, 1084, 791, 740, 136, 30>();
    COPY_CONTAINER<float, unsigned int, 84, 929, 56, 489, 240, 175, 181, 1482, 152, 1066, 57, 428>();
    COPY_CONTAINER<double, unsigned int, 854, 935, 431, 588, 994, 915, 1838, 1487, 1272, 874, 588, 487>();
    COPY_CONTAINER<double, unsigned int, 133, 299, 89, 170, 64, 226, 625, 609, 370, 402, 1, 170>();
    COPY_CONTAINER<double, unsigned int, 743, 106, 438, 66, 282, 763, 1008, 963, 594, 765, 487, 100>();
    COPY_CONTAINER<double, unsigned int, 754, 58, 226, 57, 803, 467, 1719, 324, 837, 202, 287, 904>();
    COPY_CONTAINER<float, unsigned int, 13, 192, 3, 85, 397, 515, 747, 883, 720, 822, 624, 88>();
    COPY_CONTAINER<double, unsigned int, 931, 293, 115, 130, 754, 857, 1883, 1246, 753, 721, 965, 55>();
    COPY_CONTAINER<double, unsigned int, 318, 428, 256, 419, 742, 406, 1081, 609, 436, 1, 871, 759>();
    COPY_CONTAINER<float, unsigned int, 911, 462, 849, 224, 793, 562, 1418, 631, 1414, 54, 948, 156>();
    COPY_CONTAINER<double, unsigned int, 952, 499, 6, 305, 908, 288, 1046, 1142, 460, 186, 610, 469>();
    COPY_CONTAINER<double, unsigned int, 578, 780, 225, 724, 931, 256, 1514, 791, 327, 617, 438, 616>();
    COPY_CONTAINER<double, unsigned int, 765, 902, 139, 751, 763, 494, 1180, 1111, 901, 406, 641, 208>();
    COPY_CONTAINER<float, unsigned int, 709, 613, 385, 585, 36, 811, 1134, 805, 520, 774, 124, 555>();
    COPY_CONTAINER<float, unsigned int, 892, 280, 466, 176, 757, 194, 1181, 661, 874, 547, 483, 73>();
    COPY_CONTAINER<double, unsigned int, 680, 182, 231, 178, 191, 278, 884, 1103, 123, 253, 680, 126>();
    COPY_CONTAINER<double, unsigned int, 77, 419, 37, 347, 205, 913, 798, 465, 399, 404, 603, 911>();
    COPY_CONTAINER<float, unsigned int, 170, 75, 72, 50, 313, 441, 1096, 105, 396, 30, 163, 27>();
    COPY_CONTAINER<float, unsigned int, 199, 562, 106, 315, 508, 821, 472, 1113, 253, 1095, 216, 261>();
    COPY_CONTAINER<double, unsigned int, 869, 778, 461, 724, 766, 752, 1081, 1021, 415, 53, 268, 248>();
    COPY_CONTAINER<float, unsigned int, 942, 776, 571, 462, 234, 89, 1783, 1082, 1639, 864, 400, 888>();
    COPY_CONTAINER<double, unsigned int, 461, 525, 430, 79, 372, 88, 940, 694, 765, 552, 625, 495>();
    COPY_CONTAINER<double, unsigned int, 640, 730, 186, 646, 234, 609, 1364, 1648, 452, 1478, 840, 732>();
    COPY_CONTAINER<float, unsigned int, 929, 15, 223, 13, 331, 497, 1374, 404, 914, 267, 938, 900>();
    COPY_CONTAINER<double, unsigned int, 948, 126, 309, 8, 896, 461, 1937, 1075, 1529, 1062, 930, 852>();
    COPY_CONTAINER<double, unsigned int, 926, 737, 38, 2, 910, 581, 1641, 1064, 1472, 812, 13, 922>();
    COPY_CONTAINER<double, unsigned int, 952, 187, 744, 65, 228, 27, 1461, 287, 324, 65, 961, 512>();
    COPY_CONTAINER<double, unsigned int, 805, 823, 503, 230, 825, 442, 1300, 1515, 890, 28, 52, 979>();
    COPY_CONTAINER<float, unsigned int, 72, 233, 55, 34, 348, 544, 516, 936, 333, 591, 710, 346>();
    COPY_CONTAINER<float, unsigned int, 736, 126, 482, 62, 353, 605, 1187, 375, 337, 332, 841, 448>();
    COPY_CONTAINER<float, unsigned int, 700, 984, 337, 639, 886, 959, 1024, 1535, 49, 448, 832, 82>();
    COPY_CONTAINER<double, unsigned int, 464, 46, 60, 30, 323, 576, 1302, 697, 1073, 102, 579, 495>();
    COPY_CONTAINER<float, unsigned int, 274, 390, 146, 77, 161, 198, 1129, 863, 100, 470, 376, 369>();
    COPY_CONTAINER<float, unsigned int, 106, 690, 5, 334, 960, 82, 1053, 1146, 170, 966, 728, 935>();
    COPY_CONTAINER<double, unsigned int, 935, 474, 662, 35, 873, 798, 1559, 1232, 897, 999, 357, 563>();
    COPY_CONTAINER<double, unsigned int, 669, 73, 45, 54, 959, 970, 809, 853, 210, 472, 846, 756>();
    COPY_CONTAINER<double, unsigned int, 271, 343, 239, 160, 327, 82, 486, 1054, 41, 795, 34, 110>();
    COPY_CONTAINER<double, unsigned int, 711, 20, 295, 10, 609, 133, 803, 705, 300, 262, 777, 276>();

}


TEST(RL_TOOLS_NN_CUDA, COPYING_VIEWS){
    using DEVICE_CPU = rlt::devices::DefaultCPU;
    using DEVICE_CUDA = rlt::devices::DefaultCUDA;

    DEVICE_CUDA device_cuda;
    DEVICE_CPU device_cpu;
    using DTYPE = float;
    {
        auto rng = rlt::random::default_engine(decltype(device_cpu)::SPEC::RANDOM());
        rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE_CPU::index_t, 100, 100>> matrix_cpu_data;
        rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE_CPU::index_t, 100, 100>> matrix_cpu_data_2;
        rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE_CPU::index_t, 100, 100>> matrix_cpu_data_3;
        rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE_CPU::index_t, 100, 100>> matrix_cpu_data_3_orig;
        rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE_CPU::index_t, 100, 100>> matrix_cpu_data_4;
        rlt::MatrixDynamic<rlt::matrix::Specification<DTYPE, DEVICE_CPU::index_t, 100, 100>> matrix_cuda_data;
        rlt::malloc(device_cpu, matrix_cpu_data);
        rlt::malloc(device_cpu, matrix_cpu_data_2);
        rlt::malloc(device_cpu, matrix_cpu_data_3);
        rlt::malloc(device_cpu, matrix_cpu_data_3_orig);
        rlt::malloc(device_cpu, matrix_cpu_data_4);
        rlt::malloc(device_cuda, matrix_cuda_data);

        auto matrix_cpu_view = rlt::view<DEVICE_CUDA, typename decltype(matrix_cpu_data)::SPEC, 50, 50>(device_cuda, matrix_cpu_data, 25, 25);
        auto matrix_cpu_view_2 = rlt::view<DEVICE_CUDA, typename decltype(matrix_cpu_data_2)::SPEC, 50, 50>(device_cuda, matrix_cpu_data_2, 25, 25);
        auto matrix_cpu_view_3 = rlt::view<DEVICE_CUDA, typename decltype(matrix_cpu_data_3)::SPEC, 50, 50>(device_cuda, matrix_cpu_data_3, 25, 25);
        auto matrix_cuda_view = rlt::view<DEVICE_CUDA, typename decltype(matrix_cuda_data)::SPEC, 50, 50>(device_cuda, matrix_cuda_data, 25, 25);

        auto matrix_cpu_view_alt        = rlt::view<DEVICE_CUDA, typename decltype(matrix_cpu_data       )::SPEC, 40, 5>(device_cuda, matrix_cpu_data       ,  5, 5);
        auto matrix_cpu_view_3_alt      = rlt::view<DEVICE_CUDA, typename decltype(matrix_cpu_data_3     )::SPEC, 40, 5>(device_cuda, matrix_cpu_data_3     ,  5, 5);
        auto matrix_cpu_view_3_alt_orig = rlt::view<DEVICE_CUDA, typename decltype(matrix_cpu_data_3_orig)::SPEC, 40, 5>(device_cuda, matrix_cpu_data_3_orig,  5, 5);


        {
            rlt::randn(device_cpu, matrix_cpu_data, rng);
            rlt::randn(device_cpu, matrix_cpu_data_2, rng);
            rlt::randn(device_cpu, matrix_cpu_data_3, rng);
            rlt::copy(device_cpu, device_cpu, matrix_cpu_data_3, matrix_cpu_data_3_orig);
            rlt::copy(device_cpu, device_cuda, matrix_cpu_data_2, matrix_cuda_data);
            rlt::copy(device_cpu, device_cuda, matrix_cpu_view, matrix_cuda_view);
            rlt::copy(device_cuda, device_cpu, matrix_cuda_data, matrix_cpu_data_3);
            DTYPE abs_diff = rlt::abs_diff(device_cpu, matrix_cpu_view, matrix_cpu_view_3);
            EXPECT_LT(abs_diff, 1e-5);
        }
        {
            rlt::randn(device_cpu, matrix_cpu_data, rng);
            rlt::randn(device_cpu, matrix_cpu_data_2, rng);
            rlt::randn(device_cpu, matrix_cpu_data_3, rng);
            rlt::copy(device_cpu, device_cpu, matrix_cpu_data_3, matrix_cpu_data_3_orig);
            rlt::copy(device_cpu, device_cpu, matrix_cpu_view, matrix_cpu_view_3);

            DTYPE abs_diff_3_orig = rlt::abs_diff(device_cpu, matrix_cpu_view_3_alt, matrix_cpu_view_3_alt_orig);
            EXPECT_LT(abs_diff_3_orig, 1e-5);

            rlt::copy(device_cpu, device_cuda, matrix_cpu_data, matrix_cuda_data);
            rlt::copy(device_cpu, device_cuda, matrix_cpu_view_2, matrix_cuda_view);

            rlt::copy(device_cuda, device_cpu, matrix_cuda_data, matrix_cpu_data_4);
            DTYPE abs_diff = rlt::abs_diff(device_cpu, matrix_cpu_data, matrix_cpu_data_4);
            EXPECT_GT(abs_diff, 1e-5);

            rlt::copy(device_cpu, device_cuda, matrix_cpu_view_3, matrix_cuda_view);
            rlt::copy(device_cuda, device_cpu, matrix_cuda_data, matrix_cpu_data_4);
            abs_diff = rlt::abs_diff(device_cpu, matrix_cpu_data, matrix_cpu_data_4);
            EXPECT_LT(abs_diff, 1e-5);
        }



        rlt::free(device_cpu, matrix_cpu_data);
        rlt::free(device_cpu, matrix_cpu_data_2);
        rlt::free(device_cpu, matrix_cpu_data_3);
        rlt::free(device_cpu, matrix_cpu_data_4);
        rlt::free(device_cuda, matrix_cuda_data);
    }
}
namespace copy{
    using DTYPE = float;
    using DEVICE_CPU = rlt::devices::DefaultCPU;
    using DEVICE_CUDA = rlt::devices::DefaultCUDA;

    constexpr DEVICE_CPU::index_t BATCH_SIZE = 100;
    constexpr DEVICE_CPU::index_t HIDDEN_DIM = BATCH_SIZE;

    template <typename T, typename TI, rlt::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
    using StructureSpecification = rlt::nn_models::mlp::StructureSpecification<T, TI, HIDDEN_DIM, HIDDEN_DIM, 3, HIDDEN_DIM, ACTIVATION_FUNCTION, ACTIVATION_FUNCTION, BATCH_SIZE>;

    using OPTIMIZER_PARAMETERS = rlt::nn::optimizers::adam::DefaultParametersTorch<DTYPE, typename DEVICE_CPU::index_t>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    template <typename T, typename TI, rlt::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
    using NNSPEC = rlt::nn_models::mlp::AdamSpecification<StructureSpecification<T, TI, ACTIVATION_FUNCTION>>;

    constexpr DEVICE_CPU::index_t ITERATIONS = 1;
    constexpr DEVICE_CPU::index_t NAIVE_ITERATIONS = 1;
}


TEST(RL_TOOLS_NN_CUDA, COPY) {
    using NetworkTypeCPU = rlt::nn_models::mlp::NeuralNetworkAdam<copy::NNSPEC<copy::DTYPE, copy::DEVICE_CPU::index_t, rlt::nn::activation_functions::RELU>>;
    using NetworkTypeCUDA = rlt::nn_models::mlp::NeuralNetworkAdam<copy::NNSPEC<copy::DTYPE, copy::DEVICE_CUDA::index_t, rlt::nn::activation_functions::RELU>>;
    copy::OPTIMIZER optimizer;
    copy::DEVICE_CPU device_cpu;
    copy::DEVICE_CUDA device_cuda;
    NetworkTypeCPU network_cpu;
    NetworkTypeCPU network_cpu_2;
    NetworkTypeCUDA network_cuda;
    rlt::malloc(device_cpu, network_cpu);
    rlt::malloc(device_cpu, network_cpu_2);
    rlt::malloc(device_cuda, network_cuda);

    auto rng = rlt::random::default_engine(copy::DEVICE_CPU::SPEC::RANDOM());

    rlt::init_weights(device_cpu, network_cpu, rng);
    rlt::init_weights(device_cpu, network_cpu_2, rng);
    rlt::zero_gradient(device_cpu, network_cpu);
    rlt::zero_gradient(device_cpu, network_cpu_2);
    rlt::reset_optimizer_state(device_cpu, optimizer, network_cpu);
    rlt::reset_optimizer_state(device_cpu, optimizer, network_cpu_2);
    rlt::reset_forward_state(device_cpu, network_cpu);
    rlt::reset_forward_state(device_cpu, network_cpu_2);
    auto cpu_network_diff = rlt::abs_diff(device_cpu, network_cpu, network_cpu_2);
    std::cout << "CPU network diff: " << cpu_network_diff << std::endl;
    ASSERT_GT(cpu_network_diff, 0);

    rlt::copy(device_cpu, device_cuda, network_cpu, network_cuda);
    rlt::copy(device_cuda, device_cpu, network_cuda, network_cpu_2);
    auto cpu_network_diff_round_trip = rlt::abs_diff(device_cpu, network_cpu, network_cpu_2);
    std::cout << "CPU network round-trip: " << cpu_network_diff_round_trip << std::endl;
    ASSERT_FLOAT_EQ(cpu_network_diff_round_trip, 0);

    increment(network_cpu.hidden_layers[0].weights.parameters, 0, 50, 5);

    cpu_network_diff = rlt::abs_diff(device_cpu, network_cpu, network_cpu_2);
    std::cout << "CPU network diff: " << cpu_network_diff << std::endl;
    ASSERT_FLOAT_EQ(cpu_network_diff, 5);

    rlt::copy(device_cpu, device_cuda, network_cpu, network_cuda);
    rlt::copy(device_cuda, device_cpu, network_cuda, network_cpu_2);
    cpu_network_diff_round_trip = rlt::abs_diff(device_cpu, network_cpu, network_cpu_2);
    ASSERT_FLOAT_EQ(cpu_network_diff_round_trip, 0);
    std::cout << "CPU network round-trip: " << cpu_network_diff_round_trip << std::endl;

    rlt::free(device_cpu, network_cpu);
    rlt::free(device_cpu, network_cpu_2);
    rlt::free(device_cuda, network_cuda);
}

template <typename T, typename TI, TI BATCH_SIZE, TI ITERATIONS>
void GEMM() {
    using DEVICE_CPU = rlt::devices::DefaultCPU;
    using DEVICE_CUDA = rlt::devices::DefaultCUDA;

    constexpr DEVICE_CPU::index_t HIDDEN_DIM = BATCH_SIZE;

    constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::IDENTITY;
    using StructureSpecification = rlt::nn_models::mlp::StructureSpecification<T, TI, HIDDEN_DIM, HIDDEN_DIM, 3, HIDDEN_DIM, ACTIVATION_FUNCTION, rlt::nn::activation_functions::RELU, BATCH_SIZE>;

    using OPTIMIZER_PARAMETERS = rlt::nn::optimizers::adam::DefaultParametersTorch<T, typename DEVICE_CUDA::index_t>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<copy::OPTIMIZER_PARAMETERS>;
    using NNSpecification = rlt::nn_models::mlp::AdamSpecification<StructureSpecification>;

    std::cout << "GEMM<" << (rlt::utils::typing::is_same_v<T, float> ? "float" : "double") << ", " << BATCH_SIZE << ">" << std::endl;
    using NetworkTypeCPU = rlt::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    using NetworkTypeCUDA = rlt::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    NetworkTypeCPU network_cpu;
    typename NetworkTypeCPU::template Buffer<BATCH_SIZE> network_cpu_buffers;
    NetworkTypeCUDA network_cuda;
    typename NetworkTypeCUDA::template Buffer<BATCH_SIZE> network_cuda_buffers;
    OPTIMIZER optimizer;
    rlt::malloc(device_cpu, network_cpu);
    rlt::malloc(device_cpu, network_cpu_buffers);
    rlt::malloc(device_cuda, network_cuda);
    rlt::malloc(device_cuda, network_cuda_buffers);

    auto rng = rlt::random::default_engine(DEVICE_CPU::SPEC::RANDOM());

    rlt::init_weights(device_cpu, network_cpu, rng);
    rlt::reset_optimizer_state(device_cpu, optimizer, network_cpu);
    rlt::copy(device_cpu, device_cuda, network_cpu, network_cuda);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cpu;
    rlt::malloc(device_cpu, input_cpu);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM>> output_first_layer_cpu;
    rlt::malloc(device_cpu, output_first_layer_cpu);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM>> output_first_layer_cuda_cpu;
    rlt::malloc(device_cpu, output_first_layer_cuda_cpu);

    for(typename NetworkTypeCPU::TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
        for(typename NetworkTypeCPU::TI input_i = 0; input_i < NetworkTypeCPU::INPUT_DIM; input_i++){
            set(input_cpu, batch_i, input_i, rlt::random::normal_distribution::sample(DEVICE_CPU::SPEC::RANDOM(), (T)0, (T)1, rng));
        }
    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Input:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::INPUT_DIM; ++j)
//            {
//                std::cout << input_cpu.data[i * NetworkTypeCPU::INPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Weights:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::INPUT_DIM; ++j)
//            {
//                std::cout << network_cpu.input_layer.weights.data[i * NetworkTypeCPU::INPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Biases:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM; ++i)
//        {
//            std::cout << network_cpu.input_layer.biases.data[i] << " ";
//        }
//        std::cout << std::endl;
//    }


    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cuda;
    rlt::malloc(device_cuda, input_cuda);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM>> output_first_layer_cuda;
    rlt::malloc(device_cuda, output_first_layer_cuda);

    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    rlt::evaluate(device_cpu, network_cpu.input_layer, input_cpu, output_first_layer_cpu);
    rlt::evaluate(device_cuda, network_cuda.input_layer, input_cuda, output_first_layer_cuda);
    cudaDeviceSynchronize();

    rlt::copy(device_cuda, device_cpu, output_first_layer_cuda, output_first_layer_cuda_cpu);
    auto evaluation_diff = rlt::abs_diff(device_cpu, output_first_layer_cuda_cpu, output_first_layer_cpu)/(BATCH_SIZE * NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM);

//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM <= 10){
//        std::cout << "cpu output:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM; ++j)
//            {
//                std::cout << output_first_layer_cpu.data[i * NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM <= 10){
//        std::cout << "cuda output:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i){
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM; ++j){
//                std::cout << output_first_layer_cuda_cpu.data[i * NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM <= 10){
//        std::cout << "cuda diff:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM; ++j)
//            {
//                T diff = output_first_layer_cpu.data[i * NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM + j] - output_first_layer_cuda_cpu.data[i * NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM + j];
//                diff = std::abs(diff) > 1e-7 ? diff : 0;
//                std::cout << diff << " ";
//            }
//            std::cout << std::endl;
//        }
//    }

    std::cout << "Evaluation diff: " << evaluation_diff << std::endl;
    auto threshold = (rlt::utils::typing::is_same_v<T, float> ? 1e-6 : 1e-15);
    if(std::isnan(evaluation_diff) || evaluation_diff > threshold){
        ASSERT_LT(evaluation_diff, threshold);
    }

    {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < ITERATIONS; ++i)
        {
            rlt::evaluate(device_cuda, network_cuda.input_layer, input_cuda, output_first_layer_cuda);
            cudaDeviceSynchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "CUDA evaluation time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((T)ITERATIONS) << "us" << std::endl;
    }
}
TEST(RL_TOOLS_NN_CUDA, GEMM) {
    using DEFAULT_DTYPE = float;
    GEMM<DEFAULT_DTYPE, unsigned int, 1, 1>();
    GEMM<DEFAULT_DTYPE, unsigned int, 2, 1>();
    GEMM<DEFAULT_DTYPE, unsigned int, 32, 1>();
    GEMM<DEFAULT_DTYPE, unsigned int, 1024, 1>();
    GEMM<DEFAULT_DTYPE, unsigned int, 10, 1>();
    GEMM<DEFAULT_DTYPE, unsigned int, 9, 1>();
    GEMM<double, unsigned int, 200, 1>();
    GEMM<DEFAULT_DTYPE, unsigned int, 200, 1>();
    GEMM<DEFAULT_DTYPE, unsigned int, 64, 1000>();
    GEMM<DEFAULT_DTYPE, unsigned int, 256, 1000>();
}

template <typename T, typename TI, TI BATCH_SIZE, TI ITERATIONS>
void FORWARD() {
    using DEVICE_CPU = rlt::devices::DefaultCPU;
    using DEVICE_CUDA = rlt::devices::DefaultCUDA;

    constexpr DEVICE_CPU::index_t HIDDEN_DIM = BATCH_SIZE;

    constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::IDENTITY;
    using StructureSpecification = rlt::nn_models::mlp::StructureSpecification<T, TI, HIDDEN_DIM, HIDDEN_DIM, 3, HIDDEN_DIM, ACTIVATION_FUNCTION, rlt::nn::activation_functions::RELU, BATCH_SIZE>;

    using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, typename DEVICE_CUDA::index_t>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
    using NNSpecification = rlt::nn_models::mlp::AdamSpecification<StructureSpecification>;

    std::cout << "FORWARD<" << (rlt::utils::typing::is_same_v<T, float> ? "float" : "double") << ", " << BATCH_SIZE << ">" << std::endl;
    using NetworkTypeCPU = rlt::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    using NetworkTypeCUDA = rlt::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    NetworkTypeCPU network_cpu;
    typename NetworkTypeCPU::template Buffer<BATCH_SIZE> network_cpu_buffers;
    NetworkTypeCUDA network_cuda;
    typename NetworkTypeCPU::template Buffer<BATCH_SIZE> network_cuda_buffers;
    rlt::malloc(device_cpu, network_cpu);
    rlt::malloc(device_cpu, network_cpu_buffers);
    rlt::malloc(device_cuda, network_cuda);
    rlt::malloc(device_cpu, network_cuda_buffers);

    auto rng = rlt::random::default_engine(DEVICE_CPU::SPEC::RANDOM());

    rlt::init_weights(device_cpu, network_cpu, rng);
    rlt::copy(device_cpu, device_cuda, network_cpu, network_cuda);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cpu;
    rlt::malloc(device_cpu, input_cpu);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cpu;
    rlt::malloc(device_cpu, output_cpu);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cuda_cpu;
    rlt::malloc(device_cpu, output_cuda_cpu);

    for(typename NetworkTypeCPU::TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
        for(typename NetworkTypeCPU::TI input_i = 0; input_i < NetworkTypeCPU::INPUT_DIM; input_i++){
            set(input_cpu, batch_i, input_i, rlt::random::normal_distribution::sample(DEVICE_CPU::SPEC::RANDOM(), (T)0, (T)1, rng));
        }
    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Input:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::INPUT_DIM; ++j)
//            {
//                std::cout << input_cpu.data[i * NetworkTypeCPU::INPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Weights:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < NetworkTypeCPU::OUTPUT_DIM; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::INPUT_DIM; ++j)
//            {
//                std::cout << network_cpu.input_layer.weights.data[i * NetworkTypeCPU::INPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::INPUT_DIM <= 10){
//        std::cout << "Biases:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < NetworkTypeCPU::OUTPUT_DIM; ++i)
//        {
//            std::cout << network_cpu.input_layer.biases.data[i] << " ";
//        }
//        std::cout << std::endl;
//    }


    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cuda;
    rlt::malloc(device_cuda, input_cuda);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cuda;
    rlt::malloc(device_cuda, output_cuda);

    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    rlt::forward(device_cpu, network_cpu, input_cpu);
    rlt::forward(device_cuda, network_cuda, input_cuda);
    cudaDeviceSynchronize();

    rlt::copy(device_cuda, device_cpu, network_cuda.output_layer.output, output_cuda_cpu);
    auto evaluation_diff = rlt::abs_diff(device_cpu, output_cuda_cpu, network_cpu.output_layer.output)/(BATCH_SIZE * NetworkTypeCPU::OUTPUT_DIM);

//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::OUTPUT_DIM <= 10){
//        std::cout << "cpu output:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::OUTPUT_DIM; ++j)
//            {
//                std::cout << output_cpu.data[i * NetworkTypeCPU::OUTPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::OUTPUT_DIM <= 10){
//        std::cout << "cuda output:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i){
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::OUTPUT_DIM; ++j){
//                std::cout << output_cuda_cpu.data[i * NetworkTypeCPU::OUTPUT_DIM + j] << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//
//    if(BATCH_SIZE <= 10 && NetworkTypeCPU::OUTPUT_DIM <= 10){
//        std::cout << "cuda diff:" << std::endl;
//        for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE; ++i)
//        {
//            for(typename NetworkTypeCPU::TI j = 0; j < NetworkTypeCPU::OUTPUT_DIM; ++j)
//            {
//                T diff = output_cpu.data[i * NetworkTypeCPU::OUTPUT_DIM + j] - output_cuda_cpu.data[i * NetworkTypeCPU::OUTPUT_DIM + j];
//                diff = std::abs(diff) > 1e-7 ? diff : 0;
//                std::cout << diff << " ";
//            }
//            std::cout << std::endl;
//        }
//    }

    std::cout << "Evaluation diff: " << evaluation_diff << std::endl;
    auto threshold = (rlt::utils::typing::is_same_v<T, float> ? 1e-7 : 1e-15);
    if(std::isnan(evaluation_diff) || evaluation_diff > threshold){
        ASSERT_LT(evaluation_diff, threshold);
    }

    {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < ITERATIONS; ++i)
        {
            rlt::evaluate(device_cuda, network_cuda.input_layer, input_cuda, output_cuda);
            cudaDeviceSynchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "CUDA evaluation time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((T)ITERATIONS) << "us" << std::endl;
    }
}

TEST(RL_TOOLS_NN_CUDA, FORWARD) {
    FORWARD<float, unsigned int, 1, 1>();
    FORWARD<float, unsigned int, 2, 1>();
    FORWARD<float, unsigned int, 32, 1>();
    FORWARD<float, unsigned int, 1024, 1>();
    FORWARD<float, unsigned int, 10, 1>();
    FORWARD<float, unsigned int, 9, 1>();
    FORWARD<double, unsigned int, 200, 1>();
    FORWARD<float, unsigned int, 200, 1>();
    FORWARD<float, unsigned int, 64, 10000>();
    FORWARD<float, unsigned int, 256, 100000>();
}

template <typename T, typename TI, TI BATCH_SIZE, TI INPUT_DIM, TI HIDDEN_DIM, TI OUTPUT_DIM, TI ITERATIONS>
void BACKWARD() {
    using DEVICE_CPU = rlt::devices::DefaultCPU;
    using DEVICE_CUDA = rlt::devices::DefaultCUDA;

    constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::IDENTITY;
    using StructureSpecification = rlt::nn_models::mlp::StructureSpecification<T, TI, INPUT_DIM, OUTPUT_DIM, 3, HIDDEN_DIM, rlt::nn::activation_functions::RELU, ACTIVATION_FUNCTION, BATCH_SIZE>;

    using OPTIMIZER_PARAMETERS = rlt::nn::optimizers::adam::DefaultParametersTorch<T, typename DEVICE_CUDA::index_t>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<copy::OPTIMIZER_PARAMETERS>;
    using NNSpecification = rlt::nn_models::mlp::AdamSpecification<StructureSpecification>;

    std::cout << "BACKWARD<" << (rlt::utils::typing::is_same_v<T, float> ? "float" : "double") << ", " << BATCH_SIZE << ">" << std::endl;
    using NetworkTypeCPU = rlt::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    using NetworkTypeCUDA = rlt::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    NetworkTypeCPU network_cpu;
    NetworkTypeCPU network_cpu_pre;
    NetworkTypeCPU network_cuda_cpu;
    typename NetworkTypeCPU::template Buffer<BATCH_SIZE> network_cpu_buffers;
    NetworkTypeCUDA network_cuda;
    typename NetworkTypeCPU::template Buffer<BATCH_SIZE> network_cuda_buffers;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, OUTPUT_DIM>> d_output_cpu;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, OUTPUT_DIM>> d_output_cuda;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, INPUT_DIM>> d_input_cpu;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, INPUT_DIM>> d_input_cuda;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, INPUT_DIM>> d_input_cuda_cpu;
    rlt::malloc(device_cuda, d_output_cuda);
    rlt::malloc(device_cpu, d_output_cpu);
    rlt::malloc(device_cuda, d_input_cuda);
    rlt::malloc(device_cpu, d_input_cpu);
    rlt::malloc(device_cpu, d_input_cuda_cpu);
    OPTIMIZER optimizer_cpu, optimizer_cuda;
    rlt::malloc(device_cpu, network_cpu);
    rlt::malloc(device_cpu, network_cpu_pre);
    rlt::malloc(device_cpu, network_cuda_cpu);
    rlt::malloc(device_cpu, network_cpu_buffers);
    rlt::malloc(device_cuda, network_cuda);
    rlt::malloc(device_cuda, network_cuda_buffers);

    auto rng = rlt::random::default_engine(DEVICE_CPU::SPEC::RANDOM());

    rlt::init_weights(device_cpu, network_cpu, rng);
    rlt::zero_gradient(device_cpu, network_cpu);
    rlt::reset_optimizer_state(device_cpu, optimizer_cpu, network_cpu);
    rlt::copy(device_cpu, device_cpu, network_cpu, network_cpu_pre);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cpu;
    rlt::malloc(device_cpu, input_cpu);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cpu;
    rlt::malloc(device_cpu, output_cpu);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_target_cpu;
    rlt::malloc(device_cpu, output_target_cpu);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cuda_cpu;
    rlt::malloc(device_cpu, output_cuda_cpu);

//    for(typename NetworkTypeCPU::TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
//        for(typename NetworkTypeCPU::TI input_i = 0; input_i < NetworkTypeCPU::INPUT_DIM; input_i++){
//            set(input_cpu, batch_i, input_i, rlt::random::normal_distribution(DEVICE_CPU::SPEC::RANDOM(), (T)0, (T)1, rng));
//        }
//    }
//    for(typename NetworkTypeCPU::TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
//        for(typename NetworkTypeCPU::TI input_i = 0; input_i < NetworkTypeCPU::OUTPUT_DIM; input_i++){
//            set(output_target_cpu, batch_i, input_i, rlt::random::normal_distribution(DEVICE_CPU::SPEC::RANDOM(), (T)0, (T)1, rng));
//        }
//    }
    rlt::randn(device_cpu, input_cpu, rng);
    rlt::randn(device_cpu, output_target_cpu, rng);

    rlt::forward(device_cpu, network_cpu, input_cpu);
    rlt::nn::loss_functions::mse::gradient(device_cpu, rlt::output(network_cpu), output_target_cpu, d_output_cpu);
    rlt::backward(device_cpu, network_cpu, input_cpu, d_output_cpu, network_cpu_buffers);
    rlt::copy(device_cpu, device_cuda, network_cpu, network_cuda);


    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cuda;
    rlt::malloc(device_cuda, input_cuda);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cuda;
    rlt::malloc(device_cuda, output_cuda);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_target_cuda;
    rlt::malloc(device_cuda, output_target_cuda);

    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);
    rlt::copy(device_cpu, device_cuda, output_target_cpu, output_target_cuda);

    rlt::zero_gradient(device_cpu, network_cpu);
    rlt::zero_gradient(device_cuda, network_cuda);
//    rlt::forward_backward_mse(device_cpu, network_cpu, input_cpu, output_target_cpu, network_cpu_buffers);
    {
        rlt::forward(device_cpu, network_cpu, input_cpu);
        rlt::nn::loss_functions::mse::gradient(device_cpu, rlt::output(network_cpu), output_target_cpu, d_output_cpu);
        rlt::backward(device_cpu, network_cpu, input_cpu, d_output_cpu, network_cpu_buffers);
    }
//    rlt::forward_backward_mse(device_cuda, network_cuda, input_cuda, output_target_cuda, network_cuda_buffers);
    {
        rlt::forward(device_cuda, network_cuda, input_cuda);
        rlt::nn::loss_functions::mse::gradient(device_cuda, rlt::output(network_cuda), output_target_cuda, d_output_cuda);
        rlt::backward(device_cuda, network_cuda, input_cuda, d_output_cuda, network_cuda_buffers);
    }
    cudaDeviceSynchronize();

    rlt::copy(device_cuda, device_cpu, network_cuda, network_cuda_cpu);
//    auto evaluation_diff_pre = rlt::abs_diff(device_cpu, network_cuda_cpu, network_cpu_pre)/(BATCH_SIZE * NetworkTypeCPU::OUTPUT_DIM);
    auto evaluation_diff = rlt::abs_diff(device_cpu, network_cuda_cpu, network_cpu)/(BATCH_SIZE * NetworkTypeCPU::OUTPUT_DIM);


    std::cout << "Evaluation diff: " << evaluation_diff << std::endl;
    auto threshold = (rlt::utils::typing::is_same_v<T, float> ? 1e-6 : 1e-14);
    if(std::isnan(evaluation_diff) || evaluation_diff > threshold){
        ASSERT_LT(evaluation_diff, threshold);
    }
    {

        rlt::reset_optimizer_state(device_cpu, optimizer_cpu, network_cpu);
        rlt::reset_optimizer_state(device_cuda, optimizer_cuda, network_cuda);
        rlt::zero_gradient(device_cpu, network_cpu);
        rlt::zero_gradient(device_cuda, network_cuda);
        {
            rlt::forward(device_cpu, network_cpu, input_cpu);
            rlt::nn::loss_functions::mse::gradient(device_cpu, rlt::output(network_cpu), output_target_cpu, d_output_cpu);
            rlt::backward_input(device_cpu, network_cpu, d_output_cpu, d_input_cpu, network_cpu_buffers);
        }
        {
            rlt::forward(device_cuda, network_cuda, input_cuda);
            rlt::nn::loss_functions::mse::gradient(device_cuda, rlt::output(network_cuda), output_target_cuda, d_output_cuda);
            rlt::backward_input(device_cuda, network_cuda, d_output_cuda, d_input_cuda, network_cuda_buffers);
        }
        cudaDeviceSynchronize();

        {
            rlt::copy(device_cuda, device_cpu, network_cuda, network_cuda_cpu);
            auto evaluation_diff = rlt::abs_diff(device_cpu, network_cuda_cpu, network_cpu)/(BATCH_SIZE * NetworkTypeCPU::OUTPUT_DIM);


            std::cout << "Evaluation diff: " << evaluation_diff << std::endl;
            auto threshold = (rlt::utils::typing::is_same_v<T, float> ? 1e-6 : 1e-14);
            if(std::isnan(evaluation_diff) || evaluation_diff > threshold){
                ASSERT_LT(evaluation_diff, threshold);
            }
        }
        {

            rlt::copy(device_cuda, device_cpu, d_input_cuda, d_input_cuda_cpu);
            auto evaluation_diff = rlt::abs_diff(device_cpu, d_input_cuda_cpu, d_input_cpu);


            std::cout << "d_input diff: " << evaluation_diff << std::endl;
            auto threshold = (rlt::utils::typing::is_same_v<T, float> ? 1e-6 : 1e-14);
            if(std::isnan(evaluation_diff) || evaluation_diff > threshold){
                ASSERT_LT(evaluation_diff, threshold);
            }
        }
    }

    {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < ITERATIONS; ++i)
        {
//            rlt::forward_backward_mse(device_cuda, network_cuda, input_cuda, output_target_cuda, network_cuda_buffers);
            {
                rlt::forward(device_cuda, network_cuda, input_cuda);
                rlt::nn::loss_functions::mse::gradient(device_cuda, rlt::output(network_cuda), output_target_cuda, d_output_cuda);
                rlt::backward(device_cuda, network_cuda, input_cuda, d_output_cuda, network_cuda_buffers);
            }
            cudaDeviceSynchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "CUDA evaluation time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((T)ITERATIONS) << "us" << std::endl;
    }
}

TEST(RL_TOOLS_NN_CUDA, BACKWARD) {
    using DEFAULT_DTYPE = float;
    BACKWARD<DEFAULT_DTYPE, unsigned int,    1, 1, 1, 1, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int,    1, 256,  10, 100, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int,    2, 256,  10, 100, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int,   32, 256,  10, 100, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int, 1024, 256,  10, 100, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int,   10, 256, 200, 100, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int,    9, 256,  60, 100, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int,  200, 256,  11, 100, 1>();
    BACKWARD<double       , unsigned int,  200, 256,  12, 101, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int,   64, 256,  50, 101, 1>();
    BACKWARD<DEFAULT_DTYPE, unsigned int,  256, 256, 256, 256, 10000>();
}

template <typename T, typename TI, TI BATCH_SIZE, TI INPUT_DIM, TI HIDDEN_DIM, TI OUTPUT_DIM, TI ITERATIONS>
void ADAM_UPDATE() {
    using DEVICE_CPU = rlt::devices::DefaultCPU;
    using DEVICE_CUDA = rlt::devices::DefaultCUDA;

    constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::IDENTITY;
    using StructureSpecification = rlt::nn_models::mlp::StructureSpecification<T, TI, INPUT_DIM, OUTPUT_DIM, 3, HIDDEN_DIM, rlt::nn::activation_functions::RELU, ACTIVATION_FUNCTION, BATCH_SIZE>;

    using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<copy::OPTIMIZER_SPEC>;
    using NNSpecification = rlt::nn_models::mlp::AdamSpecification<StructureSpecification>;

    std::cout << "BACKWARD<" << (rlt::utils::typing::is_same_v<T, float> ? "float" : "double") << ", " << BATCH_SIZE << ">" << std::endl;
    using NetworkTypeCPU = rlt::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    using NetworkTypeCUDA = rlt::nn_models::mlp::NeuralNetworkAdam<NNSpecification>;
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    NetworkTypeCPU network_cpu;
    NetworkTypeCPU network_cpu_pre;
    NetworkTypeCPU network_cuda_cpu;
    typename NetworkTypeCPU::template Buffer<BATCH_SIZE> network_cpu_buffers;
    NetworkTypeCUDA network_cuda;
    typename NetworkTypeCPU::template Buffer<BATCH_SIZE> network_cuda_buffers;
    OPTIMIZER optimizer_cpu, optimizer_cuda;
    rlt::malloc(device_cpu, network_cpu);
    rlt::malloc(device_cpu, network_cpu_pre);
    rlt::malloc(device_cpu, network_cuda_cpu);
    rlt::malloc(device_cpu, network_cpu_buffers);
    rlt::malloc(device_cuda, network_cuda);
    rlt::malloc(device_cuda, network_cuda_buffers);

    auto rng = rlt::random::default_engine(DEVICE_CPU::SPEC::RANDOM());

    rlt::init_weights(device_cpu, network_cpu, rng);
    rlt::zero_gradient(device_cpu, network_cpu);
    rlt::reset_optimizer_state(device_cpu, optimizer_cpu, network_cpu);
    rlt::copy(device_cpu, device_cpu, network_cpu, network_cpu_pre);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cpu;
    rlt::malloc(device_cpu, input_cpu);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cpu;
    rlt::malloc(device_cpu, output_cpu);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_target_cpu;
    rlt::malloc(device_cpu, output_target_cpu);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cuda_cpu;
    rlt::malloc(device_cpu, output_cuda_cpu);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, OUTPUT_DIM>> d_output_cpu;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, OUTPUT_DIM>> d_output_cuda;
    rlt::malloc(device_cuda, d_output_cuda);
    rlt::malloc(device_cpu, d_output_cpu);

    for(typename NetworkTypeCPU::TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
        for(typename NetworkTypeCPU::TI input_i = 0; input_i < NetworkTypeCPU::INPUT_DIM; input_i++){
            set(input_cpu, batch_i, input_i, rlt::random::normal_distribution::sample(DEVICE_CPU::SPEC::RANDOM(), (T)0, (T)1, rng));
        }
    }
    for(typename NetworkTypeCPU::TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
        for(typename NetworkTypeCPU::TI input_i = 0; input_i < NetworkTypeCPU::OUTPUT_DIM; input_i++){
            set(output_target_cpu, batch_i, input_i, rlt::random::normal_distribution::sample(DEVICE_CPU::SPEC::RANDOM(), (T)0, (T)1, rng));
        }
    }
    {
        rlt::forward(device_cpu, network_cpu, input_cpu);
        rlt::nn::loss_functions::mse::gradient(device_cpu, rlt::output(network_cpu), output_target_cpu, d_output_cpu);
        rlt::backward(device_cpu, network_cpu, input_cpu, d_output_cpu, network_cpu_buffers);
    }
    rlt::copy(device_cpu, device_cuda, network_cpu, network_cuda);


    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cuda;
    rlt::malloc(device_cuda, input_cuda);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_cuda;
    rlt::malloc(device_cuda, output_cuda);
    rlt::MatrixDynamic<rlt::matrix::Specification<T, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::OUTPUT_DIM>> output_target_cuda;
    rlt::malloc(device_cuda, output_target_cuda);

    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);
    rlt::copy(device_cpu, device_cuda, output_target_cpu, output_target_cuda);

    rlt::zero_gradient(device_cpu, network_cpu);
    rlt::zero_gradient(device_cuda, network_cuda);
    rlt::reset_optimizer_state(device_cpu, optimizer_cuda, network_cpu);
    rlt::reset_optimizer_state(device_cuda, optimizer_cuda, network_cuda);
//    rlt::forward_backward_mse(device_cpu, network_cpu, input_cpu, output_target_cpu, network_cpu_buffers);
    {
        rlt::forward(device_cpu, network_cpu, input_cpu);
        rlt::nn::loss_functions::mse::gradient(device_cpu, rlt::output(network_cpu), output_target_cpu, d_output_cpu);
        rlt::backward(device_cpu, network_cpu, input_cpu, d_output_cpu, network_cpu_buffers);
    }
    rlt::step(device_cpu, optimizer_cpu, network_cpu);
//    rlt::forward_backward_mse(device_cuda, network_cuda, input_cuda, output_target_cuda, network_cuda_buffers);
    {
        rlt::forward(device_cuda, network_cuda, input_cuda);
        rlt::nn::loss_functions::mse::gradient(device_cuda, rlt::output(network_cuda), output_target_cuda, d_output_cuda);
        rlt::backward(device_cuda, network_cuda, input_cuda, d_output_cuda, network_cuda_buffers);
    }
    rlt::step(device_cuda, optimizer_cuda, network_cuda);
    cudaDeviceSynchronize();

    rlt::copy(device_cuda, device_cpu, network_cuda, network_cuda_cpu);
    auto evaluation_diff_pre = rlt::abs_diff(device_cpu, network_cuda_cpu, network_cpu_pre)/(BATCH_SIZE * NetworkTypeCPU::OUTPUT_DIM);
    auto evaluation_diff = rlt::abs_diff(device_cpu, network_cuda_cpu, network_cpu)/(BATCH_SIZE * NetworkTypeCPU::OUTPUT_DIM);

    std::cout << "Evaluation diff: " << evaluation_diff << std::endl;
    auto threshold = (rlt::utils::typing::is_same_v<T, float> ? 1e-6 : 1e-14);
    if(std::isnan(evaluation_diff) || evaluation_diff > threshold){
        ASSERT_LT(evaluation_diff, threshold);
    }

    {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < ITERATIONS; ++i)
        {
//            rlt::forward_backward_mse(device_cuda, network_cuda, input_cuda, output_target_cuda, network_cuda_buffers);
            {
                rlt::forward(device_cuda, network_cuda, input_cuda);
                rlt::nn::loss_functions::mse::gradient(device_cuda, rlt::output(network_cuda), output_target_cuda, d_output_cuda);
                rlt::backward(device_cuda, network_cuda, input_cuda, d_output_cuda, network_cuda_buffers);
            }
            rlt::step(device_cuda, optimizer_cuda, network_cuda);
            cudaDeviceSynchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "CUDA evaluation time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / ((T)ITERATIONS) << "us" << std::endl;
    }
}

TEST(RL_TOOLS_NN_CUDA, ADAM_UPDATE) {
    using DEFAULT_DTYPE = float;
    ADAM_UPDATE<DEFAULT_DTYPE, unsigned int,    1, 256,  10, 100, 1>();
    ADAM_UPDATE<DEFAULT_DTYPE, unsigned int,    2, 256,  10, 100, 1>();
    ADAM_UPDATE<DEFAULT_DTYPE, unsigned int,   32, 256,  10, 100, 1>();
    ADAM_UPDATE<DEFAULT_DTYPE, unsigned int, 1024, 256,  10, 100, 1>();
    ADAM_UPDATE<DEFAULT_DTYPE, unsigned int,   10, 256, 200, 100, 1>();
    ADAM_UPDATE<DEFAULT_DTYPE, unsigned int,    9, 256,  60, 100, 1>();
    ADAM_UPDATE<DEFAULT_DTYPE, unsigned int,  200, 256,  11, 100, 1>();
    ADAM_UPDATE<double       , unsigned int,  200, 256,  12, 101, 1>();
    ADAM_UPDATE<DEFAULT_DTYPE, unsigned int,   64, 256,  50, 101, 10000>();
    ADAM_UPDATE<DEFAULT_DTYPE, unsigned int,  256, 256, 256, 256, 10000>();
}