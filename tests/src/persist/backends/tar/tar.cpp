#include <rl_tools/operations/cpu.h>
#include <rl_tools/persist/backends/tar/operations_cpu.h>
#include <rl_tools/persist/backends/tar/operations_generic.h>

#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>

#define RL_TOOLS_STRINGIZE(x) #x
#define RL_TOOLS_MACRO_TO_STR(macro) RL_TOOLS_STRINGIZE(macro)


namespace rlt = rl_tools;

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

using DEVICE = rl_tools::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = typename DEVICE::index_t;
using T = float;

#include <gtest/gtest.h>


TEST(TESTS_PERSIST_BACKENDS_TAR_TAR, test) {
    std::string content1 = "This is the first buffer's content.";
    std::vector<char> buffer1(content1.begin(), content1.end());

    std::string content2 = "This data is for the second entry in our archive!";
    std::vector<char> buffer2(content2.begin(), content2.end());

    const std::filesystem::path archive_path = "test_persist_backends_tar_test.tar";
    std::ofstream archive(archive_path, std::ios::binary);
    std::cout << "Creating archive: " << archive_path << std::endl;

    std::cout << "Writing 'buffer1.txt'..." << std::endl;
    DEVICE device;
    rlt::persist::backends::tar::Writer writer;
    rlt::persist::backends::tar::write_entry(device, writer, "buffer1.txt", buffer1.data(), buffer1.size());

    std::cout << "Writing 'entry2.log'..." << std::endl;
    rlt::persist::backends::tar::write_entry(device, writer, "entry2.log",  buffer2.data(), buffer2.size());

    rlt::persist::backends::tar::finalize(device, writer);

    archive.write(writer.buffer.data(), writer.buffer.size());

    archive.close();

    std::ifstream archive_file("test_persist_backends_tar_test.tar", std::ios::binary);
    std::vector<char> tar_data((std::istreambuf_iterator<char>(archive_file)), std::istreambuf_iterator<char>());
    archive_file.close();


    char buffer[500];
    TI read_size;
    rlt::persist::backends::tar::get(device, tar_data.data(), tar_data.size(), "buffer1.txt", buffer, sizeof(buffer), read_size);
    ASSERT_TRUE(rlt::persist::backends::tar::strcmp("abcdefg", "abcdefg", 7));
    ASSERT_FALSE(rlt::persist::backends::tar::strcmp("abcdefg", "abbdefg", 7));
    ASSERT_TRUE(rlt::persist::backends::tar::strcmp("abcdefg", "abcdefg ", 7));
    ASSERT_FALSE(rlt::persist::backends::tar::strcmp("abcdefg", "abcdefg ", 8));
    ASSERT_FALSE(rlt::persist::backends::tar::strcmp("abcdefg", "", 7));
    ASSERT_TRUE(rlt::persist::backends::tar::strcmp(buffer, content1.c_str(), content1.size()));
    rlt::persist::backends::tar::get(device, tar_data.data(), tar_data.size(), "entry2.log", buffer, sizeof(buffer), read_size);
    ASSERT_TRUE(rlt::persist::backends::tar::strcmp(buffer, content2.c_str(), content2.size()));
}

TEST(TEST_PERSIST_BACKENDS_TAR_TAR, tensor) {
    DEVICE device;
    RNG rng;
    constexpr TI seed = 0;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 5, 3, 2, 10>>> A, A_read_back;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 5, 4, 2, 10>>> A_read_back_fail;
    rlt::malloc(device, A);
    rlt::malloc(device, A_read_back);
    rlt::malloc(device, A_read_back_fail);
    rlt::randn(device, A, rng);
    rlt::print(device, A);
    std::string data_file_name = "test_persist_backends_tar_tensor.tar";
    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/" + data_file_name;
    rlt::persist::backends::tar::Writer writer;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> writer_group{"", writer};
    auto container_group = rlt::create_group(device, writer_group, "container");
    rlt::save(device, A, container_group, "A");
    rlt::persist::backends::tar::finalize(device, writer_group.writer);
    std::ofstream archive(data_file_path, std::ios::binary);
    std::cout << "Creating archive: " << data_file_path << std::endl;
    archive.write(writer_group.writer.buffer.data(), writer_group.writer.buffer.size());
    archive.close();
    std::ifstream archive_file(data_file_path, std::ios::binary);
    std::vector<char> tar_data((std::istreambuf_iterator<char>(archive_file)), std::istreambuf_iterator<char>());
    archive_file.close();
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> reader_group;
    reader_group.data = tar_data.data();
    reader_group.size = tar_data.size();
    auto reader_container_group = rlt::create_group(device, reader_group, "container");
    rlt::load(device, A_read_back, reader_container_group, "A");

    rlt::print(device, A_read_back);

    T abs_diff = rlt::abs_diff(device, A, A_read_back);
    ASSERT_NEAR(abs_diff, 0, 1e-6);
}

TEST(TEST_PERSIST_BACKENDS_TAR_TAR, tensor_attribute) {
    DEVICE device;
    RNG rng;
    constexpr TI seed = 0;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 5, 3, 2, 10>>> A, A_read_back;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 5, 4, 2, 10>>> A_read_back_fail;
    rlt::malloc(device, A);
    rlt::malloc(device, A_read_back);
    rlt::malloc(device, A_read_back_fail);
    rlt::randn(device, A, rng);
    rlt::print(device, A);
    std::string data_file_name = "test_persist_backends_tar_tensor_attribute.tar";
    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/" + data_file_name;
    rlt::persist::backends::tar::Writer writer;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> writer_group{"", writer};
    auto container_group = rlt::create_group(device, writer_group, "container");
    rlt::save(device, A, container_group, "A");
    rlt::set_attribute(device, writer_group, "id", 1337);
    rlt::write_attributes(device, writer_group);
    rlt::persist::backends::tar::finalize(device, writer_group.writer);
    std::ofstream archive(data_file_path, std::ios::binary);
    std::cout << "Creating archive: " << data_file_path << std::endl;
    archive.write(writer_group.writer.buffer.data(), writer_group.writer.buffer.size());
    archive.close();
    std::ifstream archive_file(data_file_path, std::ios::binary);
    std::vector<char> tar_data((std::istreambuf_iterator<char>(archive_file)), std::istreambuf_iterator<char>());
    archive_file.close();
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> reader_group;
    reader_group.data = tar_data.data();
    reader_group.size = tar_data.size();
    auto reader_container_group = rlt::create_group(device, reader_group, "container");
    rlt::load(device, A_read_back, reader_container_group, "A");

    TI id = rlt::get_attribute_int<TI>(device, reader_group, "id");
    ASSERT_EQ(id, 1337);

    rlt::print(device, A_read_back);

    T abs_diff = rlt::abs_diff(device, A, A_read_back);
    ASSERT_NEAR(abs_diff, 0, 1e-6);
}

TEST(TEST_PERSIST_BACKENDS_TAR_TAR, matrix) {
    DEVICE device;
    RNG rng;
    constexpr TI seed = 0;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    rlt::Matrix<rlt::matrix::Specification<T, TI, 5, 3>> A, A_read_back;
    rlt::malloc(device, A);
    rlt::malloc(device, A_read_back);
    rlt::randn(device, A, rng);
    rlt::print(device, A);
    std::string data_file_name = "test_persist_backends_tar_matrix.tar";
    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/" + data_file_name;
    rlt::persist::backends::tar::Writer writer;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> writer_group{"", writer};
    rlt::save(device, A, writer_group, "A");
    rlt::persist::backends::tar::finalize(device, writer_group.writer);
    std::ofstream archive(data_file_path, std::ios::binary);
    std::cout << "Creating archive: " << data_file_path << std::endl;
    archive.write(writer_group.writer.buffer.data(), writer_group.writer.buffer.size());
    archive.close();
    std::ifstream archive_file(data_file_path, std::ios::binary);
    std::vector<char> tar_data((std::istreambuf_iterator<char>(archive_file)), std::istreambuf_iterator<char>());
    archive_file.close();
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> reader_group;
    reader_group.data = tar_data.data();
    reader_group.size = tar_data.size();
    rlt::load(device, A_read_back, reader_group, "A");

    rlt::print(device, A_read_back);

    T abs_diff = rlt::abs_diff(device, A, A_read_back);
    ASSERT_NEAR(abs_diff, 0, 1e-6);
}

TEST(TEST_PERSIST_BACKENDS_TAR_TAR, dense_layer) {
    DEVICE device;
    RNG rng;
    constexpr TI seed = 0;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);

    using CONFIG = rlt::nn::layers::dense::Configuration<T, TI, 10, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CAPABILITY = rlt::nn::capability::Forward<>;
    using SPEC = rlt::nn::layers::dense::Specification<CONFIG, CAPABILITY, rlt::tensor::Shape<TI, 1, 10, 15>>;
    rlt::nn::layers::dense::LayerForward<SPEC> layer, layer_read_back;
    rlt::malloc(device, layer);
    rlt::malloc(device, layer_read_back);
    rlt::init_weights(device, layer, rng);

    std::string data_file_name = "test_persist_backends_tar_dense_layer.tar";
    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/" + data_file_name;
    rlt::persist::backends::tar::Writer writer;
    rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> writer_group{"", writer};
    auto layer_group = rlt::create_group(device, writer_group, "layer");
    rlt::save(device, layer, layer_group);
    rlt::persist::backends::tar::finalize(device, writer_group.writer);
    std::ofstream archive(data_file_path, std::ios::binary);
    std::cout << "Creating archive: " << data_file_path << std::endl;
    archive.write(writer_group.writer.buffer.data(), writer_group.writer.buffer.size());
    archive.close();

    std::ifstream archive_data(data_file_path, std::ios::binary);
    std::vector<char> tar_data((std::istreambuf_iterator<char>(archive_data)), std::istreambuf_iterator<char>());
    archive_data.close();
    rlt::persist::backends::tar::ReaderGroup<rlt::persist::backends::tar::ReaderGroupSpecification<TI>> reader_group{"", tar_data.data(), tar_data.size()};
    auto layer_group_readback = rlt::get_group(device, reader_group, "layer");
    rlt::load(device, layer_read_back, layer_group_readback);

    T abs_diff = rlt::abs_diff(device, layer, layer_read_back);
    std::cout << "Layer abs diff: " << abs_diff << std::endl;
    ASSERT_NEAR(abs_diff, 0, 1e-6);

}
