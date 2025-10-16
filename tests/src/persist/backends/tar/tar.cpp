#include <rl_tools/operations/cpu.h>
#include <rl_tools/persist/backends/tar/operations_cpu.h>
#include <rl_tools/persist/backends/tar/operations_generic.h>

namespace rlt = rl_tools;

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

using DEVICE = rl_tools::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

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
    rlt::persist::backends::tar::get(device, tar_data.data(), tar_data.size(), "buffer1.txt", buffer, sizeof(buffer));
    ASSERT_TRUE(rlt::persist::backends::tar::strcmp("abcdefg", "abcdefg", 7));
    ASSERT_FALSE(rlt::persist::backends::tar::strcmp("abcdefg", "abbdefg", 7));
    ASSERT_TRUE(rlt::persist::backends::tar::strcmp("abcdefg", "abcdefg ", 7));
    ASSERT_FALSE(rlt::persist::backends::tar::strcmp("abcdefg", "abcdefg ", 8));
    ASSERT_FALSE(rlt::persist::backends::tar::strcmp("abcdefg", "", 7));
    ASSERT_TRUE(rlt::persist::backends::tar::strcmp(buffer, content1.c_str(), content1.size()));
    rlt::persist::backends::tar::get(device, tar_data.data(), tar_data.size(), "entry2.log", buffer, sizeof(buffer));
    ASSERT_TRUE(rlt::persist::backends::tar::strcmp(buffer, content2.c_str(), content2.size()));
}
