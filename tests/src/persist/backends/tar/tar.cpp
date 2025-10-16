#include <rl_tools/operations/cpu.h>
#include <rl_tools/persist/backends/tar/operations_cpu.h>

namespace rlt = rl_tools;

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

using DEVICE = rl_tools::devices::DefaultCPU;
using TI = typename DEVICE::index_t;


int main() {
    // 1. Define some in-memory data buffers
    std::string content1 = "This is the first buffer's content.";
    std::vector<char> buffer1(content1.begin(), content1.end());

    std::string content2 = "This data is for the second entry in our archive!";
    std::vector<char> buffer2(content2.begin(), content2.end());

    // 2. Create and write to the tar archive
    const std::filesystem::path archive_path = "test_header_only.tar";
    std::ofstream archive(archive_path, std::ios::binary);
    if (!archive) {
        std::cerr << "Failed to create archive file." << std::endl;
        return 1;
    }

    std::cout << "Creating archive: " << archive_path << std::endl;

    std::cout << "Writing 'buffer1.txt'..." << std::endl;
    DEVICE device;
    rlt::persist::backends::tar::Writer writer;
    rlt::persist::backends::tar::write_entry(device, writer, "buffer1.txt", buffer1.data(), buffer1.size());

    std::cout << "Writing 'entry2.log'..." << std::endl;
    rlt::persist::backends::tar::write_entry(device, writer, "entry2.log",  buffer2.data(), buffer2.size());

    rlt::persist::backends::tar::finalize(device, writer);

    archive.write(writer.buffer.data(), writer.buffer.size());

    // std::cout << "Archive finalized." << std::endl << std::endl;
    //
    // // 3. Read the archive back from disk
    // std::cout << "--- Reading contents from " << archive_path << " ---" << std::endl;
    // auto entries = tar::read_archive(archive_path);
    //
    // if (entries.empty()) {
    //     std::cerr << "Failed to read any entries from the archive." << std::endl;
    //     return 1;
    // }
    //
    // for (const auto& [name, data] : entries) {
    //     std::cout << "Entry: " << name << " (" << data.size() << " bytes)" << std::endl;
    //     std::cout << "  Content: '";
    //     std::cout.write(data.data(), data.size());
    //     std::cout << "'" << std::endl;
    // }
    // std::cout << "---------------------------------------" << std::endl;

    return 0;
}
