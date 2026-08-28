// end-to-end smoke for the AI2ThorHab dataset source: enumerate the raw checkout, convert (or
// hit the cache) and load one scene into a Bundle, report the metadata
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>

#include <cstdlib>
#include <iostream>

namespace rlt = rl_tools;

int main(int argc, char** argv) {
    using DEVICE = rlt::devices::DEVICE_FACTORY<>;
    using T = float;
    DEVICE device;
    rlt::init(device);

    rlt::rendering::datasets::procthor::AI2ThorHab dataset;
    dataset.root = argc > 1 ? argv[1] : "/data/ai2thor-hab/ai2thor-hab";
    dataset.split = argc > 2 ? argv[2] : "Test";
    const size_t index = argc > 3 ? (size_t)std::atoll(argv[3]) : 0;

    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    std::cout << "corpus: " << corpus.references.size() << " scenes; [0]=" << corpus.references.front() << "\n";

    rlt::rendering::Bundle<T> bundle;
    if (!rlt::rendering::datasets::procthor::load(device, dataset, corpus, index, bundle)) {
        std::cerr << "load failed\n";
        return 1;
    }
    std::cout << "scene[" << index << "]: objects=" << bundle.scene.objects.size()
              << " instances=" << bundle.scene.instances.size()
              << " lights=" << bundle.scene.lights.size() << "\n";
    std::cout << "metadata: center=[" << bundle.metadata.center[0] << "," << bundle.metadata.center[1] << "," << bundle.metadata.center[2]
              << "] max_ray_length=" << bundle.metadata.max_ray_length
              << " hash=" << bundle.metadata.content_hash << "\n";
    return 0;
}
