#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/metrics.h>

#include <metra/metra.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace task = rlt::rl::environments::hyperdrone::tasks::visual_inertial_localization;

using DEVICE = rlt::devices::DefaultCPU;
using T = double;
using TI = typename DEVICE::index_t;

// scores a causal estimator's trajectory (TUM format: t x y z qx qy qz qw) against the
// exported EuRoC ground truth: no alignment — the estimate is anchored at its own first pose,
// the ground truth at the matching timestamp, and both are compared as poses relative to those
// anchors (the initialization delay is reported, not hidden)
struct Pose {
    double time;
    T position[3];
    T orientation[4]; // w, x, y, z
};

static bool load_ground_truth(const std::string& path, std::vector<Pose>& poses){
    std::ifstream file(path);
    if(!file.is_open()){
        return false;
    }
    std::string line;
    while(std::getline(file, line)){
        if(line.empty() || line[0] == '#'){
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        double t_ns;
        Pose pose;
        if(!(row >> t_ns >> pose.position[0] >> pose.position[1] >> pose.position[2] >> pose.orientation[0] >> pose.orientation[1] >> pose.orientation[2] >> pose.orientation[3])){
            continue;
        }
        pose.time = t_ns / 1e9;
        poses.push_back(pose);
    }
    return !poses.empty();
}

static bool load_tum(const std::string& path, std::vector<Pose>& poses){
    std::ifstream file(path);
    if(!file.is_open()){
        return false;
    }
    std::string line;
    while(std::getline(file, line)){
        if(line.empty() || line[0] == '#'){
            continue;
        }
        std::istringstream row(line);
        Pose pose;
        T qx, qy, qz, qw;
        if(!(row >> pose.time >> pose.position[0] >> pose.position[1] >> pose.position[2] >> qx >> qy >> qz >> qw)){
            continue;
        }
        pose.orientation[0] = qw;
        pose.orientation[1] = qx;
        pose.orientation[2] = qy;
        pose.orientation[3] = qz;
        poses.push_back(pose);
    }
    return !poses.empty();
}

static const Pose* nearest(const std::vector<Pose>& poses, double time, double tolerance){
    auto it = std::lower_bound(poses.begin(), poses.end(), time, [](const Pose& pose, double t){ return pose.time < t; });
    const Pose* best = nullptr;
    double best_distance = tolerance;
    for(auto candidate : {it, it == poses.begin() ? it : it - 1}){
        if(candidate != poses.end()){
            double distance = std::abs(candidate->time - time);
            if(distance <= best_distance){
                best_distance = distance;
                best = &*candidate;
            }
        }
    }
    return best;
}

int main(int argc, char** argv){
    if(argc < 3){
        std::fprintf(stderr, "usage: %s <euroc_dataset_dir_or_groundtruth_csv> <estimate.tum> [name]\n", argv[0]);
        return 1;
    }
    std::string ground_truth_path = argv[1];
    if(std::filesystem::is_directory(ground_truth_path)){
        ground_truth_path = (std::filesystem::path(ground_truth_path) / "mav0" / "state_groundtruth_estimate0" / "data.csv").string();
    }
    std::string estimate_path = argv[2];
    std::string name = argc > 3 ? argv[3] : "external";

    std::vector<Pose> ground_truth, estimate;
    if(!load_ground_truth(ground_truth_path, ground_truth)){
        std::fprintf(stderr, "failed to load ground truth from %s\n", ground_truth_path.c_str());
        return 1;
    }
    if(!load_tum(estimate_path, estimate)){
        std::fprintf(stderr, "failed to load estimate from %s\n", estimate_path.c_str());
        return 1;
    }
    const double ground_truth_dt = ground_truth.size() > 1 ? ground_truth[1].time - ground_truth[0].time : 0.005;
    const double tolerance = ground_truth_dt * 0.51;

    DEVICE device;
    rlt::init(device);
    const Pose& estimate_anchor = estimate.front();
    const Pose* ground_truth_anchor = nearest(ground_truth, estimate_anchor.time, tolerance);
    if(ground_truth_anchor == nullptr){
        std::fprintf(stderr, "no ground truth sample within %.4fs of the first estimate (t=%.4f)\n", tolerance, estimate_anchor.time);
        return 1;
    }
    task::TrajectoryMetricsAccumulator<T, TI> accumulator{};
    task::reset(device, accumulator);
    TI matched = 0, unmatched = 0;
    for(const Pose& sample : estimate){
        const Pose* reference = nearest(ground_truth, sample.time, tolerance);
        if(reference == nullptr){
            unmatched++;
            continue;
        }
        T estimate_relative_position[3], estimate_relative_orientation[4];
        T ground_truth_relative_position[3], ground_truth_relative_orientation[4];
        task::relative_pose(device, estimate_anchor.position, estimate_anchor.orientation, sample.position, sample.orientation, estimate_relative_position, estimate_relative_orientation);
        task::relative_pose(device, ground_truth_anchor->position, ground_truth_anchor->orientation, reference->position, reference->orientation, ground_truth_relative_position, ground_truth_relative_orientation);
        task::accumulate(device, accumulator, ground_truth_relative_position, ground_truth_relative_orientation, estimate_relative_position, estimate_relative_orientation);
        matched++;
    }
    if(matched == 0){
        std::fprintf(stderr, "no estimate samples matched the ground truth timeline\n");
        return 1;
    }
    const double initialization_time = estimate_anchor.time;
    const double scored_span = estimate.back().time - estimate_anchor.time;
    std::printf("%s: %lu/%lu samples scored (anchor/init at t=%.3f s, span %.3f s)\n", name.c_str(), (unsigned long)matched, (unsigned long)(matched + unmatched), initialization_time, scored_span);
    std::printf("  ATE %.3f m | rot mean %.4f rad max %.4f rad | drift %.4f m/m | distance %.2f m\n",
        (double)task::ate_position_rmse(device, accumulator),
        (double)task::mean_rotation_error(device, accumulator),
        (double)task::max_rotation_error(device, accumulator),
        (double)task::drift_per_distance(device, accumulator),
        (double)accumulator.distance_traveled);
    const std::string metra_prefix = "hyperdrone/visual_inertial_localization/" + name;
    metra::log(metra_prefix + "/ate_position_rmse", (double)task::ate_position_rmse(device, accumulator));
    metra::log(metra_prefix + "/rotation_error_mean", (double)task::mean_rotation_error(device, accumulator));
    metra::log(metra_prefix + "/rotation_error_max", (double)task::max_rotation_error(device, accumulator));
    metra::log(metra_prefix + "/drift_per_distance", (double)task::drift_per_distance(device, accumulator));
    metra::log(metra_prefix + "/distance_traveled", (double)accumulator.distance_traveled);
    metra::log(metra_prefix + "/initialization_time", initialization_time);
    return 0;
}
