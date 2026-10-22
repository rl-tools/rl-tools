#pragma once
#include <rl_tools/rl/environments/hyperdrone/world.h>

namespace hyperdrone_camera_config {
    namespace outputs = rl_tools::hyperlux::outputs;
    using CAMERA_OUTPUTS = rl_tools::rl::environments::hyperdrone::CameraOutputs<outputs::List<outputs::RGB<outputs::format::Float32>>>;
    using OBSERVATION = rl_tools::rl::environments::hyperdrone::ImageObservation<0>;
}
