export function visualPolicyProfile(model) {
    const environment = model.meta?.environment
    const shapes = model.input_shapes
    const sensors = "OrientationMahonyWorldZ.GyroscopeMahonyBiasCorrected.ActionHistory(8).LinearAccelerationBodyFrameHistory(8)"
    if(environment?.name !== "hyperdrone" || !environment.observation.includes(sensors)) {
        return null
    }
    if(!shapes) {
        throw new Error("Rebuild Studio's WASM modules to load this mixed visual checkpoint")
    }
    if(shapes?.length !== 2 || shapes[0].slice(2).join(",") !== "50,160,8" || shapes[1].slice(2).join(",") !== "65" || model.output_dim !== 4) {
        throw new Error("Unsupported mixed visual checkpoint input shapes")
    }
    if(environment.rendering?.image_rectification !== false) {
        throw new Error("This mixed visual policy requires raw camera images")
    }
    if(environment.rendering.gyro_bias_tracking_tau_s !== 10 || environment.rendering.gyro_calibration_residual_std_rad_s !== 0.001) {
        throw new Error("The checkpoint's gyro calibration does not match Studio's mixed visual configuration")
    }
    return {
        state_type: "VisualState",
        dynamics: "crazyflie_openmv",
        observation: environment.observation.endsWith(".InertialVelocity") ? environment.observation : environment.observation + ".InertialVelocity",
    }
}

export function packVisualPair(current, target, width, height) {
    if(current.length !== width * height * 3 || target.length !== current.length) {
        throw new Error("Camera images do not match the checkpoint resolution")
    }
    const output = new Float32Array(height * width * 2 * 8)
    for(let y = 0; y < height; y++) {
        for(let x = 0; x < width; x++) {
            const source = (y * width + x) * 3
            const left = (y * width * 2 + x) * 8
            const right = left + width * 8
            for(let c = 0; c < 3; c++) {
                output[left + c] = current[source + c]
                output[left + 3 + c] = target[source + c]
                output[right + c] = target[source + c]
                output[right + 3 + c] = current[source + c]
            }
        }
    }
    return output
}
