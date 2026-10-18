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
    if(shapes?.length !== 2 || shapes[0].slice(2).join(",") !== "50,320,8" || shapes[1].slice(2).join(",") !== "62" || model.output_dim !== 4) {
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
        observation: environment.observation,
    }
}

// stacked: per pixel [frame 0, ..., frame n - 1, target] RGB; output: pairs [frame, target], [target, frame] per frame side by side, padded to 8 channels
export function packVisualPairs(stacked, frames, width, height) {
    const channels = 3 * (frames + 1)
    if(stacked.length !== width * height * channels) {
        throw new Error("Camera images do not match the checkpoint resolution")
    }
    const pairs = 2 * frames
    const output = new Float32Array(height * width * pairs * 8)
    for(let y = 0; y < height; y++) {
        for(let x = 0; x < width; x++) {
            const source = (y * width + x) * channels
            for(let frame = 0; frame < frames; frame++) {
                const forward = (y * width * pairs + 2 * frame * width + x) * 8
                const reversed = forward + width * 8
                for(let c = 0; c < 3; c++) {
                    const image = stacked[source + 3 * frame + c]
                    const target = stacked[source + 3 * frames + c]
                    output[forward + c] = image
                    output[forward + 3 + c] = target
                    output[reversed + c] = target
                    output[reversed + 3 + c] = image
                }
            }
        }
    }
    return output
}
