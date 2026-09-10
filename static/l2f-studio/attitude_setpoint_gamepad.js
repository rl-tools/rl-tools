export const ATTITUDE_SETPOINT_GAMEPAD_STORAGE_KEY = "attitude_setpoint_gamepad_config"

export const ATTITUDE_SETPOINT_GAMEPAD_INTERFACE = {
    "roll": {
        type: "axis",
        positive_direction: "Right"
    },
    "pitch": {
        type: "axis",
        positive_direction: "Forward"
    },
    "yaw_rate": {
        type: "axis",
        positive_direction: "Clockwise"
    },
    "thrust_g": {
        type: "axis",
        positive_direction: "Up"
    },
}

const DEFAULT_LIMITS = {
    max_tilt_angle: 0.5235987755982988,
    max_yaw_rate: 2.0,
    thrust_min_g: 0.4,
    thrust_max_g: 1.4,
}

const finite_or = (value, fallback) => {
    const number = Number(value)
    return Number.isFinite(number) ? number : fallback
}

const clamp = (value, min, max) => Math.max(min, Math.min(max, value))

export class AttitudeSetpointGamepadInput{
    constructor(gamepad){
        this.output = null
        gamepad.addListener((output) => {
            this.output = output
        })
    }

    get_limits(parameters){
        const sampling = parameters?.attitude_setpoint_sampling || {}
        return {
            max_tilt_angle: finite_or(sampling.max_tilt_angle, DEFAULT_LIMITS.max_tilt_angle),
            max_yaw_rate: finite_or(sampling.max_yaw_rate, DEFAULT_LIMITS.max_yaw_rate),
            thrust_min_g: finite_or(sampling.thrust_min_g, DEFAULT_LIMITS.thrust_min_g),
            thrust_max_g: finite_or(sampling.thrust_max_g, DEFAULT_LIMITS.thrust_max_g),
        }
    }

    getObservation(parameters){
        const limits = this.get_limits(parameters)
        const output = this.output || {}
        let roll_axis = finite_or(output.roll, 0)
        let pitch_axis = finite_or(output.pitch, 0)
        const tilt_norm = Math.hypot(roll_axis, pitch_axis)
        if(tilt_norm > 1){
            roll_axis /= tilt_norm
            pitch_axis /= tilt_norm
        }

        const thrust_min = limits.thrust_min_g
        const thrust_max = limits.thrust_max_g
        const neutral_thrust = clamp(1, thrust_min, thrust_max)
        const thrust_axis = finite_or(output.thrust_g, 0)
        const thrust_scale = thrust_axis >= 0 ? thrust_max - neutral_thrust : neutral_thrust - thrust_min

        return [
            roll_axis * limits.max_tilt_angle,
            pitch_axis * limits.max_tilt_angle,
            // Training target yaw_rate is body +Z angular velocity. In the FLU frame,
            // positive +Z is counter-clockwise from above, while the UI maps clockwise as positive.
            -finite_or(output.yaw_rate, 0) * limits.max_yaw_rate,
            clamp(neutral_thrust + thrust_axis * thrust_scale, thrust_min, thrust_max),
        ]
    }
}
