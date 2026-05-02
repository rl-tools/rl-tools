function targetWorldZBody(setpoint){
    const roll = setpoint.roll
    const pitch = setpoint.pitch
    return [
        -Math.sin(pitch),
        Math.cos(pitch) * Math.sin(roll),
        Math.cos(pitch) * Math.cos(roll),
    ]
}

function quatToWorldZBody(q){
    return [
        2*q[1]*q[3] - 2*q[0]*q[2],
        2*q[2]*q[3] + 2*q[0]*q[1],
        1 - 2*q[1]*q[1] - 2*q[2]*q[2],
    ]
}

function rollFromWorldZBody(z){
    return Math.atan2(z[1], z[2])
}

function pitchFromWorldZBody(z){
    return Math.atan2(-z[0], Math.sqrt(z[1]*z[1] + z[2]*z[2]))
}

function cross(a, b){
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]
}

function rotateVectorByQuaternion(q, v){
    const qv = [q[1], q[2], q[3]]
    const t0 = cross(qv, v).map((x) => 2*x)
    const t1 = cross(qv, t0)
    return [
        v[0] + q[0]*t0[0] + t1[0],
        v[1] + q[0]*t0[1] + t1[1],
        v[2] + q[0]*t0[2] + t1[2],
    ]
}

function specificForceBodyG(state){
    const gravity = [0, 0, -9.81]
    const gravityNorm = 9.81
    if(!state.linear_acceleration || !state.orientation){
        return [NaN, NaN, NaN]
    }
    const specificForceWorld = [
        state.linear_acceleration[0] - gravity[0],
        state.linear_acceleration[1] - gravity[1],
        state.linear_acceleration[2] - gravity[2],
    ]
    const q = state.orientation
    const qConj = [q[0], -q[1], -q[2], -q[3]]
    return rotateVectorByQuaternion(qConj, specificForceWorld).map((x) => x / gravityNorm)
}

function attitudeError(sample){
    const state = sample.state
    const target = targetWorldZBody(state.attitude_setpoint)
    const actual = quatToWorldZBody(state.orientation)
    let acc = 0
    for(let i = 0; i < 3; i++){
        const diff = actual[i] - target[i]
        acc += diff * diff
    }
    return Math.sqrt(acc)
}

function ctbrRateLimit(episode){
    if(episode.parameters && episode.parameters.ctbr_controller && episode.parameters.ctbr_controller.rate_limit){
        return episode.parameters.ctbr_controller.rate_limit
    }
    return [4, 4, 4]
}

function ctbrRateSetpoint(sample, rateLimit, axis){
    return sample.action[axis + 1] * rateLimit[axis]
}

const run = data[Object.keys(data)[0]]
for(let trajectory_i = 0; trajectory_i < Math.min(10, run.length); trajectory_i++){
    const episode = run[trajectory_i]
    const trajectory = episode.trajectory
    const rateLimit = ctbrRateLimit(episode)

    output.scatter([
        {label: "target roll", data: trajectory.map((x, i) => {return {x: i, y: x.state.attitude_setpoint.roll}})},
        {label: "actual roll", data: trajectory.map((x, i) => {return {x: i, y: rollFromWorldZBody(quatToWorldZBody(x.state.orientation))}})},
        {label: "mahony roll", data: trajectory.map((x, i) => {return {x: i, y: x.state.world_z_body_estimate ? rollFromWorldZBody(x.state.world_z_body_estimate) : NaN}})},
        {label: "target pitch", data: trajectory.map((x, i) => {return {x: i, y: x.state.attitude_setpoint.pitch}})},
        {label: "actual pitch", data: trajectory.map((x, i) => {return {x: i, y: pitchFromWorldZBody(quatToWorldZBody(x.state.orientation))}})},
        {label: "mahony pitch", data: trajectory.map((x, i) => {return {x: i, y: x.state.world_z_body_estimate ? pitchFromWorldZBody(x.state.world_z_body_estimate) : NaN}})},
    ], "step", "rad", `Trajectory ${trajectory_i}: reduced attitude`)

    output.scatter([
        {label: "target yaw rate", data: trajectory.map((x, i) => {return {x: i, y: x.state.attitude_setpoint.yaw_rate}})},
        {label: "policy commanded yaw rate", data: trajectory.map((x, i) => {return {x: i, y: ctbrRateSetpoint(x, rateLimit, 2)}})},
        {label: "actual yaw rate", data: trajectory.map((x, i) => {return {x: i, y: x.state.angular_velocity[2]}})},
        {label: "yaw rate error", data: trajectory.map((x, i) => {return {x: i, y: x.state.angular_velocity[2] - x.state.attitude_setpoint.yaw_rate}})},
    ], "step", "rad/s", `Trajectory ${trajectory_i}: yaw-rate tracking`)

    output.scatter([
        {label: "ctbr target roll rate", data: trajectory.map((x, i) => {return {x: i, y: ctbrRateSetpoint(x, rateLimit, 0)}})},
        {label: "actual roll rate", data: trajectory.map((x, i) => {return {x: i, y: x.state.angular_velocity[0]}})},
        {label: "ctbr target pitch rate", data: trajectory.map((x, i) => {return {x: i, y: ctbrRateSetpoint(x, rateLimit, 1)}})},
        {label: "actual pitch rate", data: trajectory.map((x, i) => {return {x: i, y: x.state.angular_velocity[1]}})},
        {label: "ctbr target yaw rate", data: trajectory.map((x, i) => {return {x: i, y: ctbrRateSetpoint(x, rateLimit, 2)}})},
        {label: "actual yaw rate", data: trajectory.map((x, i) => {return {x: i, y: x.state.angular_velocity[2]}})},
        {label: "outer target yaw rate", data: trajectory.map((x, i) => {return {x: i, y: x.state.attitude_setpoint.yaw_rate}})},
    ], "step", "rad/s", `Trajectory ${trajectory_i}: CTBR body-rate tracking`)

    output.scatter([
        {label: "roll-rate error", data: trajectory.map((x, i) => {return {x: i, y: x.state.angular_velocity[0] - ctbrRateSetpoint(x, rateLimit, 0)}})},
        {label: "pitch-rate error", data: trajectory.map((x, i) => {return {x: i, y: x.state.angular_velocity[1] - ctbrRateSetpoint(x, rateLimit, 1)}})},
        {label: "yaw-rate error", data: trajectory.map((x, i) => {return {x: i, y: x.state.angular_velocity[2] - ctbrRateSetpoint(x, rateLimit, 2)}})},
        {label: "abs roll-rate error", data: trajectory.map((x, i) => {return {x: i, y: Math.abs(x.state.angular_velocity[0] - ctbrRateSetpoint(x, rateLimit, 0))}})},
        {label: "abs pitch-rate error", data: trajectory.map((x, i) => {return {x: i, y: Math.abs(x.state.angular_velocity[1] - ctbrRateSetpoint(x, rateLimit, 1))}})},
        {label: "abs yaw-rate error", data: trajectory.map((x, i) => {return {x: i, y: Math.abs(x.state.angular_velocity[2] - ctbrRateSetpoint(x, rateLimit, 2))}})},
    ], "step", "rad/s", `Trajectory ${trajectory_i}: CTBR body-rate errors`)

    output.scatter([
        {label: "target thrust", data: trajectory.map((x, i) => {return {x: i, y: x.state.attitude_setpoint.thrust_g}})},
        {label: "actual body z specific force", data: trajectory.map((x, i) => {return {x: i, y: specificForceBodyG(x.state)[2]}})},
        {label: "body x specific force", data: trajectory.map((x, i) => {return {x: i, y: specificForceBodyG(x.state)[0]}})},
        {label: "body y specific force", data: trajectory.map((x, i) => {return {x: i, y: specificForceBodyG(x.state)[1]}})},
    ], "step", "g", `Trajectory ${trajectory_i}: thrust/specific-force tracking`)

    output.scatter([
        {label: "reduced attitude error", data: trajectory.map((x, i) => {return {x: i, y: attitudeError(x)}})},
        {label: "abs yaw-rate error", data: trajectory.map((x, i) => {return {x: i, y: Math.abs(x.state.angular_velocity[2] - x.state.attitude_setpoint.yaw_rate)}})},
        {label: "abs thrust error", data: trajectory.map((x, i) => {return {x: i, y: Math.abs(specificForceBodyG(x.state)[2] - x.state.attitude_setpoint.thrust_g)}})},
    ], "step", "value", `Trajectory ${trajectory_i}: tracking errors`)
}
