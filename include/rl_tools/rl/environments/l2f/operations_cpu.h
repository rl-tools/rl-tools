#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_CPU_H
#include "operations_generic.h"

#include <random>
#include <string>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename PARAM_SPEC, typename DEVICE, typename SPEC>
    std::string json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::l2f::ParametersBase<PARAM_SPEC>::Dynamics& parameters) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;

        std::string json = "{";
        json += "\"mass\": " + std::to_string(parameters.mass) + ", ";
        json += "\"gravity\": [" + std::to_string(parameters.gravity[0]) + ", " + std::to_string(parameters.gravity[1]) + ", " + std::to_string(parameters.gravity[2]) + "], ";
        json += "\"J\": [";
        for (TI i = 0; i < 3; i++) {
            json += "[" + std::to_string(parameters.J[i][0]) + ", " + std::to_string(parameters.J[i][1]) + ", " + std::to_string(parameters.J[i][2]) + "]";
            if (i < 2) {
                json += ", ";
            }
        }
        json += "], ";
        json += "\"J_inv\": [";
        for (TI i = 0; i < 3; i++) {
            json += "[" + std::to_string(parameters.J_inv[i][0]) + ", " + std::to_string(parameters.J_inv[i][1]) + ", " + std::to_string(parameters.J_inv[i][2]) + "]";
            if (i < 2) {
                json += ", ";
            }
        }
        json += "], ";
        json += "\"rotors\": [";
        for (TI i = 0; i < PARAM_SPEC::N; i++) {
            json += "{";
            json += "\"thrust_curve\": [";
            json += std::to_string(parameters.rotor_thrust_coefficients[0]) + ", ";
            json += std::to_string(parameters.rotor_thrust_coefficients[1]) + ", ";
            json += std::to_string(parameters.rotor_thrust_coefficients[2]);
            json += "], ";
            json += "\"torque_constant\": " + std::to_string(parameters.rotor_torque_constant) + ", ";
            json += "\"pose\": {";
            json += "\"position\": [" + std::to_string(parameters.rotor_positions[i][0]) + ", " + std::to_string(parameters.rotor_positions[i][1]) + ", " + std::to_string(parameters.rotor_positions[i][2]) + "], ";
            T qw, qx, qy, qz;
            {
                // thrust direction to quaternion
                T x = parameters.rotor_thrust_directions[i][0];
                T y = parameters.rotor_thrust_directions[i][1];
                T z = parameters.rotor_thrust_directions[i][2];

                T z_unit[3] = {0.0f, 0.0f, 1.0f};

                T cross_x = z_unit[1] * z - z_unit[2] * y;
                T cross_y = z_unit[2] * x - z_unit[0] * z;
                T cross_z = z_unit[0] * y - z_unit[1] * x;

                T dot = z_unit[0] * x + z_unit[1] * y + z_unit[2] * z;

                T angle = math::acos(device.math, dot);

                T cross_magnitude = math::sqrt(device.math, cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
                if (cross_magnitude != 0) {
                    cross_x /= cross_magnitude;
                    cross_y /= cross_magnitude;
                    cross_z /= cross_magnitude;
                }

                T half_angle = angle / 2.0f;
                T sin_half_angle = sin(half_angle);

                qw = cos(half_angle);
                qx = cross_x * sin_half_angle;
                qy = cross_y * sin_half_angle;
                qz = cross_z * sin_half_angle;
            }
            json += "\"orientation\": [" + std::to_string(qw) + ", " + std::to_string(qx) + ", " + std::to_string(qy) + ", " + std::to_string(qz) + "]";
            json += "}, "; // closing pose
            json += "\"torque_direction\": [" + std::to_string(parameters.rotor_torque_directions[i][0]) + ", " + std::to_string(parameters.rotor_torque_directions[i][1]) + ", " + std::to_string(parameters.rotor_torque_directions[i][2]) + "]";
            json += "}"; // closing rotor
            if (i < PARAM_SPEC::N - 1) {
                json += ", ";
            }
        }
        json += "], ";
        json += "\"motor_time_constant\": " + std::to_string(parameters.motor_time_constant) + ", ";
        json += "\"hovering_throttle_relative\": " + std::to_string(parameters.hovering_throttle_relative) + ", ";
        json += "\"action_limit\": {";
        json += "\"min\": " + std::to_string(parameters.action_limit.min) + ", ";
        json += "\"max\": " + std::to_string(parameters.action_limit.max);
        json += "}"; // closing action_limit
        json += "}"; // closing main JSON
        return json;
    }
    template <typename PARAM_SPEC, typename DEVICE, typename SPEC>
    std::string json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::l2f::ParametersBase<PARAM_SPEC>::Integration& parameters) {
        return std::string("{\"dt\": ") + std::to_string(parameters.dt) + "}";
    }
    template <typename PARAM_SPEC, typename DEVICE, typename SPEC>
    std::string json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::l2f::ParametersBase<PARAM_SPEC>::MDP::Initialization& parameters) {
        std::string json_string = "{";
        json_string += "\"guidance\": " + std::to_string(parameters.guidance) + ", ";
        json_string += "\"max_position\": " + std::to_string(parameters.max_position) + ", ";
        json_string += "\"max_angle\": " + std::to_string(parameters.max_angle) + ", ";
        json_string += "\"max_linear_velocity\": " + std::to_string(parameters.max_linear_velocity) + ", ";
        json_string += "\"max_angular_velocity\": " + std::to_string(parameters.max_angular_velocity) + ", ";
        json_string += "\"relative_rpm\": " + std::to_string(parameters.relative_rpm) + ", ";
        json_string += "\"min_rpm\": " + std::to_string(parameters.min_rpm) + ", ";
        json_string += "\"max_rpm\": " + std::to_string(parameters.max_rpm);
        json_string += "}";
        return json_string;
    }
    template <typename PARAM_SPEC, typename DEVICE, typename SPEC>
    std::string json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::l2f::ParametersBase<PARAM_SPEC>::MDP::Termination& parameters) {
        std::string json_string = "{";
        json_string += "\"enabled\": " + std::to_string(parameters.enabled) + ", ";
        json_string += "\"position_threshold\": " + std::to_string(parameters.position_threshold) + ", ";
        json_string += "\"linear_velocity_threshold\": " + std::to_string(parameters.linear_velocity_threshold) + ", ";
        json_string += "\"angular_velocity_threshold\": " + std::to_string(parameters.angular_velocity_threshold) + ", ";
        json_string += "\"position_integral_threshold\": " + std::to_string(parameters.position_integral_threshold) + ", ";
        json_string += "\"orientation_integral_threshold\": " + std::to_string(parameters.orientation_integral_threshold);
        json_string += "}";
        return json_string;
    }
    template <typename PARAM_SPEC, typename DEVICE, typename SPEC>
    std::string json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::l2f::ParametersBase<PARAM_SPEC>::MDP::ObservationNoise& parameters) {
        std::string json_string = "{";
        json_string += "\"position\": " + std::to_string(parameters.position) + ", ";
        json_string += "\"orientation\": " + std::to_string(parameters.orientation) + ", ";
        json_string += "\"linear_velocity\": " + std::to_string(parameters.linear_velocity) + ", ";
        json_string += "\"angular_velocity\": " + std::to_string(parameters.angular_velocity) + ", ";
        json_string += "\"imu_acceleration\": " + std::to_string(parameters.imu_acceleration);
        json_string += "}";
        return json_string;
    }
    template <typename PARAM_SPEC, typename DEVICE, typename SPEC>
    std::string json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::l2f::ParametersBase<PARAM_SPEC>::MDP::ActionNoise& parameters) {
        std::string json_string = "{";
        json_string += "\"normalized_rpm\": " + std::to_string(parameters.normalized_rpm);
        json_string += "}";
        return json_string;
    }
    template <typename PARAM_SPEC, typename DEVICE, typename SPEC>
    std::string json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::l2f::ParametersBase<PARAM_SPEC>::MDP& parameters) {
        std::string json_string = "{";
        json_string += "\"init\": " + json<PARAM_SPEC>(device, env, parameters.init) + ", ";
        json_string += "\"observation_noise\": " + json<PARAM_SPEC>(device, env, parameters.observation_noise) + ", ";
        json_string += "\"action_noise\": " + json<PARAM_SPEC>(device, env, parameters.action_noise) + ", ";
        json_string += "\"termination\": " + json<PARAM_SPEC>(device, env, parameters.termination);
        json_string += "}";
        return json_string;
    }
    template <typename DEVICE, typename SPEC, typename PARAM_SPEC>
    std::string json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, rl::environments::l2f::ParametersBase<PARAM_SPEC>& parameters){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        std::string json = "{";
        json += "\"mass\": " + std::to_string(parameters.dynamics.mass) + ", ";
        json += "\"rotors\": [";
        for (TI i = 0; i < PARAM_SPEC::N; i++){
            json += "{";
            json += "\"thrust_curve\": [";
            json += std::to_string(parameters.dynamics.rotor_thrust_coefficients[0]) + ", ";
            json += std::to_string(parameters.dynamics.rotor_thrust_coefficients[1]) + ", ";
            json += std::to_string(parameters.dynamics.rotor_thrust_coefficients[2]);
            json += "], ";
            json += "\"pose\": {";
            json += "\"position\": [" + std::to_string(parameters.dynamics.rotor_positions[i][0]) + ", " + std::to_string(parameters.dynamics.rotor_positions[i][1]) + ", " + std::to_string(parameters.dynamics.rotor_positions[i][2]) + "], ";
            T qw, qx, qy, qz;
            {
                // thrust direction to quaternion
                T x = parameters.dynamics.rotor_thrust_directions[i][0];
                T y = parameters.dynamics.rotor_thrust_directions[i][1];
                T z = parameters.dynamics.rotor_thrust_directions[i][2];

                T z_unit[3] = {0.0f, 0.0f, 1.0f};

                T cross_x = z_unit[1] * z - z_unit[2] * y;
                T cross_y = z_unit[2] * x - z_unit[0] * z;
                T cross_z = z_unit[0] * y - z_unit[1] * x;

                T dot = z_unit[0] * x + z_unit[1] * y + z_unit[2] * z;

                T angle = math::acos(device.math, dot);

                T cross_magnitude = math::sqrt(device.math, cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
                if (cross_magnitude != 0) {
                    cross_x /= cross_magnitude;
                    cross_y /= cross_magnitude;
                    cross_z /= cross_magnitude;
                }

                T half_angle = angle / 2.0f;
                T sin_half_angle = sin(half_angle);

                qw = cos(half_angle);
                qx = cross_x * sin_half_angle;
                qy = cross_y * sin_half_angle;
                qz = cross_z * sin_half_angle;
            }
            json += "\"orientation\": [" + std::to_string(qw) + ", " + std::to_string(qx) + ", " + std::to_string(qy) + ", " + std::to_string(qz) + "]";
            json += "}"; // closing pose
            json += "}"; // closing rotor
            if (i < PARAM_SPEC::N - 1){
                json += ", ";
            }
        }
        json += "],";
        json += "\"imu\": {\"pose\": {\"position\": [0, 0, 0], \"orientation\": [1, 0, 0, 0]}}";
        std::string dynamics = rl_tools::json<PARAM_SPEC>(device, env, parameters.dynamics);
        json += ", \"dynamics\": " + dynamics;
        json += ", \"integration\": " + rl_tools::json<PARAM_SPEC>(device, env, parameters.integration);
        json += ", \"mdp\": " + rl_tools::json<PARAM_SPEC>(device, env, parameters.mdp);
        json += "}";
        return json;
    }
    template <typename DEVICE, typename SPEC, typename STATE_T, typename STATE_TI>
    std::string json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const rl::environments::l2f::StateBase<STATE_T, STATE_TI>& state){
        std::string json = "{";
        json += "\"position\": [" + std::to_string(state.position[0]) + ", " + std::to_string(state.position[1]) + ", " + std::to_string(state.position[2]) + "], ";
        json += "\"orientation\": [" + std::to_string(state.orientation[0]) + ", " + std::to_string(state.orientation[1]) + ", " + std::to_string(state.orientation[2]) + ", " + std::to_string(state.orientation[3]) + "]";
        json += "}";
        return json;
    }
    template <typename DEVICE, typename SPEC>
    std::string get_ui(DEVICE& device, rl::environments::Multirotor<SPEC>& env){
        // just the body of `function render(ctx, state, action) {` (so that it can be easily processed by `new Function("ctx", "state", "action", body)`
        std::string ui = R"RL_TOOLS_LITERAL(



import * as THREE from "three"
import {OrbitControls} from "three-orbitcontrols"

class CoordinateSystem{
    constructor(origin, length=1, diameter=0.01) {
        this.cs = new THREE.Group()
        const material_red = new THREE.MeshLambertMaterial({color: 0xAA0000})
        const material_green = new THREE.MeshLambertMaterial({color: 0x00AA00})
        const material_blue = new THREE.MeshLambertMaterial({color: 0x0000AA})
        const line = new THREE.BoxGeometry(length, diameter, diameter)
        var x = new THREE.Mesh( line, material_red);
        x.position.set(length/2, 0, 0)
        var y = new THREE.Mesh( line, material_green);
        y.position.set(0, length/2, 0)
        y.rotation.set(0, 0, Math.PI/2)
        var z = new THREE.Mesh( line, material_blue);
        z.position.set(0, 0, length/2)
        z.rotation.set(0, Math.PI/2, 0)
        this.cs.add(x)
        this.cs.add(y)
        this.cs.add(z)
        this.cs.position.set(origin[0], origin[1], origin[2])
    }
    get(){
        return this.cs
    }
}

function norm(a){
    return Math.sqrt(a.map(x => x**2).reduce((a, c) => a + c, 0))
}

function Matrix4FromRotMatTranspose(rotMat){
    const m = new THREE.Matrix4()
    m.set(
        rotMat[0][0], rotMat[1][0], rotMat[2][0], 0,
        rotMat[0][1], rotMat[1][1], rotMat[2][1], 0,
        rotMat[0][2], rotMat[1][2], rotMat[2][2], 0,
        0, 0, 0, 1)
    return m
}

function Matrix4FromRotMat(rotMat){
    const m = new THREE.Matrix4()
    m.set(
        rotMat[0][0], rotMat[0][1], rotMat[0][2], 0,
        rotMat[1][0], rotMat[1][1], rotMat[1][2], 0,
        rotMat[2][0], rotMat[2][1], rotMat[2][2], 0,
        0, 0, 0, 1)
    return m
}




class State{
    constructor(canvas, {devicePixelRatio, showAxes=false}){
        this.canvas = canvas
        this.devicePixelRatio = devicePixelRatio
        this.showAxes = showAxes
        this.cursor_grab = true // Instruct the embedding code to make the cursor a grab cursor
    }
    async initialize(){
        const width = this.canvas.width
        const height = this.canvas.height
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera( 40, width / height, 0.1, 1000 );

        const capture = false;

        this.renderer = new THREE.WebGLRenderer( {canvas: this.canvas, antialias: true, alpha: true, preserveDrawingBuffer: capture} );
        this.renderer.setPixelRatio(this.devicePixelRatio)
        this.renderer.setClearColor(0xffffff, 0);

        this.renderer.setSize(width/this.devicePixelRatio, height/this.devicePixelRatio);


        // canvasContainer.appendChild(this.renderer.domElement );

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);

        this.simulator = new THREE.Group()
        this.simulator.rotation.set(-Math.PI / 2, 0, Math.PI / 2, 'XYZ');

        // const axesHelper = new THREE.AxesHelper(5);
        // this.scene.add(axesHelper)
        this.scene.add(this.simulator)

        var light = new THREE.AmbientLight( 0xffffff,0.5 ); // soft white light
        this.scene.add(light);
        var directionalLight = new THREE.DirectionalLight( 0xffffff, 0.4 )
        directionalLight.position.set(-100, 100, 0)
        directionalLight.target.position.set(0, 0, 0)
        this.scene.add( directionalLight )
        var directionalLight = new THREE.DirectionalLight( 0xffffff, 0.3 )
        directionalLight.position.set(0, 100, 100)
        directionalLight.target.position.set(0, 0, 0)
        this.scene.add( directionalLight )
        var directionalLight = new THREE.DirectionalLight( 0xffffff, 0.2 )
        directionalLight.position.set(0, 100, -100)
        directionalLight.target.position.set(0, 0, 0)
        this.scene.add( directionalLight )

        this.camera.position.set(0.5, 0, 1)
        // this.camera.quaternion.set(-0.14, 0.70, 0.14, 0.68)
        // this.controls.target.set(0.0, 0.0, 0.0)
        // this.controls.minDistance = 1
        // this.controls.minDistance = 5
        // this.controls.update()

        this.camera_set = false
        this.THREE = THREE
    }

}


export class Drone{
  constructor(model, origin, displayIMUCoordinateSystem, displayActions){
    // console.log(model)
    this.origin = origin
    this.model = model
    this.droneFrame = new THREE.Group()
    this.drone = new THREE.Group()
    if(origin){
        this.drone.position.set(...origin)
    }
    // this.drone.add((new CoordinateSystem()).get())
    // this.drone.add((new CoordinateSystem(10 * this.scale, 0.1 * this.scale)).get())
    this.scale = model.mass
    const material = new THREE.MeshLambertMaterial({color: 0xAAAAAA})
    const clockwise_rotor_material = new THREE.MeshLambertMaterial({color: 0x00FF00})
    const counter_clockwise_rotor_material = new THREE.MeshLambertMaterial({color: 0xFF0000})

    const coordinateSystemLength = Math.cbrt(this.scale)
    const coordinateSystemThickness = 0.01 * coordinateSystemLength

    const centerSize = Math.cbrt(this.scale) / 15
    const centerForm = new THREE.BoxGeometry(centerSize, centerSize, centerSize*0.3)
    const center = new THREE.Mesh( centerForm, material);
    // this.drone.quaternion.set(Math.sqrt(0.5), Math.sqrt(0.5), 0,0) // ENUtoNED
    this.imuGroup = new THREE.Group()
    this.imuGroup.position.set(...model.imu.pose.position)
    this.imuGroup.quaternion.set(model.imu.pose.orientation[1], model.imu.pose.orientation[2], model.imu.pose.orientation[3], model.imu.pose.orientation[0])
    if (displayIMUCoordinateSystem) {
      this.imuGroup.add((new CoordinateSystem([0, 0, 0], coordinateSystemLength, coordinateSystemThickness)).get())
    }
    this.drone.add(this.imuGroup)
    this.drone.add(center)

    this.rotors = []

    const averageArmLength = model.rotors.map(rotor => norm(rotor.pose.position)).reduce((a, c) => a + c, 0) / model.rotors.length
    for(const [rotorIndex, rotor] of model.rotors.entries()){
      let rotorCageRadiusFactor = 1
      let rotorCageThicknessFactor = 1
      if (this.envParams != null){
        const rotorParams = this.envParams.rotors[rotorIndex]
        // console.log(this.envParams)
        if (rotorParams.thrust_curve.factors[1].constructor == Object){
          if (Array.isArray(rotorParams.thrust_curve.factors[1].parameters)){
            const mean1 = rotorParams.thrust_curve.factors[1].parameters.reduce((a, c)=>a+c, 0) / rotorParams.thrust_curve.factors[1].parameters.length
            rotorCageThicknessFactor = rotor.thrust_curve.factor_1/mean1
          }
          else{
            if ("upper" in rotorParams.thrust_curve.factors[1]){
              rotorCageThicknessFactor = rotor.thrust_curve.factor_1/((rotorParams.thrust_curve.factors[1].upper - rotorParams.thrust_curve.factors[1].lower)/2 + rotorParams.thrust_curve.factors[1].lower)
            }
          }
        }
        if (rotorParams.thrust_curve.factors[2].constructor == Object){
          if (Array.isArray(rotorParams.thrust_curve.factors[2].parameters)){
            const mean2 = rotorParams.thrust_curve.factors[2].parameters.reduce((a, c)=>a+c, 0) / rotorParams.thrust_curve.factors[2].parameters.length
            rotorCageThicknessFactor = rotor.thrust_curve.factor_2/mean2
          }
          else{
            if ("upper" in rotorParams.thrust_curve.factors[2]){
              rotorCageThicknessFactor = rotor.thrust_curve.factor_2/((rotorParams.thrust_curve.factors[2].upper - rotorParams.thrust_curve.factors[2].lower)/2 + rotorParams.thrust_curve.factors[2].lower)
            }
          }
        }
      }
      const rotorCageRadius =  averageArmLength/3 * Math.sqrt(rotorCageRadiusFactor)
      const rotorCageThickness = averageArmLength/20 * Math.sqrt(rotorCageThicknessFactor)
      const armGroup = new THREE.Group()
      const length = norm(rotor.pose.position)
      const armDiameter = averageArmLength/10
      const armLength = length - rotorCageRadius
      const armForm = new THREE.CylinderGeometry( armDiameter/2, armDiameter/2, armLength, 8 );
      const rot = new THREE.Quaternion(); // Geometry extends in y -> transform y to relative pos
      rot.setFromUnitVectors(new THREE.Vector3(...[0, 1, 0]), (new THREE.Vector3(...rotor.pose.position)).normalize());
      armGroup.quaternion.set(rot.x, rot.y, rot.z, rot.w)

      const arm = new THREE.Mesh(armForm, material)
      arm.position.set(0, armLength/2, 0)
      armGroup.add(arm)

      const rotorGroup = new THREE.Group()
      rotorGroup.position.set(...rotor.pose.position)
      rotorGroup.quaternion.set(rotor.pose.orientation[3], rotor.pose.orientation[0], rotor.pose.orientation[1], rotor.pose.orientation[2])
      // rotorGroup.add((new CoordinateSystem([0, 0, 0], 0.1, 0.01)).get())
      const rotorCageForm = new THREE.TorusGeometry(rotorCageRadius, rotorCageThickness, 16, 32 );
      const cageMaterial = (rotor.spin_orientation_clockwise ? clockwise_rotor_material : counter_clockwise_rotor_material)// new THREE.MeshLambertMaterial({color: 0xAAAAAA})
      const rotorCage = new THREE.Mesh(rotorCageForm, cageMaterial)
      rotorGroup.add(rotorCage)

      const forceArrow = new THREE.ArrowHelper(new THREE.Vector3(0,0,1), new THREE.Vector3(0,0,0 ), 0, 0x000000);
      if(displayActions){
        rotorGroup.add(forceArrow)
      }

      this.drone.add(rotorGroup)
      this.drone.add(armGroup)
      this.droneFrame.add(this.drone)
      this.rotors.push({
        forceArrow,
        rotorCage
      })
    }

  }
  get(){
    return this.droneFrame
  }
  setState(state){
    const mat = Matrix4FromRotMat(state.pose.orientation)
    this.droneFrame.quaternion.setFromRotationMatrix(mat)
    this.droneFrame.position.set(state.pose.position[0] + this.origin[0], state.pose.position[1] + this.origin[1], state.pose.position[2] + this.origin[2])
    const avg_rot_rate = state.rotor_states.reduce((a, c) => a + c["power"], 0)/state.rotor_states.length
    state.rotor_states.map((rotorState, i) => {
      const forceArrow = this.rotors[i].forceArrow
      const rotorCage = this.rotors[i].rotorCage
      const min_rpm = this.model.rotors[i].min_rpm
      const max_rpm = this.model.rotors[i].max_rpm


      const rot_rate = rotorState["power"]
      const force_magnitude = (rot_rate - avg_rot_rate)/max_rpm * 10///1000
      forceArrow.setDirection(new THREE.Vector3(0, 0, rot_rate)) //Math.sign(force_magnitude)))
      forceArrow.setLength(Math.cbrt(this.this.scale)/10) //Math.abs(force_magnitude))
    })
  }

}

export async function init(canvas, options){
    const state = new State(canvas, options)
    await state.initialize()
    return state
}
function clear_episode(ui_state){
    if(ui_state.drone){
      ui_state.simulator.remove(ui_state.drone.get())
      if(ui_state.showAxes){
        ui_state.simulator.remove(ui_state.origin_coordinate_system.get())
      }
    }
    if(ui_state.drones){
      ui_state.drones.map(drone => ui_state.simulator.remove(drone.get()))
      if(ui_state.showAxes){
        ui_state.origin_coordinate_systems.map(cs => ui_state.simulator.remove(cs.get()))
      }
    }
}
function set_camera(ui_state, scale){
    if(!ui_state.camera_set){
      ui_state.camera.position.set(0.5 * scale, 0.5 * scale, 1 * scale)
      ui_state.camera.lookAt(0, 0, 0)
      ui_state.camera_set = true
    }
}
export async function episode_init(ui_state, parameters){
    const camera_distance = (parameters.ui ? parameters.ui.camera_distance || 1 : 1)
    const scale = Math.cbrt(parameters.mass) * 2 * camera_distance
    set_camera(ui_state, scale)
    clear_episode(ui_state)
    ui_state.drone = new Drone(parameters, [0, 0, 0], ui_state.showAxes)
    ui_state.simulator.add(ui_state.drone.get())
    if(ui_state.showAxes){
      ui_state.origin_coordinate_system = new CoordinateSystem([0, 0, 0], 1 * scale, 0.01 * scale)
      ui_state.simulator.add(ui_state.origin_coordinate_system.get())
    }
}

export async function episode_init_multi(ui_state, parameters){
  const grid_distance = 0.0
  const grid_size = Math.ceil(Math.sqrt(parameters.length))
  set_camera(ui_state, (grid_distance > 0 ? grid_distance * grid_size * 2 : Math.cbrt(parameters[0].mass)))
  clear_episode(ui_state)
  ui_state.drones = []
  ui_state.origin_coordinate_systems = []
  parameters.map((parameter, i) => {
    const x = (i % grid_size) * grid_distance
    const y = Math.floor(i / grid_size) * grid_distance
    const drone = new Drone(parameter, [x, y, 0], ui_state.showAxes)
    ui_state.simulator.add(drone.get())
    if(ui_state.showAxes){
      const cs = new CoordinateSystem([x, y, 0], 1, 0.01)
      ui_state.simulator.add(cs.get())
      ui_state.origin_coordinate_systems.push(cs)
    }
    ui_state.drones.push(drone)
  })
}

function update_camera(ui_state){
  const width = ui_state.canvas.width/ui_state.devicePixelRatio
  const height = ui_state.canvas.height/ui_state.devicePixelRatio
  ui_state.camera.aspect =  width / height
  ui_state.camera.updateProjectionMatrix()
  ui_state.renderer.setPixelRatio(ui_state.devicePixelRatio)
  ui_state.renderer.setSize(width, height)

  ui_state.controls.update()
  ui_state.renderer.render(ui_state.scene, ui_state.camera);
}

export async function render(ui_state, parameters, state, action) {
  ui_state.drone.droneFrame.position.set(...state.position)
  ui_state.drone.droneFrame.quaternion.copy(new THREE.Quaternion(state.orientation[1], state.orientation[2], state.orientation[3], state.orientation[0]).normalize())
  update_camera(ui_state)
}

export async function render_multi(ui_state, parameters, steps){
  steps.map((step, i) => {
    ui_state.drones[i].droneFrame.position.set(...step.state.position)
    ui_state.drones[i].droneFrame.quaternion.copy(new THREE.Quaternion(step.state.orientation[1], step.state.orientation[2], step.state.orientation[3], step.state.orientation[0]).normalize())
  })
  update_camera(ui_state)
}


        

        )RL_TOOLS_LITERAL";
        return ui;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif