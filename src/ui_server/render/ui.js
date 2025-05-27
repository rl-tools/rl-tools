import * as THREE from "three"
import {OrbitControls} from "three-orbitcontrols"

export class CoordinateSystem{
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
    constructor(canvas, {devicePixelRatio, showAxes=false, capture=false, camera_position=[0.5, 0.5, 1]}){
        this.canvas = canvas
        this.devicePixelRatio = devicePixelRatio
        this.showAxes = showAxes
        this.cursor_grab = false // Instruct the embedding code to make the cursor a grab cursor
        this.render_tick = 0
        this.capture = capture
        this.camera_position = camera_position
    }
    async initialize(){
        const width = this.canvas.width
        const height = this.canvas.height
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera( 40, width / height, 0.1, 1000 );
        this.scene.add(this.camera);  

        this.renderer = new THREE.WebGLRenderer( {canvas: this.canvas, antialias: true, alpha: true, preserveDrawingBuffer: this.capture} );
        this.renderer.setPixelRatio(this.devicePixelRatio)
        this.renderer.setClearColor(0xffffff, 0);

        this.renderer.setSize(width/this.devicePixelRatio, height/this.devicePixelRatio);


        // canvasContainer.appendChild(this.renderer.domElement );

        // this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        // this.controls.enabled = false;
        // window.addEventListener('keydown', (event) => {
        //     if (event.key === 'Alt') {
        //         this.controls.enabled = true;
        //         this.canvas.style.cursor = "grab"
        //     }
        // });

        // window.addEventListener('keyup', (event) => {
        //     if (event.key === 'Alt') {
        //         this.controls.enabled = false;
        //         this.canvas.style.cursor = "default"
        //     }
        // });
        this.canvas.title = "Alt+Drag to rotate the camera. Alt+CTRL+Drag to move the camera."

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

        // this.camera.position.set(...this.camera_position)
        // this.camera.quaternion.set(-0.14, 0.70, 0.14, 0.68)
        // this.controls.target.set(0.0, 0.0, 0.0)
        // this.controls.minDistance = 1
        // this.controls.minDistance = 5
        // this.controls.update()

        this.camera_set = false
        this.THREE = THREE
    }

}


function thrust_direction_to_quaternion(thrust_direction){
    const x = thrust_direction[0];
    const y = thrust_direction[1];
    const z = thrust_direction[2];

    const z_unit = [0.0, 0.0, 1.0];

    let cross_x = z_unit[1] * z - z_unit[2] * y;
    let cross_y = z_unit[2] * x - z_unit[0] * z;
    let cross_z = z_unit[0] * y - z_unit[1] * x;

    const dot = z_unit[0] * x + z_unit[1] * y + z_unit[2] * z;

    const angle = Math.acos(dot);

    const cross_magnitude = Math.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
    if (cross_magnitude != 0) {
        cross_x /= cross_magnitude;
        cross_y /= cross_magnitude;
        cross_z /= cross_magnitude;
    }

    const half_angle = angle / 2.0;
    const sin_half_angle = Math.sin(half_angle);

    const qw = Math.cos(half_angle);
    const qx = cross_x * sin_half_angle;
    const qy = cross_y * sin_half_angle;
    const qz = cross_z * sin_half_angle;
    return [qw, qx, qy, qz];
}


export class Drone{
    constructor(parameters, origin, displayIMUCoordinateSystem, displayActions){
        const url = window.location.href;
        const urlObj = new URL(url);
        const params = new URLSearchParams(urlObj.search);
        if(params.has('L2FDisplayActions') === true){
            displayActions = params.get('L2FDisplayActions') === "true";
        }

        // console.log(model)
        this.origin = origin
        this.parameters = parameters
        this.droneFrame = new THREE.Group()
        this.drone = new THREE.Group()
        if(origin){
            this.drone.position.set(...origin)
        }
        // this.drone.add((new CoordinateSystem()).get())
        // this.drone.add((new CoordinateSystem(10 * this.scale, 0.1 * this.scale)).get())
        this.scale = parameters.dynamics.mass
        const material = new THREE.MeshLambertMaterial({color: 0xAAAAAA})
        const clockwise_rotor_material = new THREE.MeshLambertMaterial({color: 0x00FF00})
        const counter_clockwise_rotor_material = new THREE.MeshLambertMaterial({color: 0xFF0000})

        const coordinateSystemLength = Math.cbrt(this.scale)
        const coordinateSystemThickness = 0.01 * coordinateSystemLength

        const centerSize = Math.cbrt(this.scale) / 15
        const centerForm = new THREE.BoxGeometry(centerSize, centerSize, centerSize*0.3)
        const center = new THREE.Mesh( centerForm, material);

        this.parameters.dynamics["imu_position"] = [0, 0, 0]
        this.parameters.dynamics["imu_orientation"] = [1, 0, 0, 0]

        this.imuGroup = new THREE.Group()
        this.imuGroup.position.set(...this.parameters.dynamics.imu_position)
        this.imuGroup.quaternion.set(this.parameters.dynamics.imu_orientation[1], this.parameters.dynamics.imu_orientation[2], this.parameters.dynamics.imu_orientation[3], this.parameters.dynamics.imu_orientation[0])
        if (displayIMUCoordinateSystem) {
            this.imuGroup.add((new CoordinateSystem([0, 0, 0], coordinateSystemLength, coordinateSystemThickness)).get())
        }
        this.drone.add(this.imuGroup)
        this.drone.add(center)

        this.rotors = []

        const averageArmLength = this.parameters.dynamics.rotor_positions.map(position => norm(position)).reduce((a, c) => a + c, 0) / this.parameters.dynamics.rotor_positions.length
        for(const [rotorIndex, rotor_position] of this.parameters.dynamics.rotor_positions.entries()){
            let rotorCageRadiusFactor = 1
            let rotorCageThicknessFactor = 1
            const rotorCageRadius =  averageArmLength/3 * Math.sqrt(rotorCageRadiusFactor)
            const rotorCageThickness = averageArmLength/20 * Math.sqrt(rotorCageThicknessFactor)
            const armGroup = new THREE.Group()
            const length = norm(rotor_position)
            const armDiameter = averageArmLength/10
            const armLength = length - rotorCageRadius
            const armForm = new THREE.CylinderGeometry( armDiameter/2, armDiameter/2, armLength, 8 );
            const rot = new THREE.Quaternion(); // Geometry extends in y -> transform y to relative pos
            rot.setFromUnitVectors(new THREE.Vector3(...[0, 1, 0]), (new THREE.Vector3(...rotor_position)).normalize());
            armGroup.quaternion.set(rot.x, rot.y, rot.z, rot.w)

            const arm = new THREE.Mesh(armForm, material)
            arm.position.set(0, armLength/2, 0)
            armGroup.add(arm)

            const rotorGroup = new THREE.Group()
            rotorGroup.position.set(...rotor_position)

            const thrust_orientation = thrust_direction_to_quaternion(this.parameters.dynamics.rotor_thrust_directions[rotorIndex])
            rotorGroup.quaternion.set(thrust_orientation[3], thrust_orientation[0], thrust_orientation[1], thrust_orientation[2])
            // rotorGroup.add((new CoordinateSystem([0, 0, 0], 0.1, 0.01)).get())
            const rotorCageForm = new THREE.TorusGeometry(rotorCageRadius, rotorCageThickness, 16, 32 );
            const cageMaterial = (this.parameters.dynamics.rotor_thrust_directions[rotorIndex][2] < 0 ? clockwise_rotor_material : counter_clockwise_rotor_material)// new THREE.MeshLambertMaterial({color: 0xAAAAAA})
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
    // setState(state){
    //   const mat = Matrix4FromRotMat(state.orientation)
    //   this.droneFrame.quaternion.setFromRotationMatrix(mat)
    //   this.droneFrame.position.set(state.pose.position[0] + this.origin[0], state.pose.position[1] + this.origin[1], state.pose.position[2] + this.origin[2])
    //   const avg_rot_rate = state.rotor_states.reduce((a, c) => a + c["power"], 0)/state.rotor_states.length
    //   state.rotor_states.map((rotorState, i) => {
    //     const forceArrow = this.rotors[i].forceArrow
    //     const rotorCage = this.rotors[i].rotorCage
    //     const min_rpm = this.model.rotors[i].min_rpm
    //     const max_rpm = this.model.rotors[i].max_rpm


    //     const rot_rate = rotorState["power"]
    //     const force_magnitude = (rot_rate - avg_rot_rate)/max_rpm * 10///1000
    //     forceArrow.setDirection(new THREE.Vector3(0, 0, rot_rate)) //Math.sign(force_magnitude)))
    //     forceArrow.setLength(Math.cbrt(this.this.scale)/10) //Math.abs(force_magnitude))
    //   })
    // }

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
        ui_state.camera.position.set(ui_state.camera_position[0] * scale, ui_state.camera_position[1] * scale, ui_state.camera_position[2] * scale)
        ui_state.camera.lookAt(0, 0, 0)
        ui_state.camera_set = true
    }
}
export async function episode_init(ui_state, parameters){
    const camera_distance = (parameters.ui ? parameters.ui.camera_distance || 1 : 1)
    const scale = Math.cbrt(parameters.dynamics.mass) * 2 * camera_distance
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
    set_camera(ui_state, (grid_distance > 0 ? grid_distance * grid_size * 2 : Math.cbrt(parameters[0].dynamics.mass)))
    clear_episode(ui_state)
    ui_state.drones = []
    if(!ui_state.showAxes && ui_state.origin_coordinate_systems){
        ui_state.origin_coordinate_systems.forEach(cs => {
            ui_state.simulator.remove(cs.get())
        })
    }
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
    if(ui_state.render_tick % 10 == 0){
        const width = ui_state.canvas.width/ui_state.devicePixelRatio
        const height = ui_state.canvas.height/ui_state.devicePixelRatio
        ui_state.camera.aspect =  width / height
        ui_state.camera.updateProjectionMatrix()
        ui_state.renderer.setPixelRatio(ui_state.devicePixelRatio)
        ui_state.renderer.setSize(width, height)
    }

    // ui_state.controls.update()
    ui_state.renderer.render(ui_state.scene, ui_state.camera);

    ui_state.render_tick += 1
}

function clip_position(scale, position){
    const extent = Math.cbrt(scale) * 300 // to maybe prevent threejs from exploding
    const max_position = extent
    const min_position = -extent
    return position.map((p) => {
        if(p > max_position){
            return max_position
        }
        else if(p < min_position){
            return min_position
        }
        else{
            return p
        }
    })
}

export async function render(ui_state, parameters, state, action) {
    ui_state.drone.droneFrame.position.set(...clip_position(parameters.dynamics.mass, state.position))
    ui_state.drone.droneFrame.quaternion.copy(new THREE.Quaternion(state.orientation[1], state.orientation[2], state.orientation[3], state.orientation[0]).normalize())
    update_camera(ui_state)
}

export async function render_multi(ui_state, parameters, states, actions){
    states.map((state, i) => {
        const action = actions[i]
        const current_parameters = parameters[i]
        ui_state.drones[i].droneFrame.position.set(...clip_position(current_parameters.dynamics.mass, state.position))
        ui_state.drones[i].droneFrame.quaternion.copy(new THREE.Quaternion(state.orientation[1], state.orientation[2], state.orientation[3], state.orientation[0]).normalize())
        for(let j = 0; j < 4; j++){
            const forceArrow = ui_state.drones[i].rotors[j].forceArrow
            const force_magnitude = action[j]
            forceArrow.setDirection(new THREE.Vector3(0, 0, force_magnitude))
            forceArrow.setLength(Math.cbrt(ui_state.drones[i].scale)/10)
        }
    })
    update_camera(ui_state)
}
