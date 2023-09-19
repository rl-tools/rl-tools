import * as THREE from "./lib/three.module.js"
import {OrbitControls} from "./lib/OrbitControls.js"


class Simulator {
    constructor(canvasContainer, parameters) {
        const capture = parameters["capture"] === true
        console.log(`Capture is ${capture}`)
        window.capture = capture
        var width = window.innerWidth
        var height = window.innerHeight
        if (capture){
            width = parameters["width"]
            height = parameters["height"]
        }
        console.log(`Hello ${new Date()}`)

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera( 40, window.innerWidth / window.innerHeight, 0.1, 1000 );

        this.renderer = new THREE.WebGLRenderer( {antialias: true, alpha: true, preserveDrawingBuffer: capture} );
        this.renderer.setPixelRatio(window.devicePixelRatio)
        this.renderer.setClearColor(0xffffff, 0);
        this.renderer.setSize( width, height );


        canvasContainer.appendChild(this.renderer.domElement );

        var controls = new OrbitControls(this.camera, this.renderer.domElement);

        this.simulator = new THREE.Group()
        this.simulator.rotation.set(-Math.PI/2, 0, 0)

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

        // camera.position.x = 0.25;
        // camera.position.y = 0.15;
        // camera.position.z = 0.15;

        let distance_factor = parameters["cameraDistanceToOrigin"]
        this.camera.position.x = -2.5 * distance_factor
        this.camera.position.y = 2.5 * distance_factor
        this.camera.position.z = 2.0 * distance_factor
        this.camera.lookAt(0, 0, 0)

        let animate = () =>{
            requestAnimationFrame(animate);
            controls.update()
            this.renderer.render(this.scene, this.camera );
        }
        animate();
    }
    add(object){
        this.simulator.add(object)
    }
    remove(object){
        this.simulator.remove(object)
    }
}

export {Simulator}