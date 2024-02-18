import {Track} from './track.js';
import {Client as ClientWS} from './client.js';
import {Client as ClientWASM} from './client_wasm.js';


console.log("Car UI")

// const forceWASM = false //true
const forceWASM = true

const keyThrottleValue = 0.5
const orientationGainSteering = 3
const orientationGainThrottle = 3
let first_orientation = null
let playbackSpeed = 100

function createAndAppendElement(tag, attributes, parent) {
    const element = document.createElement(tag);
    Object.keys(attributes).forEach(attr => element[attr] = attributes[attr]);
    parent.appendChild(element);
    return element;
}

function renderDOM(parentElement) {
    const messageContainer = createAndAppendElement('div', {id: 'messageContainer'}, parentElement);
    createAndAppendElement('div', {id: 'loadingLabel', textContent: 'Loading...'}, messageContainer);
    createAndAppendElement('div', {id: 'drawLabel', textContent: 'Draw track onto the canvas!', style: 'display: none;'}, messageContainer);
    createAndAppendElement('div', {id: 'playLabel', textContent: 'Use arrow keys to drive the car!', style: 'display: none;'}, messageContainer);
    createAndAppendElement('div', {id: 'trainLabel', textContent: 'Training...', style: 'display: none;'}, messageContainer);

    const controlContainer = createAndAppendElement('div', {id: 'controlContainer'}, parentElement);
    createAndAppendElement('input', {type: 'button', id: 'resetTrackButton', value: 'Reset Track', style: 'display: none;'}, controlContainer);
    createAndAppendElement('input', {type: 'button', id: 'saveTrackButton', value: 'Save Track', style: 'display: none;'}, controlContainer);
    createAndAppendElement('input', {type: 'button', id: 'playButton', value: 'Play Interactively', style: 'display: none;'}, controlContainer);
    createAndAppendElement('input', {type: 'button', id: 'trainButton', value: 'Train', style: 'display: none;'}, controlContainer);
    const label = createAndAppendElement('label', {id: 'playbackSpeedCheckboxLabel', style: 'display: none;'}, controlContainer);
    createAndAppendElement('input', {type: 'checkbox', id: 'playbackSpeedCheckbox'}, label);
    label.appendChild(document.createTextNode(' Realtime'));

    const canvasContainer = createAndAppendElement('div', {id: 'canvasContainer', style: 'display: none;'}, parentElement);
    const aspectRatioContainerContainerContainer = createAndAppendElement('div', {id: 'aspectRatioContainerContainerContainer'}, canvasContainer);
    const aspectRatioContainerContainer = createAndAppendElement('div', {id: 'aspectRatioContainerContainer'}, aspectRatioContainerContainerContainer);
    const aspectRatioContainer = createAndAppendElement('div', {id: 'aspectRatioContainer'}, aspectRatioContainerContainer);
    const canvasContainerInner = createAndAppendElement('div', {id: 'canvasContainerInner'}, aspectRatioContainer);
    createAndAppendElement('canvas', {id: 'drawingCanvas', width: '100', height: '100'}, canvasContainerInner);
}
let async_main = async () => {
    const canvas = document.getElementById('drawingCanvas');
    const canvasContainer = document.getElementById('canvasContainer');
    const resetTrackButton = document.getElementById('resetTrackButton');
    const saveTrackButton = document.getElementById('saveTrackButton');
    const playButton = document.getElementById('playButton');
    const trainButton = document.getElementById('trainButton');
    const drawLabel = document.getElementById('drawLabel');
    const loadingLabel = document.getElementById('loadingLabel');
    const trainLabel = document.getElementById('trainLabel');
    const playLabel = document.getElementById('playLabel');
    const playbackSpeedCheckbox = document.getElementById('playbackSpeedCheckbox');
    const playbackSpeedCheckboxLabel = document.getElementById('playbackSpeedCheckboxLabel');

    let response = await fetch('./scenario');
    let Client = ClientWASM;
    if(response.status == 200 && !forceWASM){
        Client = ClientWS;
    }

    // const client = new Client();
    const client = new Client();
    let track = null;
    client.setEnvironmentCallbacks(
        {
            setParametersCallback: (parameters)=>{
                console.log('Parameters:', parameters);
                track = new Track(canvas, parameters);
                loadingLabel.style.display = "none";
                drawLabel.style.display = "block";
                resetTrackButton.style.display = "block";
                saveTrackButton.style.display = "block";
                canvasContainer.style.display = "block";
                track.resizeCanvas()
            },
            setStateCallback: (state)=>{
                if(track){
                    track.state = [state.state];
                    track.action = [state.action];
                }
            },
            setActionCallback: (data)=>{
                if(track){
                    track.action = [data.action];
                }
            },
        }
    )
    resetTrackButton.addEventListener('click', ()=>{
        track.reset();
    });
    saveTrackButton.addEventListener('click', ()=>{
        track.disable_drawing()
        client.sendMessage("setTrack", track.track);
        resetTrackButton.style.display = "none";
        saveTrackButton.style.display = "none";
        playButton.style.display = "block";
        trainButton.style.display = "block";
        drawLabel.style.display = "none";
    });

    let mode_interactive = false;

    const input_action = {
        "throttle": 0,
        "steering": 0
    }
    document.addEventListener('keydown', function(event) {
        if(mode_interactive){
            let update = false;
            switch(event.key) {
                case "ArrowUp":
                    input_action["throttle"] = keyThrottleValue;
                    update = true;
                    break;
                case "ArrowDown":
                    input_action["throttle"] = -keyThrottleValue;
                    update = true;
                    break;
                case "ArrowLeft":
                    input_action["steering"] = 1;
                    update = true;
                    break;
                case "ArrowRight":
                    input_action["steering"] = -1;
                    update = true;
                    break;
                default:
                    break;
            }
            if(update){
                event.preventDefault();
                client.sendMessage("setAction", [input_action["throttle"], input_action["steering"]]);
            }
        }
    });
    document.addEventListener('keyup', function(event) {
        if(mode_interactive){
            let update = false;
            switch(event.key) {
                case "ArrowUp":
                    input_action["throttle"] = 0;
                    update = true;
                    break;
                case "ArrowDown":
                    input_action["throttle"] = 0;
                    update = true;
                    break;
                case "ArrowLeft":
                    input_action["steering"] = 0;
                    update = true;
                    break;
                case "ArrowRight":
                    input_action["steering"] = 0;
                    update = true;
                    break;
                default:
                    break;
            }
            if(update){
                client.sendMessage("setAction", [input_action["throttle"], input_action["steering"]]);
            }
        }
    });
    playButton.addEventListener('click', ()=>{
        mode_interactive = true
        client.sendMessage("setAction", [0, 0]);
        playButton.style.display = "none";
        playLabel.style.display = "block";
        if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
            DeviceOrientationEvent.requestPermission()
                .then(permissionState => {
                    if (permissionState === 'granted') {
                        // document.getElementById('status').textContent = 'Permission granted';
                    }
                })
                .catch(console.error); // Handle errors
        } else {
            // Handle regular non-iOS 13+ devices
            // window.addEventListener('deviceorientation', handleOrientationEvent);
            // document.getElementById('status').textContent = 'Permission API not required';
        }
    });
    trainButton.addEventListener('click', ()=>{
        mode_interactive = false
        client.sendMessage("startTraining", null);
        playButton.style.display = "none";
        trainButton.style.display = "none";
        trainLabel.style.display = "block";
        playLabel.style.display = "none";
        playbackSpeedCheckboxLabel.style.display = "block";
        client.sendMessage("setPlaybackSpeed", playbackSpeedCheckbox.checked ? 1 : playbackSpeed);
    });
    playbackSpeedCheckbox.addEventListener('change', ()=>{
        client.sendMessage("setPlaybackSpeed", playbackSpeedCheckbox.checked ? 1 : playbackSpeed);
    });
    window.addEventListener('deviceorientation', function(event) {
        if(mode_interactive){
            if(!first_orientation){
                first_orientation = event;
            }
            input_action["steering"] = Math.max(-1, Math.min(1, (-event.gamma + first_orientation.gamma)/90 * orientationGainSteering));
            input_action["throttle"] = Math.max(-1, Math.min(1, (-event.beta + first_orientation.beta)/90 * orientationGainThrottle));
            client.sendMessage("setAction", [input_action["throttle"], input_action["steering"]]);
        }
    });
}

window.addEventListener('DOMContentLoaded', ()=>{
    renderDOM(document.body);
})
window.addEventListener('load', ()=>{
    async_main();
})
