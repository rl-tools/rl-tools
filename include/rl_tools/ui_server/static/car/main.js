import {Track} from './track.js';
// import {Client} from './client.js';
import {Client} from './client_wasm.js';


console.log("Car UI")


const keyThrottleValue = 0.5


window.addEventListener('load', ()=>{
    const canvas = document.getElementById('drawingCanvas');
    const steeringSlider = document.getElementById('steeringSlider');
    const resetTrackButton = document.getElementById('resetTrackButton');
    const saveTrackButton = document.getElementById('saveTrackButton');
    const playButton = document.getElementById('playButton');
    const trainButton = document.getElementById('trainButton');


    // const client = new Client();
    const client = new Client();
    let track = null;
    client.setEnvironmentCallbacks(
        {
            setParametersCallback: (parameters)=>{
                console.log('Parameters:', parameters);
                track = new Track(canvas, parameters);
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
    steeringSlider.addEventListener('input', (e)=>{
        track.setSteering(parseFloat(e.target.value)/100);
    });
    resetTrackButton.addEventListener('click', ()=>{
        track.reset();
    });
    saveTrackButton.addEventListener('click', ()=>{
        track.disable_drawing()
        client.sendMessage("setTrack", track.track);
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
    });
    trainButton.addEventListener('click', ()=>{
        mode_interactive = false
        client.sendMessage("startTraining", null);
    });
});