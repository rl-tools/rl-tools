import {Track} from './track.js';
import {Client} from './client.js';
const canvas = document.getElementById('drawingCanvas');
const steeringSlider = document.getElementById('steeringSlider');


console.log("Car UI")

window.addEventListener('load', ()=>{
    const client = new Client();
    let track = null;
    client.setEnvironmentCallbacks(
        {
            setParametersCallback: (parameters)=>{
                console.log('Parameters:', parameters);
                track = new Track(canvas, parameters);
            },
            setStateCallback: (state)=>{
                track.state = state;
            },
            setActionCallback: (action)=>{
                track.action = action;
            }
        }
    )
    steeringSlider.addEventListener('input', (e)=>{
        track.setSteering(parseFloat(e.target.value)/100);
    });
});