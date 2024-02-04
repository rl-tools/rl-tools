import {Track} from './track.js';
import {Client} from './client.js';


console.log("Car UI")

window.addEventListener('load', ()=>{
    const canvas = document.getElementById('drawingCanvas');
    const steeringSlider = document.getElementById('steeringSlider');
    const resetTrackButton = document.getElementById('resetTrackButton');
    const saveTrackButton = document.getElementById('saveTrackButton');

    const client = new Client();
    let track = null;
    client.setEnvironmentCallbacks(
        {
            setParametersCallback: (parameters)=>{
                console.log('Parameters:', parameters);
                track = new Track(canvas, parameters);
            },
            setStateCallback: (state)=>{
                track.state = [state.state];
                track.action = [state.action];
            },
            setActionCallback: (data)=>{
                track.action = [data.action];
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
});