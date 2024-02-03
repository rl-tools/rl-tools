import {Track} from './track.js';
const canvas = document.getElementById('drawingCanvas');
const steeringSlider = document.getElementById('steeringSlider');



window.addEventListener('load', ()=>{
    const track = new Track(canvas);
    steeringSlider.addEventListener('input', (e)=>{
        track.setSteering(parseFloat(e.target.value)/100);
    });
});