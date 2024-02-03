import {drawCar} from './car.js';

const canvas = document.getElementById('drawingCanvas');
const steeringSlider = document.getElementById('steeringSlider');
const ctx = canvas.getContext('2d');
ctx.imageSmoothingEnabled = false

let pixelSize = 10;
let ratio = 1

const gridWidth = 100;
const gridHeight = 100;
const pixelOverlap = 1
const trackColor = 'black';
const backgroundColor = 'white';
const maxSteeringAngle = Math.PI / 4;


window.addEventListener('load', resizeCanvas);
window.addEventListener('resize', resizeCanvas);


let track= Array(gridHeight).fill().map(() => Array(gridWidth).fill(false));

function drawPixel(gridX, gridY) {
    if (gridX >= 0 && gridX < gridWidth && gridY >= 0 && gridY < gridHeight){
        ctx.fillStyle = 'black';
        ctx.fillRect(gridX * pixelSize - pixelOverlap, gridY * pixelSize - pixelOverlap, pixelSize + 2 * pixelOverlap, pixelSize + 2 * pixelOverlap);
        track[gridY][gridX] = true;
    }
}

function redrawTrack() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for(let y = 0; y < gridHeight; y++){
        for(let x = 0; x < gridWidth; x++){
            if(track[y][x]){
                ctx.fillStyle = trackColor;
                ctx.fillRect(x * pixelSize - pixelOverlap, y * pixelSize - pixelOverlap, pixelSize + 2 * pixelOverlap, pixelSize + 2 * pixelOverlap);
            }
        }
    }
}

ctx.fillStyle = trackColor
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

function startDrawing(e) {
    drawing = true;
    draw(e);
}

function stopDrawing() {
    drawing = false;
    ctx.beginPath();
}

function draw(e) {
    if (!drawing) return;

    const rect = canvas.getBoundingClientRect();
    const scaleFactor = {
        x: canvas.width / rect.width,
        y: canvas.height / rect.height
    };

    const x = Math.floor(((e.clientX - rect.left) * scaleFactor.x / ratio)/pixelSize - 0.5);
    const y = Math.floor(((e.clientY - rect.top) * scaleFactor.y / ratio)/pixelSize - 0.5);

    const brushSize = 3;

    for (let i = -brushSize*2; i < brushSize*2; i++) {
        for (let j = -brushSize*2; j < brushSize*2; j++) {
            if((i*i + j*j) <= brushSize*brushSize){
                drawPixel(x + i, y + j);
            }
        }
    }
}

function render() {
    redrawTrack()
    const steering = steeringSlider.value / 100 * maxSteeringAngle;
    drawCar(canvas, ctx, {lf: 0.5, lr: 0.5}, {x: 0, y: 0, mu: 0, vx: 0, vy: 0, omega: 0}, {throttle: 0, steering: steering}, ratio, pixelSize * 10)
}

function resizeCanvas() {
    const canvasWidth = canvas.parentElement.offsetWidth;
    const canvasHeight = canvas.parentElement.offsetHeight;
    ratio = window.devicePixelRatio || 1;
    canvas.width = canvasWidth * ratio;
    canvas.height = canvasHeight * ratio;
    pixelSize = canvas.width / gridWidth;
    ctx.scale(ratio, ratio);
    canvas.style.width = `${canvas.width / ratio}px`;
    canvas.style.height = `${canvas.height / ratio}px`;
    render()
}

function animate() {
    requestAnimationFrame(animate);
    render();
}

animate()


canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
canvas.addEventListener('mousemove', draw);

const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);