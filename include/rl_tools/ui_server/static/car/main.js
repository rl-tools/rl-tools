// Get a reference to the canvas element and its context
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
ctx.imageSmoothingEnabled = false

let pixelSize = 10;
let ratio = 1

const gridWidth = 100;
const gridHeight = 100;
const pixelOverlap = 1


window.addEventListener('load', resizeCanvas);
window.addEventListener('resize', resizeCanvas);


let track= Array(gridHeight).fill().map(() => Array(gridWidth).fill(0));

function drawPixel(gridX, gridY) {
    if (gridX >= 0 && gridX < gridWidth && gridY >= 0 && gridY < gridHeight){
        ctx.fillStyle = 'black';
        ctx.fillRect(gridX * pixelSize - pixelOverlap, gridY * pixelSize - pixelOverlap, pixelSize + 2 * pixelOverlap, pixelSize + 2 * pixelOverlap);
        track[gridY][gridX] = true;
    }
}

function redrawPixels() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for(let y = 0; y < gridHeight; y++){
        for(let x = 0; x < gridWidth; x++){
            if(track[y][x]){
                ctx.fillStyle = 'black';
                ctx.fillRect(x * pixelSize - pixelOverlap, y * pixelSize - pixelOverlap, pixelSize + 2 * pixelOverlap, pixelSize + 2 * pixelOverlap);
            }
        }
    }
    // Object.keys(drawnPixels).forEach(key => {
    //     const [gridX, gridY] = key.split(',').map(Number);
    //     ctx.fillStyle = 'black';
    //     ctx.fillRect(gridX * pixelSize - pixelOverlap, gridY * pixelSize - pixelOverlap, pixelSize + 2 * pixelOverlap, pixelSize + 2 * pixelOverlap);
    // });
}

ctx.fillStyle = 'black'
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
    redrawPixels();
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
canvas.addEventListener('mousemove', draw);

const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);