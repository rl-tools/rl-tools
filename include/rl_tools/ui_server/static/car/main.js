// Get a reference to the canvas element and its context
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
ctx.imageSmoothingEnabled = false

let pixelSize = 10;
let ratio = 1
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
    // Redraw your canvas here if needed
}

const gridWidth = 100;
const gridHeight = 100;


window.addEventListener('load', resizeCanvas);
window.addEventListener('resize', resizeCanvas);


let drawnPixels = {}; // Object to store drawn pixel positions

function drawPixel(gridX, gridY) {
    if (!drawnPixels[`${gridX},${gridY}`]) {
        ctx.fillStyle = 'black';
        const overlap = 1
        // ctx.fillRect(gridX * pixelSize, gridY * pixelSize, pixelSize, pixelSize);
        ctx.fillRect(gridX * pixelSize - overlap, gridY * pixelSize - overlap, pixelSize + 2 * overlap, pixelSize + 2 * overlap);
        // Store the pixel
        drawnPixels[`${gridX},${gridY}`] = true;
    }
}

function redrawPixels() {
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Redraw all pixels from the stored positions
    Object.keys(drawnPixels).forEach(key => {
        const [gridX, gridY] = key.split(',').map(Number);
        ctx.fillStyle = 'black';
        ctx.fillRect(gridX * pixelSize, gridY * pixelSize, pixelSize, pixelSize);
    });
}

ctx.fillStyle = 'black'
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

function startDrawing(e) {
    drawing = true;
    draw(e); // Draw immediately in case of a click without a move
}

function stopDrawing() {
    drawing = false;
    ctx.beginPath(); // Reset the path
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

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
canvas.addEventListener('mousemove', draw);

const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);