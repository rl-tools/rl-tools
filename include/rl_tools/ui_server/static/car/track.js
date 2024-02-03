import {drawCar} from "./car.js";


export class Track{
    constructor(canvas){
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.ctx.imageSmoothingEnabled = false
        this.pixelSize = 10;
        this.ratio = 1

        this.pixelWidth = 0.02; // in meters
        this.pixelToMeter = 1 / this.pixelWidth;

        this.gridWidth = 100;
        this.gridHeight = 100;
        this.pixelOverlap = 1
        this.trackColor = 'black';
        this.backgroundColor = 'white';
        this.maxSteeringAngle = Math.PI / 4;
        this.carParameters = {lf: 0.029, lr: 0.033};

        // state
        this.track= Array(this.gridHeight).fill().map(() => Array(this.gridWidth).fill(false));
        this.drawing = false;
        this.state = {x: 0, y: 0, mu: 0, vx: 0, vy: 0, omega: 0};
        this.action = {throttle: 0, steering: 0};

        // hooks
        window.addEventListener('load', ()=>this.resizeCanvas());
        window.addEventListener('resize', ()=>this.resizeCanvas());
        canvas.addEventListener('mousedown', (e)=>this.startDrawing(e));
        canvas.addEventListener('mouseup', ()=>this.stopDrawing());
        canvas.addEventListener('mouseout', ()=>this.stopDrawing());
        canvas.addEventListener('mousemove', (e)=>this.draw(e));
        this.animate()
    }

    drawPixel(gridX, gridY) {
        if (gridX >= 0 && gridX < this.gridWidth && gridY >= 0 && gridY < this.gridHeight){
            this.ctx.fillStyle = 'black';
            this.ctx.fillRect(gridX * this.pixelSize - this.pixelOverlap, gridY * this.pixelSize - this.pixelOverlap, this.pixelSize + 2 * this.pixelOverlap, this.pixelSize + 2 * this.pixelOverlap);
            this.track[gridY][gridX] = true;
        }
    }

    redrawTrack() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        for(let y = 0; y < this.gridHeight; y++){
            for(let x = 0; x < this.gridWidth; x++){
                if(this.track[y][x]){
                    this.ctx.fillStyle = this.trackColor;
                    this.ctx.fillRect(x * this.pixelSize - this.pixelOverlap, y * this.pixelSize - this.pixelOverlap, this.pixelSize + 2 * this.pixelOverlap, this.pixelSize + 2 * this.pixelOverlap);
                }
            }
        }
    }


    startDrawing(e){
        this.drawing = true;
        this.draw(e);
    }

    stopDrawing() {
        this.drawing = false;
        this.ctx.beginPath();
    }

    draw(e) {
        if (!this.drawing) return;

        const rect = this.canvas.getBoundingClientRect();
        const scaleFactor = {
            x: this.canvas.width / rect.width,
            y: this.canvas.height / rect.height
        };

        const x = Math.floor(((e.clientX - rect.left) * scaleFactor.x / ratio)/this.pixelSize - 0.5);
        const y = Math.floor(((e.clientY - rect.top) * scaleFactor.y / ratio)/this.pixelSize - 0.5);

        const brushSize = 3;

        for (let i = -brushSize*2; i < brushSize*2; i++) {
            for (let j = -brushSize*2; j < brushSize*2; j++) {
                if((i*i + j*j) <= brushSize*brushSize){
                    this.drawPixel(x + i, y + j);
                }
            }
        }
    }

    render() {
        this.redrawTrack()
        drawCar(this.canvas, this.ctx, this.carParameters, this.state, this.action, this.ratio, this.pixelSize * this.pixelToMeter);
    }

    resizeCanvas() {
        const canvasWidth = this.canvas.parentElement.offsetWidth;
        const canvasHeight = this.canvas.parentElement.offsetHeight;
        this.ratio = window.devicePixelRatio || 1;
        this.canvas.width = canvasWidth * this.ratio;
        this.canvas.height = canvasHeight * this.ratio;
        const pixelSize = this.canvas.width / this.gridWidth;
        this.ctx.scale(this.ratio, this.ratio);
        this.canvas.style.width = `${this.canvas.width / this.ratio}px`;
        this.canvas.style.height = `${this.canvas.height / this.ratio}px`;
        this.render()
    }
    setSteering(steering) {
        this.action.steering = steering * this.maxSteeringAngle;
    }

    animate() {
        requestAnimationFrame(()=>this.animate());
        this.render();
    }
}








