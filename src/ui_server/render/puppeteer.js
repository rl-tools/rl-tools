const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const http = require('http');
const { parse } = require('url');

class Renderer{
    constructor(UI){
        if(!UI) {
            this.ui = fs.readFileSync(path.join(__dirname, "ui.js"), 'utf8');
        }
    }
    async init(){
        this.browser = await puppeteer.launch({headless: "new", args: ['--no-sandbox', '--disable-setuid-sandbox'], timeout: 0});
        this.page = await this.browser.newPage();
        await this.page.setViewport({ width: WIDTH, height: HEIGHT });
        await this.page.goto(`http://localhost:${PORT}/`);
    }
    async render(parameters, state){
        await this.page.evaluate(async (ui, parameters, state) => {
            await window.init(ui);
            await window.render_single_frame(parameters, state)
        }, this.ui, parameters, state);
        const canvas = await this.page.$('canvas');
        const buffer = await canvas.screenshot({ type: 'png', omitBackground: true});

        await this.browser.close();
        return buffer
    }
    async render_trajectory(parameters, trajectory, options = {}) {
        
        await this.page.evaluate(async (ui, options) => {
            await window.init(ui, options);
        }, this.ui, options);
        const frames = await this.page.evaluate(async (parameters, trajectory) => {
            return await window.render_trajectory(parameters, trajectory)
        }, parameters, trajectory);
        const mean_dt = trajectory.reduce((a, c) => a + c.dt, 0) / trajectory.length;
        const fps = Math.round(1 / mean_dt);
        console.log(`FPS: ${fps}`);
        return frames
    }
    async close(){
        await this.browser.close();
    }
}



const server = http.createServer((req, res) => {
    console.log(req.url)
    const parsedUrl = parse(req.url, true);
    const pathname = parsedUrl.pathname;

    if (pathname === '/' || pathname === '/' + "puppeteer.html"){
        res.writeHead(200, { 'Content-Type': 'text/html' });
        const HTML_CONTENT = fs.readFileSync(path.join(__dirname, "puppeteer.html"), 'utf8');
        res.end(HTML_CONTENT);
    } else if (req.url.startsWith('/lib/')) {
        const filePath = path.join(__dirname, req.url);
        fs.readFile(filePath, (err, data) => {
            if(err){
                res.writeHead(404);
                res.end('File not found');
                return;
            }

            const ext = path.extname(filePath).toLowerCase();
            let contentType = 'application/octet-stream';
            if (ext === '.js') contentType = 'application/javascript';
            else if (ext === '.wasm') contentType = 'application/wasm';
            else if (ext === '.json') contentType = 'application/json';

            res.writeHead(200, { 'Content-Type': contentType });
            res.end(data);
        })
    } else if (req.method === 'GET' && req.url === '/data.json') {
        const DATA_CONTENT = fs.readFileSync(path.join(__dirname, "data.json"), 'utf8')
        const DATA = JSON.parse(DATA_CONTENT)
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(DATA));
    } else if (req.method === 'GET' && req.url === '/ui.js') {
        res.writeHead(200, { 'Content-Type': 'application/javascript' });
        const UI = fs.readFileSync(path.join(__dirname, "ui.js"), 'utf8');
        res.end(UI);
    } else if (req.method === 'POST' && req.url === '/render') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });

        req.on('end', async () => {
            try {
                const payload = JSON.parse(body);
                res.writeHead(200, {
                    'Content-Type': 'image/png',
                    'Content-Length': buffer.length
                });
                res.end(buffer);
            } catch (err) {
                console.error('Render error:', err);
                res.writeHead(500);
                res.end('Failed to render canvas.');
            }
        });
    } else {
        res.writeHead(404);
        res.end('Not Found');
    }
});



const DATA_CONTENT = fs.readFileSync(path.join(__dirname, "data.json"), 'utf8')
const DATA = JSON.parse(DATA_CONTENT)

const WIDTH = 2000
const HEIGHT = 2000
const PORT = 3010;
async function main(){
    await new Promise(resolve => server.listen(PORT, resolve));
    const renderer = new Renderer();
    await renderer.init();
    // const buffer = await renderer.render(DATA[0].parameters, DATA[0].trajectory[0])
    frames = await renderer.render_trajectory(DATA[0].parameters, DATA[0].trajectory, {frame_counter: false})
    await renderer.close();

    const outputDir = path.join(__dirname, 'canvas_frames');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }
    const file_promises = frames.map((frame, i) => {
        const buffer = Buffer.from(frame, 'base64');
        const outputPath = path.join(outputDir, `frame_${i.toString().padStart(5, '0')}.png`);
        console.log(`Saved frame ${i} to ${outputPath}`);
        return fs.promises.writeFile(outputPath, buffer);
    })
    await Promise.all(file_promises);
    // const outputPath = path.join(__dirname, 'output.png');
    // fs.writeFileSync(outputPath, buffer);
    // console.log(`Screenshot saved to ${outputPath}`);
    await new Promise(resolve => setTimeout(resolve, 100000000));
    server.close(() => console.log('Server closed.'));
}

main()

