const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const http = require('http');
const { parse } = require('url');


const WIDTH = 2000
const HEIGHT = 2000
const PORT = 3010;
const OUTPUT_DIR_NAME = 'canvas_frames';
const RECORDING_DURATION_MS = 10000;
const TARGET_FPS = 60;


const DATA_CONTENT = fs.readFileSync(path.join(__dirname, "data.json"), 'utf8')
const DATA = JSON.parse(DATA_CONTENT)
const UI = fs.readFileSync(path.join(__dirname, "ui.js"), 'utf8');


async function recordCanvasAnimation() {
    const outputDir = path.join(__dirname, OUTPUT_DIR_NAME);
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    } else {
        console.log(`Cleaning up old frames in ${outputDir}...`);
        const files = await fs.promises.readdir(outputDir);
        for (const file of files) {
            if (file.startsWith('frame-') && file.endsWith('.png')) {
                await fs.promises.unlink(path.join(outputDir, file));
            }
        }
    }


    console.log('Launching Puppeteer...');
    const browser = await puppeteer.launch();
    const page    = await browser.newPage();
    await page.setViewport({ width: WIDTH, height: HEIGHT});
    
    const client = await page.target().createCDPSession();
    
    let frameCounter  = 0;
    const frameWrites = [];
    
    client.on('Page.screencastFrame', async ({ data, sessionId }) => {
        const framePath = path.join(
            outputDir,
            `frame-${String(frameCounter).padStart(5, '0')}.png`
        );
    
        const buffer = Buffer.from(data, 'base64');
        frameWrites.push(fs.promises.writeFile(framePath, buffer));
        frameCounter++;
        process.stdout.write(`Captured frame: ${frameCounter}\r`);
        await client.send('Page.screencastFrameAck', { sessionId });
    });
    
    console.log(`Navigating to http://localhost:${PORT}/`);
    await page.goto(`http://localhost:${PORT}/`, { waitUntil: 'networkidle0' });
    
    console.log('Starting screencast...');
    await client.send('Page.startScreencast', {
        format: 'png',
        quality: 100,
        everyNthFrame: 1
    });
    console.log(`Recording for ${RECORDING_DURATION_MS / 1000} seconds...`);
    await new Promise(r => setTimeout(r, RECORDING_DURATION_MS));
    
    console.log('\nStopping screencast...');
    await client.send('Page.stopScreencast');
    
    console.log('Waiting for all frames to be saved…');
    await Promise.all(frameWrites);
    console.log(`${frameCounter} frames saved in: ${outputDir}`);
    
    console.log('Closing browser and server…');
    await browser.close();
    server.close(() => console.log('Server closed.'));
    
    if (frameCounter > 0) {
        console.log('\n--- Video Creation with FFmpeg ---');
        console.log('To create a video from the captured frames, run a command like this in your terminal:');
        const ffmpegCommand = `ffmpeg -framerate ${TARGET_FPS} -i "${path.join(outputDir, 'frame-%05d.png')}" -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow output.mp4`;
        console.log(ffmpegCommand);
        console.log(`\nNotes on FFmpeg command:
    - Ensure FFmpeg is installed and in your system's PATH.
    - Adjust '-framerate ${TARGET_FPS}' if your animation's natural framerate is different or you desire a different output speed.
    - '-crf 18' controls quality (lower is better, 18 is visually lossless for many cases).
    - '-preset slow' provides better compression (options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow).
    - 'output.mp4' is the output file name.`);
    } else {
        console.log('No frames were captured.');
    }
}

// recordCanvasAnimation().catch(error => {
//     console.error('An error occurred during the recording process:', error);
//     process.exit(1);
// });

async function render(parameters, state){
    const browser = await puppeteer.launch({headless: "new"});
    const page = await browser.newPage();
    await page.setViewport({ width: WIDTH, height: HEIGHT });
    await page.goto(`http://localhost:${PORT}/`);

    await page.evaluate(async (UI, parameters, state) => {
        await window.init(UI);
        await window.render_single_frame(parameters, state)
    }, UI, parameters, state);
    const canvas = await page.$('canvas');
    const buffer = await canvas.screenshot({ type: 'png', omitBackground: true});
    const outputPath = path.join(__dirname, 'output.png');
    fs.writeFileSync(outputPath, buffer);
    console.log(`Screenshot saved to ${outputPath}`);

    await browser.close();
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
            if (err) {
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


async function main(){
    await new Promise(resolve => server.listen(PORT, resolve));
    await render(DATA[0].parameters, DATA[0].trajectory[0])
    // await new Promise(resolve => setTimeout(resolve, 100000000));
    server.close(() => console.log('Server closed.'));
}

main()

