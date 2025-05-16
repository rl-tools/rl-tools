const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const http = require('http');

const PORT = 3010;
const OUTPUT_DIR_NAME = 'canvas_frames';
const RECORDING_DURATION_MS = 10000;
const TARGET_FPS = 60;

const HTML_FILE_NAME = 'puppeteer.html';

const HTML_CONTENT = fs.readFileSync(path.join(__dirname, HTML_FILE_NAME), 'utf8');
console.log(`HTML content loaded from ${HTML_FILE_NAME}: \n${HTML_CONTENT}`);

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

    const server = http.createServer((req, res) => {
        if (req.url === '/' || req.url === '/' + HTML_FILE_NAME) {
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(HTML_CONTENT);
        } else {
            res.writeHead(404);
            res.end('Not Found');
        }
    });

    await new Promise(resolve => server.listen(PORT, resolve));
    console.log(`Local server running at http://localhost:${PORT}/`);

    console.log('Launching Puppeteer...');
    const browser = await puppeteer.launch(/* your launch args */);
    const page    = await browser.newPage();
    await page.setViewport({ width: 800, height: 600 });

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

recordCanvasAnimation().catch(error => {
    console.error('An error occurred during the recording process:', error);
    process.exit(1);
});


