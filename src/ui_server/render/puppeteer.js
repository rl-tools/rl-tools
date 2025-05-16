const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const http = require('http');

// --- Configuration ---
const PORT = 3010;
const OUTPUT_DIR_NAME = 'canvas_frames';
const RECORDING_DURATION_MS = 10000;
const TARGET_FPS = 60;
// --- End Configuration ---

const HTML_FILE_NAME = 'canvas-animation.html';

const HTML_CONTENT = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Canvas Animation for Recording</title>
    <style>
        body { margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #222; overflow: hidden; }
        canvas { border: 1px solid #444; background-color: #333; }
    </style>
</head>
<body>
    <canvas id="myCanvas" width="800" height="600"></canvas>
    <script>
        const canvas = document.getElementById('myCanvas');
        const ctx = canvas.getContext('2d');
        let x = canvas.width / 2;
        let y = canvas.height / 2;
        let radius = 20;
        let dx = (Math.random() - 0.5) * 8;
        let dy = (Math.random() - 0.5) * 8;
        let hue = 0;
        let lastTime = 0;
        let frameCount = 0;

        function draw(timestamp) {
            if (!lastTime) {
                lastTime = timestamp;
            }
            const deltaTimeFactor = (timestamp - lastTime) / (1000 / 60);
            lastTime = timestamp;

            ctx.fillStyle = 'rgba(40, 40, 40, 0.15)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = \`hsl(\${hue}, 100%, 60%)\`;
            ctx.shadowColor = \`hsl(\${hue}, 100%, 50%)\`;
            ctx.shadowBlur = 15;
            ctx.fill();
            ctx.closePath();
            ctx.shadowBlur = 0;

            x += dx * deltaTimeFactor;
            y += dy * deltaTimeFactor;

            if (x + radius > canvas.width || x - radius < 0) {
                dx = -dx;
                x = Math.max(radius, Math.min(x, canvas.width - radius));
                hue = (hue + 30) % 360;
            }
            if (y + radius > canvas.height || y - radius < 0) {
                dy = -dy;
                y = Math.max(radius, Math.min(y, canvas.height - radius));
                hue = (hue + 30) % 360;
            }

            hue = (hue + 0.5 * deltaTimeFactor) % 360;

            frameCount++;
            requestAnimationFrame(draw);
        }

        requestAnimationFrame(draw);
    </script>
</body>
</html>
`;

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


    // … all previous code unchanged …

// 3. Launch Puppeteer and set up recording  ──────────────────────────────
    console.log('Launching Puppeteer...');
    const browser = await puppeteer.launch(/* your launch args */);
    const page    = await browser.newPage();
    await page.setViewport({ width: 800, height: 600 });

// ---- NEW: create a single CDP session and wire up screencast -----------
    const client = await page.target().createCDPSession();

    let frameCounter  = 0;
    const frameWrites = [];

// listen for CDP screencast frames
//     client.on('Page.screencastFrame', async ({ data, sessionId }) => {
//         const framePath = path.join(
//             outputDir,
//             `frame-${String(frameCounter).padStart(5, '0')}.png`
//         );
//
//         // write asynchronously; don’t await here to avoid blocking ACK
//         frameWrites.push(fs.promises.writeFile(framePath, data));
//         frameCounter++;
//         process.stdout.write(`Captured frame: ${frameCounter} \r`);
//
//         // acknowledge so Chrome keeps streaming
//         await client.send('Page.screencastFrameAck', { sessionId });
//     });
    client.on('Page.screencastFrame', async ({ data, sessionId }) => {
        const framePath = path.join(
            outputDir,
            `frame-${String(frameCounter).padStart(5, '0')}.png`
        );

        // convert the base-64 screen-capture to real bytes
        const buffer = Buffer.from(data, 'base64');
        frameWrites.push(fs.promises.writeFile(framePath, buffer));

        frameCounter++;
        process.stdout.write(`Captured frame: ${frameCounter}\r`);

        // ACK so Chrome keeps streaming
        await client.send('Page.screencastFrameAck', { sessionId });
    });


// ------------------------------------------------------------------------

    console.log(`Navigating to http://localhost:${PORT}/`);
    await page.goto(`http://localhost:${PORT}/`, { waitUntil: 'networkidle0' });

    console.log('Starting screencast...');
    await client.send('Page.startScreencast', {
        format: 'png',       // png|jpeg
        quality: 100,        // only for jpeg
        everyNthFrame: 1     // 1 = capture every frame
    });

// 5. Record for the specified duration
    console.log(`Recording for ${RECORDING_DURATION_MS / 1000} seconds...`);
    await new Promise(r => setTimeout(r, RECORDING_DURATION_MS));

// 6. Stop screencast and cleanup
    console.log('\nStopping screencast...');
    await client.send('Page.stopScreencast');

    console.log('Waiting for all frames to be saved…');
    await Promise.all(frameWrites);
    console.log(`${frameCounter} frames saved in: ${outputDir}`);

    console.log('Closing browser and server…');
    await browser.close();
    server.close(() => console.log('Server closed.'));

// … the FFmpeg hint section is unchanged …



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


