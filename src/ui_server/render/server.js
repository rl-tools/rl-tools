const fs = require('fs');
const http = require('http');
const path = require('path');
const { parse } = require('url');
const HTML_FILE_NAME = 'puppeteer.html';

module.exports = http.createServer((req, res) => {
    console.log(req.url)
    const parsedUrl = parse(req.url, true);
    const pathname = parsedUrl.pathname;

    if (pathname === '/' || pathname === '/' + HTML_FILE_NAME) {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        const HTML_CONTENT = fs.readFileSync(path.join(__dirname, HTML_FILE_NAME), 'utf8');
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