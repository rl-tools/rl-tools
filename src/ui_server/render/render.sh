docker run -it --init --rm --cap-add=SYS_ADMIN -v $(pwd):/mnt --platform=linux/amd64 -w /mnt ghcr.io/puppeteer/puppeteer:latest node puppeteer.js
