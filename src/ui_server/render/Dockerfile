FROM ghcr.io/puppeteer/puppeteer:latest
USER root
RUN mkdir render
COPY package.json render/
COPY package-lock.json render/
COPY download_dependencies.sh render/
RUN ./download_dependencies.sh
COPY puppeteer.* render/
RUN chown -R pptruser:pptruser render
USER pptruser
WORKDIR render
RUN npm install
CMD ["node", "puppeteer.js"]