# syntax=docker/dockerfile:1.7
FROM node:20-alpine

WORKDIR /app

ENV NPM_CONFIG_UPDATE_NOTIFIER=false \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_AUDIT=false

COPY package.json ./
COPY package-lock.json* ./
COPY tsconfig.json .
COPY vite.config.ts .
COPY index.html .
COPY src ./src

RUN --mount=type=cache,target=/root/.npm sh -lc "if [ -f package-lock.json ]; then npm ci --verbose; else npm install --verbose; fi"

CMD ["npm", "run", "dev"]
