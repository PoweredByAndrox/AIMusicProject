# syntax=docker/dockerfile:1.7
FROM node:20-alpine

WORKDIR /app

COPY package.json .
COPY package-lock.json .
COPY tsconfig.json .
COPY vite.config.ts .
COPY index.html .
COPY src ./src

RUN --mount=type=cache,target=/root/.npm npm ci --verbose

CMD ["npm", "run", "dev"]
