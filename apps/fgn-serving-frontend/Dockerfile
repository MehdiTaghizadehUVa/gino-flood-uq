FROM node:22-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json ./
COPY next.config.ts ./
COPY app ./app
COPY public ./public
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
