FROM caddy:2-alpine

COPY deployment/fgn-serving/Caddyfile /etc/caddy/Caddyfile
