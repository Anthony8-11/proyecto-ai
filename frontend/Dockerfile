FROM oven/bun:1 AS base
WORKDIR /app

COPY package.json bunfig.toml ./
RUN bun install --frozen-lockfile

COPY . .

EXPOSE 5173
CMD ["bun", "run", "dev", "--host"]
