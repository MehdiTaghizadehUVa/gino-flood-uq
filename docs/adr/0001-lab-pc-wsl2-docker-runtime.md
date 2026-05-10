# 0001. Lab PC Runtime Uses WSL2 And Docker Compose

## Status

Accepted

## Context

The V1 Coastal FGN Serving deployment targets the lab PC, a Windows 11 machine with an RTX 4090 GPU. The serving stack is Linux-oriented and uses FastAPI, Celery, Redis, Postgres, Caddy, and a CUDA/PyTorch worker.

## Decision

Run the production stack in WSL2 with Docker Desktop and GPU-enabled Docker Compose. Keep Windows as the host operating system and use bind mounts from a required `FGN_DATA_ROOT` on a large data disk.

## Consequences

- The deployment stays close to the Rivanna/Linux development environment.
- GPU validation must include both host `nvidia-smi` and Docker `--gpus all` smoke checks.
- Docker Desktop, WSL2 integration, and NVIDIA container support become explicit setup requirements.
