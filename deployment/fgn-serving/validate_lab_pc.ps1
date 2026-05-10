$ErrorActionPreference = "Stop"

Write-Host "Host GPU:"
nvidia-smi

Write-Host "WSL distributions:"
wsl -l -v

Write-Host "Docker GPU smoke:"
wsl -- docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

Write-Host "Compose config:"
wsl -- bash -lc "cd deployment/fgn-serving && docker compose config >/tmp/fgn-serving-compose.yml && head -40 /tmp/fgn-serving-compose.yml"
