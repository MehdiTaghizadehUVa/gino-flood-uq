param(
  [string]$Distribution = "Ubuntu-22.04"
)

$ErrorActionPreference = "Stop"

Write-Host "Checking WSL installation..."
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
  throw "wsl.exe is not available. Enable Windows Subsystem for Linux from Windows Features first."
}

$installed = & wsl.exe -l -q 2>$null
if ($installed -notcontains $Distribution) {
  Write-Host "Installing $Distribution. A reboot may be required if WSL was not enabled."
  & wsl.exe --install -d $Distribution
} else {
  Write-Host "$Distribution is already installed."
}

Write-Host "Install Docker Desktop with WSL2 integration and NVIDIA Container Toolkit support."
Write-Host "After Docker Desktop is installed, run:"
Write-Host "  wsl -d $Distribution -- docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"
Write-Host "Then copy .env.example to .env and set FGN_DATA_ROOT to the large data disk mount."
