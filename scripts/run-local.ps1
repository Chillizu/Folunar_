$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $root
Set-Location $root

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$python = Join-Path $root ".venv\\Scripts\\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtualenv python not found at $python"
}

& $python -m pip install -U pip
& $python -m pip install -r requirements.txt

if (-not (Test-Path "config.yaml")) {
    Copy-Item "config.example.yaml" "config.yaml"
}

& $python main.py
