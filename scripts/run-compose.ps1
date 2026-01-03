$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $root
Set-Location $root

if (-not (Test-Path "config.yaml")) {
    Copy-Item "config.example.yaml" "config.yaml"
}

@("logs", "data", "data\\sandbox") | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

Write-Host "Starting AgentContainer (API + sandbox) via docker compose..." -ForegroundColor Cyan
docker compose up -d --build
