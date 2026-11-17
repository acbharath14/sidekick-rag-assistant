# One-shot launcher for the Sidekick React GUI + FastAPI backend.
# PowerShell equivalent of scripts/dev.sh: stops stale servers, starts the
# backend in the background (log: $env:TEMP\sidekick-api.log), then starts
# the Vite dev server in the foreground.
#
# Usage: .\scripts\dev.ps1
#   Backend:  http://127.0.0.1:8000
#   Frontend: http://localhost:5173
#
# If PowerShell refuses to run this script, run once:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

function Stop-PortListeners {
    param([int[]]$Ports)
    foreach ($port in $Ports) {
        $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        foreach ($conn in $connections) {
            try { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction Stop } catch { }
        }
    }
}

Write-Host "==> Stopping stale servers..."
Stop-PortListeners -Ports 8000, 5173
Start-Sleep -Seconds 1

Write-Host "==> Starting backend (log: $env:TEMP\sidekick-api.log)..."
if (-not (Test-Path ".venv")) {
    Write-Error ".venv not found in $RepoRoot"
    exit 1
}
. .\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
$backend = Start-Process -FilePath "python" -ArgumentList "-m", "rag_demo.api" `
    -RedirectStandardOutput "$env:TEMP\sidekick-api.log" `
    -RedirectStandardError "$env:TEMP\sidekick-api.log" -PassThru -NoNewWindow
Write-Host "    backend PID $($backend.Id)"

Write-Host "==> Waiting for backend..."
$up = $false
for ($i = 1; $i -le 20; $i++) {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/health" -UseBasicParsing -TimeoutSec 2 | Out-Null
        $up = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}
if ($up) {
    Write-Host "    backend up."
} else {
    Write-Host "    backend did not answer /api/health after 20s; check $env:TEMP\sidekick-api.log"
}

Write-Host "==> Starting frontend..."
Set-Location web
npm run dev
