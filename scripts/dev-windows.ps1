# Development mode on Windows: run the full pipeline with your webcam,
# no Kinect and no Home Assistant needed (dry-run volume simulation).
#
#   .\scripts\dev-windows.ps1              # webcam + web UI at http://localhost:8420
#   .\scripts\dev-windows.ps1 -Preview     # + native preview window (press q to quit)
#
# To also drive your real Home Assistant from your PC:
#   $env:KK_HA_URL = "http://192.168.1.10:8123"
#   $env:KK_HA_TOKEN = "<long-lived token>"
#   $env:KK_VOLUME_ENTITY = "media_player.bose_soundbar_700"
#   $env:KK_MEDIA_ENTITY = "media_player.spotify_yash"
param([switch]$Preview)

$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

# mediapipe 0.10.35+ ships a universal py3 Windows wheel, so any Python >=3.10 works.
$pyCmd = $null
foreach ($v in @("3.13", "3.12", "3.11", "3.10")) {
    try { & py "-$v" --version *> $null; if ($LASTEXITCODE -eq 0) { $pyCmd = @("py", "-$v"); break } } catch {}
}
if (-not $pyCmd) {
    # No py launcher — fall back to whatever `python` is, if it's new enough.
    try {
        $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and [version]$ver -ge [version]"3.10") { $pyCmd = @("python") }
    } catch {}
}
if (-not $pyCmd) {
    Write-Error "Python 3.10+ required. Install from python.org."
    exit 1
}

if (-not (Test-Path ".venv-dev")) {
    Write-Host "Creating dev venv with $($pyCmd -join ' ')..."
    & $pyCmd[0] @($pyCmd[1..($pyCmd.Length)] | Where-Object { $_ }) -m venv .venv-dev
}
& .\.venv-dev\Scripts\python.exe -m pip install --quiet --upgrade pip
& .\.venv-dev\Scripts\python.exe -m pip install --quiet -r requirements.txt
# Editable install so `python -m kinectknob` resolves from src/.
& .\.venv-dev\Scripts\python.exe -m pip install --quiet --no-deps -e .

$args = @("-m", "kinectknob", "--backend", "webcam")
if ($Preview) { $args += "--preview" }
Write-Host "Web UI: http://localhost:8420  (Ctrl+C to stop)"
& .\.venv-dev\Scripts\python.exe @args
