# One-shot launcher: share the HeYan web UI over the public internet.
#
# IMPORTANT: keep this file ASCII-only. Windows PowerShell 5.1 reads a .ps1
# without a BOM as ANSI (GBK on zh-CN systems), so non-ASCII text gets
# mis-decoded and can break the parser. Same rule as tools\serve_lan.ps1.
#
# Usage (from the repo root, or just double-click tools\share_public.bat):
#   powershell -ExecutionPolicy Bypass -File tools\share_public.ps1
#   powershell -ExecutionPolicy Bypass -File tools\share_public.ps1 -Client cpolar
#   powershell -ExecutionPolicy Bypass -File tools\share_public.ps1 -Port 8080 -Token my-secret
#   powershell -ExecutionPolicy Bypass -File tools\share_public.ps1 -DryRun
#
# `heyan tunnel` starts its own server, so a port that is already serving
# something is a hard error there (exit 7) rather than a silent hijack of an
# older, unhardened process. This wrapper picks a free port for you unless you
# insist on a specific one.
param(
    [int]$Port = 0,
    [string]$Client = "",
    [string]$Token = "",
    [string]$Bundle = "",
    [string]$Lang = "zh",
    [string]$Region = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo

function Test-PortFree {
    param([int]$Candidate)
    $listener = $null
    try {
        $listener = New-Object System.Net.Sockets.TcpListener(
            [System.Net.IPAddress]::Loopback, $Candidate)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) { $listener.Stop() }
    }
}

if ($Port -eq 0) {
    $picked = 0
    foreach ($candidate in 8080..8099) {
        if (Test-PortFree $candidate) { $picked = $candidate; break }
    }
    if ($picked -eq 0) { throw "No free port in 8080-8099. Close something and retry." }
    $Port = $picked
    Write-Output "[share] Picked a free port: $Port"
} elseif (-not (Test-PortFree $Port)) {
    throw ("Port {0} is already in use (an older 'heyan serve' still running?). Stop it first, or pass -Port 8081." -f $Port)
}

$tunnelArgs = @("-X", "utf8", "-m", "heyan.cli", "tunnel",
                "--port", "$Port", "--lang", $Lang)
if ($Client) { $tunnelArgs += @("--client", $Client) }
if ($Token) { $tunnelArgs += @("--access-token", $Token) }
if ($Bundle) { $tunnelArgs += @("--bundle", $Bundle) }
if ($Region) { $tunnelArgs += @("--region", $Region) }

if ($DryRun) {
    Write-Output "[share] Dry run. Would execute: python $($tunnelArgs -join ' ')"
    exit 0
}

Write-Output "[share] Before you send the link to anyone:"
Write-Output "[share]   - It ends with ?token=... and that token unlocks every recognition"
Write-Output "[share]     record on this machine. Send it to named people, not to a group."
Write-Output "[share]   - Free tiers hand out a new random domain on every restart, so links"
Write-Output "[share]     you already sent stop working the next time you run this."
Write-Output "[share]   - Keep this window open. Ctrl+C stops the server and the tunnel together."
Write-Output "[share] Starting server + tunnel on port $Port..."
& python @tunnelArgs
$code = $LASTEXITCODE
if ($code -eq 7) {
    Write-Output "[share] Port $Port got taken between the check and the start. Re-run, or pass -Port 8081."
}
exit $code
