# One-shot launcher: expose the HeYan web UI to other devices on the same LAN.
#
# IMPORTANT: keep this file ASCII-only. Windows PowerShell 5.1 reads a .ps1
# without a BOM as ANSI (GBK on zh-CN systems), so non-ASCII text gets
# mis-decoded and can break the parser. Same rule as heyan/tts/sapi_render.ps1.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File tools\serve_lan.ps1
#   powershell -ExecutionPolicy Bypass -File tools\serve_lan.ps1 -Port 8080 -Https
#   powershell -ExecutionPolicy Bypass -File tools\serve_lan.ps1 -OpenFirewall
#   powershell -ExecutionPolicy Bypass -File tools\serve_lan.ps1 -RemoveFirewallRule
#
# -OpenFirewall needs an elevated shell; without it this script only prints
# the netsh command for you to paste into an admin PowerShell.
param(
    [int]$Port = 8080,
    [string]$Bundle = "",
    [string]$Lang = "zh",
    [switch]$Https,
    [switch]$OpenFirewall,
    [switch]$RemoveFirewallRule
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$ruleName = "heyan-web-$Port"

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

function Show-LanAddresses {
    Write-Output "[lan] Share one of these with the other device (same WiFi/router):"
    $found = $false
    foreach ($ip in (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue)) {
        if ($ip.IPAddress.StartsWith("127.") -or $ip.IPAddress.StartsWith("169.254.")) { continue }
        Write-Output ("[lan]   http://{0}:{1}" -f $ip.IPAddress, $Port)
        $found = $true
    }
    if (-not $found) {
        Write-Output "[lan]   (no LAN IPv4 found - check the network connection, run ipconfig)"
    }
}

if ($RemoveFirewallRule) {
    if (-not $isAdmin) { throw "Removing a firewall rule needs an elevated PowerShell." }
    Remove-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    Write-Output "[lan] Removed inbound rule: $ruleName"
    return
}

if ($OpenFirewall) {
    if (-not $isAdmin) {
        throw "Adding a firewall rule needs an elevated PowerShell (Run as administrator)."
    }
    if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
            -Protocol TCP -LocalPort $Port -Profile Any | Out-Null
        Write-Output "[lan] Added inbound rule: $ruleName (TCP $Port)"
    } else {
        Write-Output "[lan] Inbound rule already present: $ruleName"
    }
} else {
    Write-Output "[lan] Windows Firewall may block inbound TCP $Port for other devices."
    Write-Output "[lan] Re-run with -OpenFirewall from an admin shell, or paste this there:"
    Write-Output ("[lan]   netsh advfirewall firewall add rule name=""{0}"" dir=in action=allow protocol=TCP localport={1}" `
        -f $ruleName, $Port)
}

$serveArgs = @("-X", "utf8", "-m", "heyan.cli", "serve", "--lan", "--port", "$Port", "--lang", $Lang)
if ($Bundle) { $serveArgs += @("--bundle", $Bundle) }

if ($Https) {
    $cert = Join-Path $repo "artifacts\certs\heyan-lan.crt"
    $key = Join-Path $repo "artifacts\certs\heyan-lan.key"
    if (-not (Test-Path -LiteralPath $cert)) {
        Write-Output "[lan] No certificate yet, generating a self-signed one..."
        & python -X utf8 (Join-Path $repo "tools\make_dev_cert.py")
        if ($LASTEXITCODE -ne 0) { throw "Certificate generation failed (exit $LASTEXITCODE)." }
    }
    $serveArgs += @("--ssl-cert", $cert, "--ssl-key", $key)
    Write-Output "[lan] HTTPS on. Visitors must accept the self-signed certificate warning once."
} else {
    Write-Output "[lan] Plain HTTP: browsers disable the in-page viewfinder and offline cache."
    Write-Output "[lan] The shutter button falls back to the system camera/album, so it still works."
    Write-Output "[lan] For the full experience re-run with -Https."
}

Show-LanAddresses
Write-Output "[lan] Starting server (Ctrl+C to stop)..."
& python @serveArgs
exit $LASTEXITCODE
