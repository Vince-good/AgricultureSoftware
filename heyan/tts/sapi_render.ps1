# Offline Mandarin/Cantonese speech synthesis via Windows System.Speech (SAPI5).
#
# IMPORTANT: keep this file ASCII-only. Windows PowerShell 5.1 reads a .ps1
# without a BOM as ANSI (GBK on zh-CN systems), so non-ASCII comments get
# mis-decoded and can break the parser. Text arrives through a UTF-8 file for
# the same reason -- command-line arguments lose their encoding.
param(
    [string]$TextFile = "",
    [string]$OutFile = "",
    [string]$Voice = "",
    [string]$Culture = "",
    [int]$Rate = 0,
    [int]$Volume = 100,
    [int]$SampleRate = 22050,
    [switch]$ListVoices
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

if ($ListVoices) {
    foreach ($v in $synth.GetInstalledVoices()) {
        $i = $v.VoiceInfo
        Write-Output ("{0}|{1}|{2}|{3}" -f $i.Name, $i.Culture.Name, $i.Gender, $v.Enabled)
    }
    $synth.Dispose()
    exit 0
}

if ([string]::IsNullOrWhiteSpace($OutFile)) {
    Write-Error "OutFile is required"
    exit 2
}

if ([string]::IsNullOrWhiteSpace($TextFile)) {
    Write-Error "TextFile is required"
    exit 2
}

$text = [System.IO.File]::ReadAllText($TextFile, [System.Text.Encoding]::UTF8)
if ([string]::IsNullOrWhiteSpace($text)) {
    Write-Error "empty text"
    exit 3
}

$selected = $false
foreach ($name in ($Voice -split ";")) {
    $n = $name.Trim()
    if ($n -eq "") { continue }
    try { $synth.SelectVoice($n); $selected = $true; break } catch { }
}
if (-not $selected -and -not [string]::IsNullOrWhiteSpace($Culture)) {
    # Exact culture match first (zh-HK), then base language (zh-*).
    $base = ($Culture -split "-")[0]
    foreach ($v in $synth.GetInstalledVoices()) {
        try {
            if ($v.VoiceInfo.Culture.Name -eq $Culture) {
                $synth.SelectVoice($v.VoiceInfo.Name); $selected = $true; break
            }
        } catch { }
    }
    if (-not $selected) {
        foreach ($v in $synth.GetInstalledVoices()) {
            $c = $v.VoiceInfo.Culture.Name
            try {
                if ($c -and ($c -split "-")[0] -eq $base) {
                    $synth.SelectVoice($v.VoiceInfo.Name); $selected = $true; break
                }
            } catch { }
        }
    }
}
if (-not $selected) {
    # Last resort: whatever the OS default is. Better than silence, and the
    # caller records which voice actually spoke so the build log stays honest.
    Write-Output ("WARN|no voice matched, using default|" + $synth.Voice.Name)
}

$synth.Rate = [Math]::Max(-10, [Math]::Min(10, $Rate))
$synth.Volume = [Math]::Max(0, [Math]::Min(100, $Volume))

# The channel enum is spelled "AudioChannel" on .NET Framework (Windows
# PowerShell 5.1) and "AudioChannelCount" on the .NET System.Speech package
# (pwsh 7). Probe both, and if neither works just let SAPI pick its default
# format -- a voice's default wav format is always playable.
$bits = [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen
$chan = $null
foreach ($tn in @("AudioChannel", "AudioChannelCount")) {
    try {
        $ct = $synth.GetType().Assembly.GetType("System.Speech.AudioFormat.$tn")
        if ($ct) { $chan = [System.Enum]::ToObject($ct, 1); break }
    } catch { }
}
$fmt = $null
if ($chan) {
    try {
        $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo($SampleRate, $bits, $chan)
    } catch { $fmt = $null }
}

$dir = Split-Path -Parent $OutFile
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

if ($fmt) { $synth.SetOutputToWaveFile($OutFile, $fmt) } else { $synth.SetOutputToWaveFile($OutFile) }
try {
    $synth.Speak($text)
    $usedVoice = $synth.Voice.Name
} finally {
    $synth.SetOutputToNull()
    $synth.Dispose()
}
Write-Output ("OK|{0}|{1}" -f $usedVoice, (Get-Item $OutFile).Length)
