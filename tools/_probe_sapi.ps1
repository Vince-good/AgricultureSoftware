$ErrorActionPreference = "Continue"
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
Write-Output ("DefaultVoice: " + $synth.Voice.Name + " / " + $synth.Voice.Culture.Name)
foreach ($v in $synth.GetInstalledVoices()) {
    $i = $v.VoiceInfo
    Write-Output ("Installed: " + $i.Name + " | " + $i.Culture.Name + " | enabled=" + $v.Enabled)
    try {
        $synth.SelectVoice($i.Name)
        Write-Output ("  SelectVoice OK -> " + $synth.Voice.Name)
    } catch {
        Write-Output ("  SelectVoice FAILED: " + $_.Exception.Message)
    }
}
$out = Join-Path $env:TEMP "heyan_probe_default.wav"
try {
    $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannelCount]::Mono)
    $synth.SetOutputToWaveFile($out, $fmt)
    $txt = "hello"
    if ($args.Count -gt 0 -and (Test-Path $args[0])) {
        $txt = [System.IO.File]::ReadAllText($args[0], [System.Text.Encoding]::UTF8)
    }
    $synth.Speak($txt)
    $synth.SetOutputToNull()
    Write-Output ("SpeakDefault OK bytes=" + (Get-Item $out).Length)
} catch {
    Write-Output ("SpeakDefault FAILED: " + $_.Exception.GetType().Name + " : " + $_.Exception.Message)
}
$synth.Dispose()
