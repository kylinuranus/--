$pricePid = 30100
$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$workdir = $scriptRoot
$logsDir = Join-Path $workdir 'logs'
$stdout = Join-Path $logsDir 'manual_minute_sync_stdout.log'
$stderr = Join-Path $logsDir 'manual_minute_sync_stderr.log'
$queueScript = Join-Path $workdir 'queue_all_minute_periods.ps1'

if (-not (Test-Path -LiteralPath $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

while (Get-Process -Id $pricePid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

if (Test-Path $stdout) {
    Remove-Item -LiteralPath $stdout -Force
}
if (Test-Path $stderr) {
    Remove-Item -LiteralPath $stderr -Force
}

Set-Location $workdir
powershell -NoProfile -ExecutionPolicy Bypass -File $queueScript 1> $stdout 2> $stderr
