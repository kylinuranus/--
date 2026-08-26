param(
    [string]$MinutePeriods = '60m,30m,5m',
    [string]$TargetDate = '',
    [string]$StartDate = '',
    [string]$EndDate = ''
)

$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$workdir = $scriptRoot
$logsDir = Join-Path $workdir 'logs'
$stdout = Join-Path $logsDir 'minute_master_queue.log'
$stderr = Join-Path $logsDir 'minute_master_queue.err.log'

function Normalize-DateInput {
    param(
        [string]$Value,
        [string]$ParameterName
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        $text = ''
    }
    else {
        $text = $Value.Trim()
    }
    if (-not $text) {
        return ''
    }

    try {
        return [datetime]::ParseExact(
            $text,
            'yyyy-MM-dd',
            [System.Globalization.CultureInfo]::InvariantCulture
        ).ToString('yyyy-MM-dd')
    }
    catch {
        throw "Invalid $ParameterName value '$Value'. Expected format YYYY-MM-DD."
    }
}

$normalizedTargetDate = Normalize-DateInput -Value $TargetDate -ParameterName 'TargetDate'
$StartDate = Normalize-DateInput -Value $StartDate -ParameterName 'StartDate'
$EndDate = Normalize-DateInput -Value $EndDate -ParameterName 'EndDate'
if ($StartDate -and $EndDate) {
    $start = [datetime]::ParseExact($StartDate, 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    $end = [datetime]::ParseExact($EndDate, 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    if ($start -gt $end) {
        throw 'StartDate must be on or before EndDate.'
    }
}
$targetDate = if ($normalizedTargetDate) { $normalizedTargetDate } elseif ($EndDate) { $EndDate } else { Get-Date -Format 'yyyy-MM-dd' }
$defaultWorkers = @{
    '60m' = 8
    '30m' = 6
    '15m' = 4
    '5m'  = 2
}

function Get-SelectedPeriods {
    param([string]$Value)

    if (-not $Value) {
        return @()
    }

    return @(
        $Value.Split(',') |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
    )
}

function Write-MasterLog {
    param([string]$Message)

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "[$timestamp] $Message" | Out-File -FilePath $stdout -Append -Encoding utf8
}

function Get-LatestFailedCount {
    param([string]$StatePath)

    if (-not (Test-Path -LiteralPath $StatePath)) {
        return 0
    }

    $rows = Import-Csv -LiteralPath $StatePath
    if (-not $rows) {
        return 0
    }

    $latestBySymbol = @{}
    foreach ($row in $rows) {
        if ($row.symbol) {
            $latestBySymbol[$row.symbol] = $row
        }
    }

    return @($latestBySymbol.Values | Where-Object { $_.status -eq 'failed' }).Count
}

function Invoke-QueuePass {
    param(
        [hashtable]$Step,
        [string]$PassLabel,
        [string]$StartFromSymbol = ""
    )

    Write-MasterLog "starting periods=$($Step.periods) pass=$PassLabel workers=$($Step.workers)"
    $arguments = @(
        'run_minute_sync_queue.py',
        '--periods', $Step.periods,
        '--timeout-seconds', $Step.timeout,
        '--sleep-seconds', '0.5',
        '--workers', $Step.workers,
        '--state-file', $Step.state,
        '--target-date', $targetDate
    )
    if ($StartDate) {
        $arguments += @('--start-date', $StartDate)
    }
    if ($EndDate) {
        $arguments += @('--end-date', $EndDate)
    }
    if ($StartFromSymbol) {
        $arguments += @('--start-from-symbol', $StartFromSymbol)
    }
    python @arguments 1>> $Step.log 2>> $Step.err
    $exitCode = $LASTEXITCODE
    Write-MasterLog "finished periods=$($Step.periods) pass=$PassLabel exit_code=$exitCode"
    return $exitCode
}

if (Test-Path $stdout) {
    Remove-Item -LiteralPath $stdout -Force
}
if (Test-Path $stderr) {
    Remove-Item -LiteralPath $stderr -Force
}
if (-not (Test-Path -LiteralPath $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$steps = @(
    @{ periods = '60m'; timeout = 240; workers = $defaultWorkers['60m']; state = 'logs/minute_queue_state_60m.csv'; log = 'logs/minute_queue_60m.log'; err = 'logs/minute_queue_60m.err.log' },
    @{ periods = '30m'; timeout = 300; workers = $defaultWorkers['30m']; state = 'logs/minute_queue_state_30m.csv'; log = 'logs/minute_queue_30m.log'; err = 'logs/minute_queue_30m.err.log' },
    @{ periods = '15m'; timeout = 420; workers = $defaultWorkers['15m']; state = 'logs/minute_queue_state_15m.csv'; log = 'logs/minute_queue_15m.log'; err = 'logs/minute_queue_15m.err.log' },
    @{ periods = '5m'; timeout = 900; workers = $defaultWorkers['5m']; state = 'logs/minute_queue_state_5m.csv'; log = 'logs/minute_queue_5m.log'; err = 'logs/minute_queue_5m.err.log' }
)
$selectedPeriods = @(Get-SelectedPeriods -Value $MinutePeriods)
$steps = @($steps | Where-Object { $_.periods -in $selectedPeriods })

Set-Location $workdir
Write-MasterLog 'minute queue started'

foreach ($step in $steps) {
    $firstExitCode = Invoke-QueuePass -Step $step -PassLabel 'initial'
    $statePath = Join-Path $workdir $step.state
    $latestFailedCount = Get-LatestFailedCount -StatePath $statePath
    Write-MasterLog "periods=$($step.periods) latest_failed_after_initial=$latestFailedCount"

    if ($latestFailedCount -gt 0) {
        Write-MasterLog "retrying periods=$($step.periods) for latest_failed_symbols=$latestFailedCount"
        $retryExitCode = Invoke-QueuePass -Step $step -PassLabel 'retry'
        $remainingFailedCount = Get-LatestFailedCount -StatePath $statePath
        Write-MasterLog "periods=$($step.periods) latest_failed_after_retry=$remainingFailedCount"

        if ($retryExitCode -ne 0) {
            Write-MasterLog "period $($step.periods) retry returned non-zero, continuing to next period"
        }
    }
    else {
        Write-MasterLog "periods=$($step.periods) retry skipped because latest_failed_after_initial=0"
    }

    if ($firstExitCode -ne 0) {
        Write-MasterLog "period $($step.periods) initial pass returned non-zero, continuing to next period"
    }
}

Write-MasterLog 'minute queue finished'
