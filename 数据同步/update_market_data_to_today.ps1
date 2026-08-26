param(
    [switch]$WaitForMinuteQueue,
    [switch]$SkipMinuteFullRefresh,
    [switch]$SkipRepair,
    [string]$TargetDate = "",
    [string]$StartDate = "",
    [string]$EndDate = "",
    [string]$UpdatePeriods = "monthly,weekly,daily",
    [string]$MinutePeriods = "60m,30m,5m",
    [int]$UpdateWorkers = 8,
    [int]$MinuteWorkers60m = 8,
    [int]$MinuteWorkers30m = 6,
    [int]$MinuteWorkers15m = 4,
    [int]$MinuteWorkers5m = 2,
    [int]$PollSeconds = 60,
    [int]$MaxRepairPasses = 2,
    [int]$MaxUpdateRetryPasses = 2
)

$ErrorActionPreference = "Stop"

function Normalize-DateInput {
    param(
        [string]$Value,
        [string]$ParameterName
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        $text = ""
    }
    else {
        $text = $Value.Trim()
    }
    if (-not $text) {
        return ""
    }

    try {
        return [datetime]::ParseExact(
            $text,
            "yyyy-MM-dd",
            [System.Globalization.CultureInfo]::InvariantCulture
        ).ToString("yyyy-MM-dd")
    }
    catch {
        throw "Invalid $ParameterName value '$Value'. Expected format YYYY-MM-DD."
    }
}

function Resolve-DateOptions {
    param(
        [string]$TargetDateValue,
        [string]$StartDateValue,
        [string]$EndDateValue
    )

    $normalizedTargetDate = Normalize-DateInput -Value $TargetDateValue -ParameterName "TargetDate"
    $normalizedStartDate = Normalize-DateInput -Value $StartDateValue -ParameterName "StartDate"
    $normalizedEndDate = Normalize-DateInput -Value $EndDateValue -ParameterName "EndDate"

    if ($normalizedTargetDate -and -not $normalizedEndDate) {
        $normalizedEndDate = $normalizedTargetDate
    }

    if ($normalizedStartDate -and -not $normalizedEndDate) {
        $normalizedEndDate = $normalizedStartDate
    }

    if ($normalizedStartDate -and $normalizedEndDate) {
        $start = [datetime]::ParseExact($normalizedStartDate, "yyyy-MM-dd", [System.Globalization.CultureInfo]::InvariantCulture)
        $end = [datetime]::ParseExact($normalizedEndDate, "yyyy-MM-dd", [System.Globalization.CultureInfo]::InvariantCulture)
        if ($start -gt $end) {
            throw "StartDate must be on or before EndDate."
        }
    }

    return @{
        targetDate = if ($normalizedEndDate) { $normalizedEndDate } else { Get-Date -Format "yyyy-MM-dd" }
        startDate = $normalizedStartDate
        endDate = $normalizedEndDate
    }
}

$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$workdir = $scriptRoot
$logsDir = Join-Path $workdir "logs"
$configPath = Join-Path $workdir "market_store_manual_sync.json"
$syncScriptPath = Join-Path $workdir "sync_market_data_store.py"
$queueScriptPath = Join-Path $workdir "run_minute_sync_queue.py"
$minuteQueueOrchestratorPath = Join-Path $workdir "queue_all_minute_periods.ps1"
$stdoutLog = Join-Path $logsDir "update_market_data_to_today.log"
$stderrLog = Join-Path $logsDir "update_market_data_to_today.err.log"
$minuteUsesExplicitDateWindow = (-not [string]::IsNullOrWhiteSpace($StartDate)) -or (-not [string]::IsNullOrWhiteSpace($EndDate))
$dateOptions = Resolve-DateOptions -TargetDateValue $TargetDate -StartDateValue $StartDate -EndDateValue $EndDate
$targetDate = [string]$dateOptions.targetDate
$syncStartDate = [string]$dateOptions.startDate
$syncEndDate = [string]$dateOptions.endDate

function Get-SelectedPeriods {
    param([string]$Value)

    if (-not $Value) {
        return @()
    }

    return @(
        $Value.Split(",") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
    )
}

$selectedMinutePeriods = @(Get-SelectedPeriods -Value $MinutePeriods)

if (-not (Test-Path -LiteralPath $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$minuteSteps = @(
    @{ periods = "60m"; timeout = 240; state = "logs/minute_queue_state_60m.csv"; workers = $MinuteWorkers60m },
    @{ periods = "30m"; timeout = 300; state = "logs/minute_queue_state_30m.csv"; workers = $MinuteWorkers30m },
    @{ periods = "15m"; timeout = 420; state = "logs/minute_queue_state_15m.csv"; workers = $MinuteWorkers15m },
    @{ periods = "5m"; timeout = 900; state = "logs/minute_queue_state_5m.csv"; workers = $MinuteWorkers5m }
)
$minuteSteps = @($minuteSteps | Where-Object { $_.periods -in $selectedMinutePeriods })

function Write-UpdateLog {
    param([string]$Message)

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $Message" | Out-File -FilePath $stdoutLog -Append -Encoding utf8
}

function Write-UpdateError {
    param([string]$Message)

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $Message" | Out-File -FilePath $stderrLog -Append -Encoding utf8
    Write-UpdateLog "ERROR $Message"
}

function Get-PythonExecutable {
    $command = Get-Command python -ErrorAction Stop
    return $command.Source
}

function Format-ArgumentList {
    param([string[]]$Arguments)

    return ($Arguments | ForEach-Object {
            if ($_ -match "\s") {
                '"' + $_ + '"'
            }
            else {
                $_
            }
        }) -join " "
}

function Invoke-PythonLogged {
    param(
        [string]$PythonExe,
        [string[]]$Arguments,
        [string]$Label
    )

    Write-UpdateLog "starting $Label command=$PythonExe $(Format-ArgumentList -Arguments $Arguments)"
    & $PythonExe @Arguments 2>&1 | ForEach-Object {
        $line = $_.ToString().TrimEnd()
        if ($line) {
            Write-UpdateLog "[$Label] $line"
        }
    }
    $exitCode = $LASTEXITCODE
    Write-UpdateLog "finished $Label exit_code=$exitCode"
    if ($exitCode -ne 0) {
        Write-UpdateError "$Label returned non-zero exit_code=$exitCode"
    }
    return $exitCode
}

function Get-ActiveMinuteQueueProcesses {
    $processes = @(Get-CimInstance Win32_Process | Where-Object {
            ($_.Name -eq "powershell.exe" -and $_.CommandLine -match "queue_all_minute_periods\.ps1") -or
            ($_.Name -eq "python.exe" -and $_.CommandLine -match "run_minute_sync_queue\.py")
        })
    return $processes
}

function Wait-ForMinuteQueueToFinish {
    while ($true) {
        $processes = @(Get-ActiveMinuteQueueProcesses)
        if ($processes.Count -eq 0) {
            Write-UpdateLog "no active minute queue detected; continuing"
            return
        }

        $summary = ($processes | ForEach-Object { "$($_.Name):$($_.ProcessId)" }) -join ", "
        Write-UpdateLog "waiting for active minute queue to finish poll_seconds=$PollSeconds processes=$summary"
        Start-Sleep -Seconds $PollSeconds
    }
}

function Get-LatestRowsBySymbol {
    param([string]$StatePath)

    if (-not (Test-Path -LiteralPath $StatePath)) {
        return @()
    }

    $rows = Import-Csv -LiteralPath $StatePath
    if (-not $rows) {
        return @()
    }

    $latestBySymbol = @{}
    foreach ($row in $rows) {
        if ($row.symbol) {
            $latestBySymbol[$row.symbol] = $row
        }
    }

    return @($latestBySymbol.Values)
}

function Get-LatestFailedRows {
    param([string]$StatePath)

    return @(Get-LatestRowsBySymbol -StatePath $StatePath | Where-Object { $_.status -eq "failed" } | Sort-Object symbol)
}

function Log-MinuteStateSummary {
    foreach ($step in $minuteSteps) {
        $statePath = Join-Path $workdir $step.state
        $latestRows = @(Get-LatestRowsBySymbol -StatePath $statePath)
        $failedRows = @($latestRows | Where-Object { $_.status -eq "failed" })
        $okRows = @($latestRows | Where-Object { $_.status -eq "ok" })
        Write-UpdateLog "state_summary period=$($step.periods) latest_ok=$($okRows.Count) latest_failed=$($failedRows.Count) latest_total=$($latestRows.Count)"
        if ($failedRows.Count -gt 0) {
            $sample = ($failedRows | Select-Object -ExpandProperty symbol -First 10) -join ","
            Write-UpdateLog "state_summary period=$($step.periods) failed_sample=$sample"
        }
    }
}

function Repair-MinuteFailures {
    param([string]$PythonExe)

    foreach ($step in $minuteSteps) {
        $statePath = Join-Path $workdir $step.state

        for ($pass = 1; $pass -le $MaxRepairPasses; $pass++) {
            $failedBefore = @(Get-LatestFailedRows -StatePath $statePath)
            if ($failedBefore.Count -eq 0) {
                Write-UpdateLog "repair skipped period=$($step.periods) because latest_failed=0"
                break
            }

            $sample = ($failedBefore | Select-Object -ExpandProperty symbol -First 10) -join ","
            $stockList = ($failedBefore | Select-Object -ExpandProperty symbol | Sort-Object -Unique) -join ","
            Write-UpdateLog "repair starting period=$($step.periods) pass=$pass latest_failed_before=$($failedBefore.Count) sample=$sample"

            $arguments = @(
                $queueScriptPath,
                "--config", $configPath,
                "--stocks", $stockList,
                "--periods", $step.periods,
                "--timeout-seconds", [string]$step.timeout,
                "--sleep-seconds", "0.5",
                "--workers", [string]$step.workers,
                "--state-file", $step.state,
                "--target-date", $targetDate
            )
            # Only pass an explicit minute sync window when the caller requested one.
            # TargetDate alone should keep minute fallback sources on incremental refresh mode.
            if ($minuteUsesExplicitDateWindow -and $syncStartDate) {
                $arguments += @("--start-date", $syncStartDate)
            }
            if ($minuteUsesExplicitDateWindow -and $syncEndDate) {
                $arguments += @("--end-date", $syncEndDate)
            }
            [void](Invoke-PythonLogged -PythonExe $PythonExe -Arguments $arguments -Label "repair-$($step.periods)-pass$pass")

            $failedAfter = @(Get-LatestFailedRows -StatePath $statePath)
            Write-UpdateLog "repair finished period=$($step.periods) pass=$pass latest_failed_after=$($failedAfter.Count)"

            if ($failedAfter.Count -eq 0) {
                break
            }

            if ($failedAfter.Count -ge $failedBefore.Count) {
                Write-UpdateError "repair period=$($step.periods) pass=$pass made no progress latest_failed_after=$($failedAfter.Count)"
                break
            }
        }

        $remaining = @(Get-LatestFailedRows -StatePath $statePath)
        if ($remaining.Count -gt 0) {
            $symbols = ($remaining | Select-Object -ExpandProperty symbol -First 20) -join ","
            Write-UpdateError "repair period=$($step.periods) remaining_failed=$($remaining.Count) sample=$symbols"
        }
    }
}

function Invoke-MinuteFullRefresh {
    if ($selectedMinutePeriods.Count -eq 0) {
        Write-UpdateLog "minute full refresh skipped because selected_minute_periods=0"
        return 0
    }

    $periodList = $selectedMinutePeriods -join ","
    $arguments = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $minuteQueueOrchestratorPath,
        "-MinutePeriods", $periodList,
        "-TargetDate", $targetDate
    )
    if ($minuteUsesExplicitDateWindow -and $syncStartDate) {
        $arguments += @("-StartDate", $syncStartDate)
    }
    if ($minuteUsesExplicitDateWindow -and $syncEndDate) {
        $arguments += @("-EndDate", $syncEndDate)
    }

    Write-UpdateLog "starting minute-full-refresh command=powershell $(Format-ArgumentList -Arguments $arguments)"
    & powershell @arguments 2>&1 | ForEach-Object {
        $line = $_.ToString().TrimEnd()
        if ($line) {
            Write-UpdateLog "[minute-full-refresh] $line"
        }
    }
    $exitCode = $LASTEXITCODE
    Write-UpdateLog "finished minute-full-refresh exit_code=$exitCode"
    if ($exitCode -ne 0) {
        Write-UpdateError "minute-full-refresh returned non-zero exit_code=$exitCode"
    }
    return $exitCode
}

function Log-RegistrySummary {
    $registryPath = "D:\a_stock_data\metadata\sync_registry.json"
    if (-not (Test-Path -LiteralPath $registryPath)) {
        Write-UpdateError "sync registry not found after update: $registryPath"
        return @()
    }

    $registry = Get-Content -LiteralPath $registryPath -Raw | ConvertFrom-Json
    $successCount = if ($null -ne $registry.success_count) { [int]$registry.success_count } else { -1 }
    $failureCount = if ($null -ne $registry.failure_count) { [int]$registry.failure_count } else { -1 }
    $syncedAt = if ($registry.synced_at) { [string]$registry.synced_at } else { "" }
    Write-UpdateLog "update_registry synced_at=$syncedAt success_count=$successCount failure_count=$failureCount"

    $failedSymbols = @()
    if ($failureCount -gt 0 -and $registry.failures) {
        $failedSymbols = @($registry.failures | ForEach-Object { [string]$_.symbol } | Where-Object { $_ })
        $sample = @($failedSymbols | Select-Object -First 10) -join ","
        Write-UpdateError "update_registry remaining_failures=$failureCount sample=$sample"
    }
    return $failedSymbols
}

function Invoke-UpdateToToday {
    param([string]$PythonExe)

    $baseArguments = @($syncScriptPath, $configPath, "--skip-concepts", "--workers", [string]$UpdateWorkers)
    if ($UpdatePeriods.Trim()) {
        $baseArguments += @("--periods", $UpdatePeriods)
    }
    if ($syncStartDate) {
        $baseArguments += @("--start-date", $syncStartDate)
    }
    if ($syncEndDate) {
        $baseArguments += @("--end-date", $syncEndDate)
    }

    [void](Invoke-PythonLogged -PythonExe $PythonExe -Arguments $baseArguments -Label "update-to-today")
    $failedSymbols = @(Log-RegistrySummary)

    for ($pass = 1; $pass -le $MaxUpdateRetryPasses; $pass++) {
        if ($failedSymbols.Count -eq 0) {
            Write-UpdateLog "update retry skipped because remaining_failures=0"
            break
        }

        $stockList = ($failedSymbols | Sort-Object -Unique) -join ","
        $sample = @($failedSymbols | Select-Object -First 10) -join ","
        Write-UpdateLog "update retry starting pass=$pass remaining_failures=$($failedSymbols.Count) sample=$sample"

        $retryArguments = @($syncScriptPath, $configPath, "--skip-concepts", "--workers", [string]$UpdateWorkers, "--stocks", $stockList)
        if ($UpdatePeriods.Trim()) {
            $retryArguments += @("--periods", $UpdatePeriods)
        }
        if ($syncStartDate) {
            $retryArguments += @("--start-date", $syncStartDate)
        }
        if ($syncEndDate) {
            $retryArguments += @("--end-date", $syncEndDate)
        }

        [void](Invoke-PythonLogged -PythonExe $PythonExe -Arguments $retryArguments -Label "update-retry-pass$pass")
        $newFailedSymbols = @(Log-RegistrySummary)

        if ($newFailedSymbols.Count -eq 0) {
            break
        }

        if ($newFailedSymbols.Count -ge $failedSymbols.Count) {
            Write-UpdateError "update retry pass=$pass made no progress remaining_failures=$($newFailedSymbols.Count)"
            $failedSymbols = $newFailedSymbols
            break
        }

        $failedSymbols = $newFailedSymbols
    }
}

Set-Location $workdir
Write-UpdateLog "========== update run started target_date=$targetDate start_date=$syncStartDate end_date=$syncEndDate update_periods=$UpdatePeriods update_workers=$UpdateWorkers minute_periods=$($selectedMinutePeriods -join ',') minute_workers=60m:$MinuteWorkers60m,30m:$MinuteWorkers30m,15m:$MinuteWorkers15m,5m:$MinuteWorkers5m =========="

try {
    $pythonExe = Get-PythonExecutable
    Write-UpdateLog "python_executable=$pythonExe"

    if ($WaitForMinuteQueue) {
        Wait-ForMinuteQueueToFinish
    }

    Log-MinuteStateSummary

    if (-not $SkipMinuteFullRefresh) {
        [void](Invoke-MinuteFullRefresh)
        Log-MinuteStateSummary
    }
    else {
        Write-UpdateLog "minute full refresh skipped because SkipMinuteFullRefresh was set"
    }

    if (-not $SkipRepair) {
        Repair-MinuteFailures -PythonExe $pythonExe
        Log-MinuteStateSummary
    }
    else {
        Write-UpdateLog "minute repair skipped because SkipRepair was set"
    }

    Invoke-UpdateToToday -PythonExe $pythonExe
    Write-UpdateLog "========== update run finished =========="
}
catch {
    Write-UpdateError "update run failed: $($_.Exception.Message)"
    throw
}
