param(
    [string]$TaskName = "FX_Strategy1_DailyPipeline",
    [string]$StartTime = "18:05"
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$pipelineScript = Resolve-Path (Join-Path $PSScriptRoot "run_daily_paper_pipeline.ps1")

if ($StartTime -notmatch '^\d{2}:\d{2}$') {
    throw "StartTime must be HH:MM in 24-hour format."
}

$errors = @()

try {
    $atTime = [datetime]::ParseExact($StartTime, "HH:mm", $null)
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$pipelineScript`""
    $trigger = New-ScheduledTaskTrigger -Daily -At $atTime
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
    $taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Output "Task '$TaskName' is scheduled daily at $StartTime (local machine time)."
    Write-Output "Next run time: $($taskInfo.NextRunTime)"
    Write-Output "Pipeline script: $pipelineScript"
    Write-Output "Repo root: $repoRoot"
    exit 0
} catch {
    $errors += "ScheduledTasks cmdlets failed: $($_.Exception.Message)"
}

try {
    $taskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$pipelineScript`""
    & schtasks.exe /Create /F /SC DAILY /TN $TaskName /TR $taskCommand /ST $StartTime | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "schtasks create failed rc=$LASTEXITCODE"
    }
    & schtasks.exe /Query /TN $TaskName /V /FO LIST | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "schtasks query failed rc=$LASTEXITCODE"
    }
    Write-Output "Task '$TaskName' is scheduled daily at $StartTime (local machine time)."
    Write-Output "Pipeline script: $pipelineScript"
    Write-Output "Repo root: $repoRoot"
    exit 0
} catch {
    $errors += "schtasks fallback failed: $($_.Exception.Message)"
}

throw ("Unable to register scheduled task. " + ($errors -join " | "))
