param(
    [string]$Manifest = "manifest.json",
    [string]$Profile = "strategy_1_profile.json",
    [string]$ReportsDir = "reports",
    [string]$EventsJsonl = "",
    [string]$ReportDate = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

function Resolve-RepoPath {
    param([string]$PathValue)
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path $repoRoot $PathValue)
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$CmdArgs
    )
    Add-Content -Path $script:logPath -Value "=== $Name ==="
    Write-Output "[PIPELINE] $Name"
    Add-Content -Path $script:logPath -Value ("CMD: python " + ($CmdArgs -join " "))
    $output = (& python @CmdArgs 2>&1 | Out-String)
    $rc = $LASTEXITCODE
    if (-not [string]::IsNullOrWhiteSpace($output)) {
        Add-Content -Path $script:logPath -Value $output.TrimEnd()
    }
    Add-Content -Path $script:logPath -Value "RC=$rc"
    if ($rc -ne 0) {
        Add-Content -Path $script:logPath -Value "FAILED rc=$rc"
        Write-Output "[PIPELINE] $Name failed rc=$rc"
        $script:hadFailure = $true
    }
    $script:stepRc[$Name] = $rc
}

function Copy-LatestMatch {
    param(
        [string]$SourceDir,
        [string]$Pattern,
        [string]$TargetPath
    )
    $latest = Get-ChildItem -Path $SourceDir -Filter $Pattern -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($latest) {
        Copy-Item -Path $latest.FullName -Destination $TargetPath -Force
    }
}

$manifestPath = Resolve-RepoPath $Manifest
$profilePath = Resolve-RepoPath $Profile
$reportsPath = Resolve-RepoPath $ReportsDir
if ([string]::IsNullOrWhiteSpace($ReportDate)) {
    $ReportDate = (Get-Date).ToString("yyyy-MM-dd")
}
$dailyReportDir = Join-Path $reportsPath $ReportDate

$manifestObj = $null
if (Test-Path $manifestPath) {
    $manifestObj = Get-Content -Raw $manifestPath | ConvertFrom-Json
}
$profileObj = $null
if (Test-Path $profilePath) {
    $profileObj = Get-Content -Raw $profilePath | ConvertFrom-Json
}

if ([string]::IsNullOrWhiteSpace($EventsJsonl)) {
    if ($manifestObj -and $manifestObj.event_contract -and $manifestObj.event_contract.events_file) {
        $EventsJsonl = [string]$manifestObj.event_contract.events_file
    } else {
        $EventsJsonl = "events.jsonl"
    }
}
$eventsPath = Resolve-RepoPath $EventsJsonl

$canonicalWindowStart = ""
if ($profileObj -and $profileObj.reporting -and $profileObj.reporting.canonical_window_start_utc) {
    $canonicalWindowStart = [string]$profileObj.reporting.canonical_window_start_utc
} elseif ($manifestObj -and $manifestObj.event_contract -and $manifestObj.event_contract.canonical_window_start_utc) {
    $canonicalWindowStart = [string]$manifestObj.event_contract.canonical_window_start_utc
}

$dailySummaryRel = "daily_summary.json"
if ($manifestObj -and $manifestObj.event_contract -and $manifestObj.event_contract.daily_summary_file) {
    $dailySummaryRel = [string]$manifestObj.event_contract.daily_summary_file
}
$dailySummaryCanonicalPath = Resolve-RepoPath $dailySummaryRel
$dailySummaryArtifactPath = Join-Path $dailyReportDir "daily_summary.json"

New-Item -ItemType Directory -Path $reportsPath -Force | Out-Null
New-Item -ItemType Directory -Path $dailyReportDir -Force | Out-Null
$pipelineLogDir = Join-Path $reportsPath "pipeline_logs"
New-Item -ItemType Directory -Path $pipelineLogDir -Force | Out-Null

if (-not (Test-Path $eventsPath)) {
    New-Item -ItemType File -Path $eventsPath -Force | Out-Null
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$script:logPath = Join-Path $pipelineLogDir "strategy_1_daily_pipeline_$stamp.log"
$script:hadFailure = $false
$script:stepRc = @{}

if ([string]::IsNullOrWhiteSpace($canonicalWindowStart)) {
    $dailySummaryArgs = @(
        "scripts/build_daily_summary.py",
        "--events-jsonl", $eventsPath,
        "--manifest", $manifestPath,
        "--out", $dailySummaryArtifactPath,
        "--date", $ReportDate
    )
} else {
    $dailySummaryArgs = @(
        "scripts/build_daily_summary.py",
        "--events-jsonl", $eventsPath,
        "--manifest", $manifestPath,
        "--window-start", $canonicalWindowStart,
        "--out", $dailySummaryArtifactPath,
        "--date", $ReportDate
    )
}
Invoke-Step -Name "daily_summary" -CmdArgs $dailySummaryArgs
if (($script:stepRc["daily_summary"] -eq 0) -and (Test-Path $dailySummaryArtifactPath)) {
    Copy-Item -Path $dailySummaryArtifactPath -Destination $dailySummaryCanonicalPath -Force
}

if ([string]::IsNullOrWhiteSpace($canonicalWindowStart)) {
    $paperReportArgs = @(
        "scripts/paper_trading_mode_report.py",
        "--events-jsonl", $eventsPath,
        "--manifest", $manifestPath,
        "--profile", $profilePath,
        "--reports-dir", $dailyReportDir,
        "--out-prefix", "paper_report"
    )
} else {
    $paperReportArgs = @(
        "scripts/paper_trading_mode_report.py",
        "--events-jsonl", $eventsPath,
        "--manifest", $manifestPath,
        "--profile", $profilePath,
        "--window-start", $canonicalWindowStart,
        "--reports-dir", $dailyReportDir,
        "--out-prefix", "paper_report"
    )
}
Invoke-Step -Name "paper_report" -CmdArgs $paperReportArgs
if ($script:stepRc["paper_report"] -eq 0) {
    Copy-LatestMatch -SourceDir $dailyReportDir -Pattern "paper_report_*.json" -TargetPath (Join-Path $dailyReportDir "paper_report.json")
    Copy-LatestMatch -SourceDir $dailyReportDir -Pattern "paper_report_*.md" -TargetPath (Join-Path $dailyReportDir "paper_report.md")
}

Invoke-Step -Name "health_report" -CmdArgs @(
    "scripts/daily_health_report.py",
    "--manifest", $manifestPath,
    "--profile", $profilePath,
    "--reports-dir", $dailyReportDir,
    "--paper-prefix", "paper_report",
    "--out-prefix", "daily_health",
    "--daily-summary-json", $dailySummaryArtifactPath,
    "--events-jsonl", $eventsPath
)
if ($script:stepRc["health_report"] -eq 0) {
    Copy-LatestMatch -SourceDir $dailyReportDir -Pattern "daily_health_*.json" -TargetPath (Join-Path $dailyReportDir "daily_health.json")
    Copy-LatestMatch -SourceDir $dailyReportDir -Pattern "daily_health_*.md" -TargetPath (Join-Path $dailyReportDir "daily_health.md")
}

Copy-Item -Path $script:logPath -Destination (Join-Path $dailyReportDir "daily_pipeline.log") -Force

Write-Output "[PIPELINE] log=$script:logPath"
Write-Output "[PIPELINE] daily_artifacts=$dailyReportDir"
if ($script:hadFailure) {
    exit 1
}
exit 0
