param(
    [Parameter(Mandatory = $true)]
    [string]$ExePath,

    [Parameter(Mandatory = $true)]
    [string]$VideoPath,

    [int]$Runs = 1,
    [int]$HealthTimeoutSeconds = 60,
    [int]$JobTimeoutSeconds = 900
)

# smoke_test_artifact version: v2-curl-upload
$ErrorActionPreference = "Stop"

function Wait-ForHealth {
    param([int]$TimeoutSeconds)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 5
            if ($resp.status -eq "ok") {
                return $true
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

function Wait-ForJobDone {
    param(
        [string]$JobId,
        [int]$TimeoutSeconds
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $status = Invoke-RestMethod -Uri ("http://127.0.0.1:8000/jobs/{0}/status" -f $JobId) -Method Get -TimeoutSec 10
        if ($status.status -eq "done") {
            return $status
        }
        if ($status.status -eq "error") {
            throw "Job failed: $($status.error)"
        }
        Start-Sleep -Seconds 2
    }
    throw "Timed out waiting for job completion for $JobId"
}

function Start-AnalyzeJob {
    param([string]$VideoFile)
    $curlOutput = & curl.exe -sS -i -X POST -F "video=@$VideoFile" -F "preset=balanced" "http://127.0.0.1:8000/analyze"
    if ($LASTEXITCODE -ne 0) {
        throw "curl analyze request failed with exit code $LASTEXITCODE"
    }
    if ($curlOutput -notmatch "HTTP\/1\.[01]\s+(302|303)") {
        throw "Analyze returned unexpected response: $curlOutput"
    }
    if ($curlOutput -notmatch "(?im)^location:\s*(.+)\s*$") {
        throw "Analyze response missing location header: $curlOutput"
    }
    $locationText = $Matches[1].Trim()
    if ($locationText -notmatch "/jobs/([^/]+)$") {
        throw "Cannot parse job id from location: $locationText"
    }
    return $Matches[1]
}

function Assert-OutputFiles {
    param([string]$OutputRoot)
    $jobsDir = Join-Path $OutputRoot "outputs\jobs"
    if (!(Test-Path $jobsDir)) {
        throw "Missing jobs output directory: $jobsDir"
    }
    $jobDirs = Get-ChildItem -Path $jobsDir -Directory | Sort-Object LastWriteTime -Descending
    if ($jobDirs.Count -lt 1) {
        throw "No job directory produced in $jobsDir"
    }
    $jobDir = $jobDirs[0].FullName
    $required = @("summary.json", "events.csv", "report.html")
    foreach ($name in $required) {
        $full = Join-Path $jobDir $name
        if (!(Test-Path $full)) {
            throw "Missing output file: $full"
        }
        if ((Get-Item $full).Length -le 0) {
            throw "Output file is empty: $full"
        }
    }
    $summaryPath = Join-Path $jobDir "summary.json"
    $summary = Get-Content -Path $summaryPath -Raw | ConvertFrom-Json
    if ($null -eq $summary.total_crossings) {
        throw "summary.json missing total_crossings"
    }
}

function Assert-LogsClean {
    param([string]$OutputRoot)
    $logPaths = @(
        (Join-Path $OutputRoot "people-counter.log"),
        (Join-Path $OutputRoot "people-counter-launcher.log")
    )
    $patterns = @("WinError 2", "Traceback", "Unhandled exception")
    foreach ($logPath in $logPaths) {
        if (!(Test-Path $logPath)) {
            continue
        }
        $text = Get-Content -Path $logPath -Raw
        foreach ($p in $patterns) {
            if ($text -match [regex]::Escape($p)) {
                throw "Critical error marker '$p' found in $logPath"
            }
        }
    }
}

$resolvedExe = (Resolve-Path $ExePath).Path
$resolvedVideo = (Resolve-Path $VideoPath).Path

for ($run = 1; $run -le $Runs; $run++) {
    Write-Host "=== Smoke test run $run/$Runs ==="
    $workDir = Join-Path $env:TEMP ("people_counter_smoke_" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $workDir | Out-Null
    $runExePath = Join-Path $workDir "people-counter.exe"
    Copy-Item -Path $resolvedExe -Destination $runExePath -Force

    try {
        $proc = Start-Process -FilePath $runExePath -WorkingDirectory $workDir -PassThru
        if (-not (Wait-ForHealth -TimeoutSeconds $HealthTimeoutSeconds)) {
            throw "Server health endpoint did not become ready in time"
        }

        $jobId = Start-AnalyzeJob -VideoFile $resolvedVideo
        [void](Wait-ForJobDone -JobId $jobId -TimeoutSeconds $JobTimeoutSeconds)

        Assert-OutputFiles -OutputRoot $workDir
        Assert-LogsClean -OutputRoot $workDir
        Write-Host "Run $run passed."
    } finally {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force
            Start-Sleep -Milliseconds 400
        }
    }
}

Write-Host "All smoke test runs passed."
