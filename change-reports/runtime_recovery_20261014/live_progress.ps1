param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('E1', 'E2')]
    [string]$Study,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$launch = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'production_launch.json') -Raw | ConvertFrom-Json
$studyInfo = if ($Study -eq 'E1') { $launch.e1 } else { $launch.e2 }
$batch = $studyInfo.batch
$offsets = @{}
$lastWorkerLog = ''
$host.UI.RawUI.WindowTitle = "Polymer $Study - Live progress"
Write-Host "Polymer $Study - progress from the existing simulation" -ForegroundColor Cyan
Write-Host 'Closing this viewer leaves the simulations running.'
Write-Host "Results: $batch"
Write-Host ''

function Show-NewLogText([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $stream = $null
    $reader = $null
    try {
        $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        $first = -not $offsets.ContainsKey($Path)
        $offset = if ($first) { 0 } else { [long]$offsets[$Path] }
        if ($offset -gt $stream.Length) { $offset = 0; $first = $true }
        [void]$stream.Seek($offset, [System.IO.SeekOrigin]::Begin)
        $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
        $text = $reader.ReadToEnd()
        $offsets[$Path] = $stream.Position
        if ($text.Length -gt 0) {
            if ($first) {
                ($text.TrimEnd() -split '\r?\n' | Select-Object -Last 10) | ForEach-Object { Write-Host $_ }
            } else {
                Write-Host $text -NoNewline
            }
        }
    } finally {
        if ($reader) { $reader.Dispose() }
        elseif ($stream) { $stream.Dispose() }
    }
}

while ($true) {
    try {
        Show-NewLogText $studyInfo.log
        Show-NewLogText ([System.IO.Path]::ChangeExtension($studyInfo.log, '.err'))
        $workerLog = $null
        $finished = $false
        if ($Study -eq 'E1') {
            $manifest = Get-Content -LiteralPath (Join-Path $batch 'manifest.json') -Raw | ConvertFrom-Json
            $job = $manifest.jobs | Where-Object { $_.state -eq 'running' } | Select-Object -First 1
            if ($job -and $job.attempts.Count -gt 0) {
                $workerLog = Join-Path (Join-Path $batch ($job.attempts[-1])) 'worker.log'
            }
            $finished = @($manifest.jobs | Where-Object { $_.state -ne 'completed' }).Count -eq 0
        } else {
            $manifest = Get-Content -LiteralPath (Join-Path $batch 'manifest.json') -Raw | ConvertFrom-Json
            foreach ($method in $manifest.methods) {
                $methodDirectory = Join-Path $batch $method
                $statusPath = Join-Path $methodDirectory 'status.json'
                if (Test-Path -LiteralPath $statusPath) {
                    $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
                    if ($status.state -eq 'running') { $workerLog = Join-Path $methodDirectory 'run.log'; break }
                }
            }
            $batchStatus = Join-Path $batch 'status.json'
            if (Test-Path -LiteralPath $batchStatus) {
                $status = Get-Content -LiteralPath $batchStatus -Raw | ConvertFrom-Json
                $finished = $status.state -eq 'completed'
            }
        }
        if ($workerLog) {
            if ($workerLog -ne $lastWorkerLog) {
                Write-Host "`nFollowing: $workerLog" -ForegroundColor Green
                $lastWorkerLog = $workerLog
            }
            Show-NewLogText $workerLog
        }
        if ($finished) { Write-Host "`n$Study completed." -ForegroundColor Green; break }
        if ($Once) { break }
        $parent = Get-Process -Id $studyInfo.parent_pid -ErrorAction SilentlyContinue
        if (-not $parent) {
            Show-NewLogText $studyInfo.log
            Show-NewLogText ([System.IO.Path]::ChangeExtension($studyInfo.log, '.err'))
            Write-Host "`nThe study launcher has exited. Check the last messages and saved status." -ForegroundColor Yellow
            break
        }
    } catch {
        Write-Host "Waiting for the next log/status update: $($_.Exception.Message)" -ForegroundColor DarkYellow
        if ($Once) { throw }
    }
    Start-Sleep -Seconds 2
}
