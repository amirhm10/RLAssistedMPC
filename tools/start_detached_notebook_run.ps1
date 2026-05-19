param(
    [string]$NotebookPath,
    [string]$RepoRoot = "",
    [string]$OutputRoot = ".detached-notebook-runs",
    [string]$PythonPath = "",
    [string]$KernelName = "rl-env",
    [string]$RunName = "",
    [switch]$NoLaunch,
    [switch]$VisibleWindow
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    param([string]$ExplicitRepoRoot)

    if ($ExplicitRepoRoot) {
        return (Resolve-Path -LiteralPath $ExplicitRepoRoot).Path
    }

    return (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
}

function Resolve-NotebookPath {
    param(
        [string]$NotebookArg,
        [string]$ResolvedRepoRoot
    )

    if (-not $NotebookArg) {
        throw "NotebookPath is required."
    }

    if ([System.IO.Path]::IsPathRooted($NotebookArg)) {
        return (Resolve-Path -LiteralPath $NotebookArg).Path
    }

    return (Resolve-Path -LiteralPath (Join-Path $ResolvedRepoRoot $NotebookArg)).Path
}

function Resolve-OutputRoot {
    param(
        [string]$OutputRootArg,
        [string]$ResolvedRepoRoot
    )

    if ([System.IO.Path]::IsPathRooted($OutputRootArg)) {
        return $OutputRootArg
    }

    return (Join-Path $ResolvedRepoRoot $OutputRootArg)
}

function Resolve-PythonExecutable {
    param([string]$ExplicitPythonPath)

    $candidates = New-Object System.Collections.Generic.List[string]
    if ($ExplicitPythonPath) {
        $candidates.Add($ExplicitPythonPath)
    }

    $userProfile = [Environment]::GetFolderPath("UserProfile")
    $candidates.Add((Join-Path $userProfile ".conda\envs\rl-env\python.exe"))
    $candidates.Add((Join-Path $userProfile "miniconda3\envs\rl-env\python.exe"))

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCmd) {
        return $pythonCmd.Source
    }

    throw "Could not find a Python executable. Pass -PythonPath explicitly."
}

$resolvedRepoRoot = Resolve-RepoRoot -ExplicitRepoRoot $RepoRoot
$resolvedNotebookPath = Resolve-NotebookPath -NotebookArg $NotebookPath -ResolvedRepoRoot $resolvedRepoRoot
$resolvedOutputRoot = Resolve-OutputRoot -OutputRootArg $OutputRoot -ResolvedRepoRoot $resolvedRepoRoot
$resolvedPythonPath = Resolve-PythonExecutable -ExplicitPythonPath $PythonPath

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$notebookStem = [System.IO.Path]::GetFileNameWithoutExtension($resolvedNotebookPath)
$runStem = if ($RunName) { $RunName } else { $notebookStem }
$safeRunStem = ($runStem -replace "[^A-Za-z0-9_.-]", "_")
$runDirName = "{0}_{1}" -f $timestamp, $safeRunStem
$runDir = Join-Path $resolvedOutputRoot $runDirName

New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$outputNotebookName = "{0}.executed.ipynb" -f $notebookStem
$stdoutLogPath = Join-Path $runDir "stdout.log"
$stderrLogPath = Join-Path $runDir "stderr.log"
$manifestPath = Join-Path $runDir "run_manifest.json"

$argumentList = @(
    "-m",
    "jupyter",
    "nbconvert",
    "--to",
    "notebook",
    "--execute",
    "--ExecutePreprocessor.timeout=-1",
    "--ExecutePreprocessor.kernel_name=$KernelName",
    "--ExecutePreprocessor.allow_errors=False",
    "--output",
    $outputNotebookName,
    "--output-dir",
    $runDir,
    $resolvedNotebookPath
)

$manifest = [ordered]@{
    repo_root = $resolvedRepoRoot
    notebook_path = $resolvedNotebookPath
    run_dir = $runDir
    output_notebook = (Join-Path $runDir $outputNotebookName)
    stdout_log = $stdoutLogPath
    stderr_log = $stderrLogPath
    python_path = $resolvedPythonPath
    kernel_name = $KernelName
    created_at = (Get-Date).ToString("o")
    launched = $false
    pid = $null
    command = @($resolvedPythonPath) + $argumentList
}

if ($NoLaunch) {
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath
    Write-Host "Detached notebook run prepared but not launched."
    Write-Host "Notebook   : $resolvedNotebookPath"
    Write-Host "Run dir    : $runDir"
    Write-Host "Python     : $resolvedPythonPath"
    Write-Host "Kernel     : $KernelName"
    Write-Host "Stdout log : $stdoutLogPath"
    Write-Host "Stderr log : $stderrLogPath"
    Write-Host "Manifest   : $manifestPath"
    return
}

$windowStyle = if ($VisibleWindow) { "Normal" } else { "Hidden" }
$process = Start-Process `
    -FilePath $resolvedPythonPath `
    -ArgumentList $argumentList `
    -WorkingDirectory $resolvedRepoRoot `
    -RedirectStandardOutput $stdoutLogPath `
    -RedirectStandardError $stderrLogPath `
    -WindowStyle $windowStyle `
    -PassThru

$manifest.launched = $true
$manifest.pid = $process.Id
$manifest.started_at = (Get-Date).ToString("o")
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath

Write-Host "Detached notebook run launched."
Write-Host "PID        : $($process.Id)"
Write-Host "Notebook   : $resolvedNotebookPath"
Write-Host "Run dir    : $runDir"
Write-Host "Stdout log : $stdoutLogPath"
Write-Host "Stderr log : $stderrLogPath"
Write-Host "Manifest   : $manifestPath"
