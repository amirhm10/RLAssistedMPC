param(
    [string]$PythonPath = "",
    [string]$OutputRoot = ".detached-notebook-runs",
    [string]$KernelName = "rl-env",
    [string]$RunName = "distillation_markov_ls_only",
    [switch]$NoLaunch,
    [switch]$VisibleWindow
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$launcherPath = Join-Path $PSScriptRoot "start_detached_notebook_run.ps1"

& $launcherPath `
    -NotebookPath "distillation_RL_assisted_MPC_markov_ls_only_unified.ipynb" `
    -OutputRoot $OutputRoot `
    -PythonPath $PythonPath `
    -KernelName $KernelName `
    -RunName $RunName `
    -NoLaunch:$NoLaunch `
    -VisibleWindow:$VisibleWindow
