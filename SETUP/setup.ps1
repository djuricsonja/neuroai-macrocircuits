<#
.SYNOPSIS
    Sets up the project environment with uv (Windows).

.DESCRIPTION
    Installs uv if it is missing, creates .venv with the Python version pinned in
    .python-version, and installs the locked dependencies from requirements.lock.
    Safe to re-run: it converges the environment to the lock file.

.EXAMPLE
    .\SETUP\setup.ps1
    .\SETUP\setup.ps1 -Cuda
#>
param(
    [switch]$Cuda  # Install a CUDA build of PyTorch instead of the CPU build.
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# uv reports progress on stderr. Under Windows PowerShell 5.1 an $ErrorActionPreference
# of 'Stop' turns that into a terminating NativeCommandError, so run uv with the
# preference relaxed and decide success from the exit code instead.
function Invoke-Uv {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$UvArgs)

    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & uv @UvArgs
        if ($LASTEXITCODE -ne 0) {
            throw "uv $($UvArgs -join ' ') failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

$repo = Split-Path -Parent $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host 'uv not found - installing it...' -ForegroundColor Yellow
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    # The installer only updates PATH for future shells, so extend this one.
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'uv installed but is still not on PATH. Open a new terminal and re-run this script.'
    }
}
Write-Host "Using $(Invoke-Uv --version)" -ForegroundColor Green

Push-Location $repo
try {
    # uv venv reads .python-version and downloads that interpreter if it is missing.
    # It refuses to write into an existing directory unless told how to treat it, so
    # reuse a healthy .venv and rebuild one left half-written by an interrupted run.
    $venv = Join-Path $repo '.venv'
    $venvPython = Join-Path $venv 'Scripts\python.exe'
    if ((Test-Path $venv) -and -not (Test-Path $venvPython)) {
        Write-Host 'Existing .venv is incomplete - recreating it...' -ForegroundColor Yellow
        Invoke-Uv venv --clear
    }
    else {
        Invoke-Uv venv --allow-existing
    }

    Invoke-Uv pip sync SETUP\requirements.lock

    if ($Cuda) {
        Write-Host 'Installing a CUDA build of PyTorch...' -ForegroundColor Yellow
        Invoke-Uv pip install torch --torch-backend=auto
    }

    Write-Host ''
    Write-Host 'Done. Activate with:  .\.venv\Scripts\Activate.ps1' -ForegroundColor Green
    Write-Host 'In VS Code, select .venv as the notebook kernel.' -ForegroundColor Green
}
finally {
    Pop-Location
}
