param(
    [switch]$Force,
    [switch]$SkipPackages,
    [switch]$InstallDev
)

$projectRoot = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
Set-Location $projectRoot

Write-Host "Project root: $projectRoot"

function Get-Python312Executable {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        try {
            $path = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
            if ($path) { return $path.Trim() }
        } catch {
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        try {
            $version = & $python.Source -c "import sys; print(sys.version.split()[0])" 2>$null
            if ($version -and $version.Trim().StartsWith('3.12')) {
                return $python.Source
            }
        } catch {
        }
    }

    return $null
}

$pythonPath = Get-Python312Executable
if (-not $pythonPath) {
    Write-Error "Python 3.12 was not found. Install Python 3.12 and ensure `py -3.12` or `python` points to it."
    exit 1
}

Write-Host "Python 3.12 executable: $pythonPath"

$venvDir = Join-Path $projectRoot ".venv"
if (-not (Test-Path $venvDir) -or $Force) {
    Write-Host "Creating virtual environment at $venvDir"
    & $pythonPath -m venv $venvDir
}

$venvPython = Join-Path $venvDir "Scripts\python.exe"
$venvPip = Join-Path $venvDir "Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment Python not found at $venvPython"
    exit 1
}

Write-Host "Using venv Python: $venvPython"
& $venvPython --version

Write-Host "Upgrading pip, setuptools, and wheel..."
& $venvPython -m pip install --upgrade pip setuptools wheel --no-cache-dir

if (-not $SkipPackages) {
    if (-not (Test-Path "requirements.txt")) {
        Write-Error "requirements.txt not found in project root."
        exit 1
    }

    Write-Host "Purging pip cache to reduce disk usage..."
    & $venvPython -m pip cache purge | Out-Null

    Write-Host "Installing requirements from requirements.txt"
    & $venvPip install --no-cache-dir -r requirements.txt

    if ($InstallDev) {
        if (-not (Test-Path "requirements-dev.txt")) {
            Write-Error "requirements-dev.txt not found in project root."
            exit 1
        }

        Write-Host "Installing developer dependencies from requirements-dev.txt"
        & $venvPip install --no-cache-dir -r requirements-dev.txt
    }
}

Write-Host "Checking SUMO installation..."
$sumo = Get-Command sumo -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue
$sumoGui = Get-Command sumo-gui -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue

if ($sumo) {
    Write-Host "sumo found: $sumo"
    & $sumo --version
} else {
    Write-Warning "sumo executable not found in PATH."
}

if ($sumoGui) {
    Write-Host "sumo-gui found: $sumoGui"
    & $sumoGui --version
} else {
    Write-Warning "sumo-gui executable not found in PATH."
}

if (-not $env:SUMO_HOME) {
    Write-Warning "SUMO_HOME is not set. Set SUMO_HOME to your SUMO install folder to make SUMO tools available to Python imports."
} else {
    Write-Host "SUMO_HOME=$env:SUMO_HOME"
}

Write-Host "Bootstrap complete. Activate the venv with `.\.venv\Scripts\Activate.ps1` and use `python` from the venv."