$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONPATH = $projectRoot + [IO.Path]::PathSeparator + (Join-Path $projectRoot '.vendor')
$pythonPath = if ($env:MADINAH_PYTHON) { $env:MADINAH_PYTHON } else {
    $projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $projectPython) { $projectPython } else {
        Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    }
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python 3.12 was not found at $pythonPath"
}
& $pythonPath -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('torch') else 1)"
if ($LASTEXITCODE -ne 0 -and -not $env:MADINAH_PYTHON) {
    $teammatePython = 'C:\Users\96655\Downloads\T05_v2_plus_new_forecast\T05_v2_plus_new_forecast\.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $teammatePython) { $pythonPath = $teammatePython }
}
& $pythonPath (Join-Path $PSScriptRoot 'build_t05_integrated.py') @args
exit $LASTEXITCODE
