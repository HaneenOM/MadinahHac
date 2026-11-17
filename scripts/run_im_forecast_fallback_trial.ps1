$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONPATH = $projectRoot + [IO.Path]::PathSeparator + (Join-Path $projectRoot '.vendor')
$pythonPath = if ($env:MADINAH_PYTHON) { $env:MADINAH_PYTHON } else {
    $projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $projectPython) { $projectPython } else {
        Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    }
}
& $pythonPath (Join-Path $projectRoot 'src\im_forecast_fallback_trial.py') @args
exit $LASTEXITCODE
