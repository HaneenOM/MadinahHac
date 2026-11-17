$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONPATH = (Join-Path $projectRoot '.vendor')
$scriptPath = Join-Path $projectRoot 'src\hidden_future_benchmark.py'
$pythonPath = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python 3.12 was not found at $pythonPath"
}
& $pythonPath $scriptPath
exit $LASTEXITCODE
