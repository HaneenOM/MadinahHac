$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:PYTHONPATH = Join-Path $projectRoot '.vendor'
$pythonPath = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python 3.12 was not found at $pythonPath"
}
& $pythonPath (Join-Path $projectRoot 'src\local_score.py')
exit $LASTEXITCODE
