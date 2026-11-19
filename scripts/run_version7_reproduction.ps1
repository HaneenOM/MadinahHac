# Preserve previous outputs, then run the original notebook with its accepted checks intact.
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv_lab/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Create .venv_lab with Python 3.12 and install requirements_version7.txt first.' }
$previousLocation = Get-Location
$savedEnvironment = @{}
foreach ($name in @('HF_HOME', 'HF_HUB_OFFLINE', 'PYTHONUTF8', 'MADINAH_PYTHON')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    Set-Location -LiteralPath $taskRoot
    $archive = Join-Path $taskRoot ('outputs/version7_runs/before_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    $versionOutput = Join-Path $taskRoot 'outputs/version7'
    # Cache and reference are read-only inputs to the run; all generated artifacts are backed up.
    if (Test-Path -LiteralPath $versionOutput) {
        $archiveVersion = Join-Path $archive 'version7'
        New-Item -ItemType Directory -Path $archiveVersion -Force | Out-Null
        Get-ChildItem -LiteralPath $versionOutput -Force |
            Where-Object { $_.Name -notin @('huggingface', 'reference') } |
            ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $archiveVersion -Recurse -Force }
    }
    $intermediates = Join-Path $taskRoot 'outputs/noran'
    if (Test-Path -LiteralPath $intermediates) {
        New-Item -ItemType Directory -Path $archive -Force | Out-Null
        Copy-Item -LiteralPath $intermediates -Destination (Join-Path $archive 'noran') -Recurse
    }
    Write-Host "Previous generated outputs preserved under $archive"
    $env:HF_HOME = Join-Path $versionOutput 'huggingface'
    $env:HF_HUB_OFFLINE = '1'
    $env:PYTHONUTF8 = '1'
    $env:MADINAH_PYTHON = $taskPython
    & $taskPython (Join-Path $taskRoot 'scripts/run_version7_notebook.py')
    if ($LASTEXITCODE -ne 0) { throw "Version 7 runner failed (exit $LASTEXITCODE). Inspect outputs/version7/verification.json and logs; accepted verification remains enabled." }
    & $taskPython (Join-Path $versionOutput 'verify_reproduction.py')
    if ($LASTEXITCODE -ne 0) { throw 'Independent accepted-reference verification failed.' }
}
finally {
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
    Set-Location -LiteralPath $previousLocation.Path
}
