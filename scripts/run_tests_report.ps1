$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$evidence = Join-Path $root "docs\evidencias"
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { (Get-Command python -ErrorAction Stop).Source }
New-Item -ItemType Directory -Force -Path $evidence | Out-Null
Push-Location $root
try {
    $pytestPath = Join-Path $evidence "pytest_output.txt"
    $coveragePath = Join-Path $evidence "coverage.txt"

    $pytestOutput = & $python -m pytest -q 2>&1
    $pytestExit = $LASTEXITCODE
    $pytestOutput | Out-File -FilePath $pytestPath -Encoding utf8
    if ($pytestExit -ne 0) {
        throw "pytest falló con código $pytestExit. Revisa $pytestPath."
    }

    $coverageOutput = & $python -m pytest --cov=framework_ia --cov-report=term-missing -q 2>&1
    $coverageExit = $LASTEXITCODE
    $coverageOutput | Out-File -FilePath $coveragePath -Encoding utf8
    if ($coverageExit -ne 0) {
        throw "La ejecución de cobertura falló con código $coverageExit. Revisa $coveragePath."
    }
}
finally {
    Pop-Location
}
