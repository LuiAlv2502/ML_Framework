$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$docs = Join-Path $root "docs"
$log = Join-Path $docs "technical_report-build.log"
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { (Get-Command python -ErrorAction Stop).Source }
$diagramScript = Join-Path $root "scripts\generate_report_diagrams.py"
& $python $diagramScript
if ($LASTEXITCODE -ne 0) {
    throw "No se pudieron generar los diagramas del informe."
}

$command = Get-Command pdflatex -ErrorAction SilentlyContinue
$localMiktex = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
if ($command) {
    $pdflatex = $command.Source
} elseif (Test-Path $localMiktex) {
    $pdflatex = $localMiktex
} else {
    throw "No se encontró pdflatex en PATH ni en la instalación estándar de MiKTeX."
}
$bibtex = Join-Path (Split-Path $pdflatex) "bibtex.exe"
if (-not (Test-Path $bibtex)) {
    throw "No se encontró bibtex.exe junto a pdflatex. Instale una distribución TeX completa."
}

Push-Location $docs
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    Set-Content -Path $log -Value "" -Encoding utf8
    foreach ($pass in 1, 2, 3) {
        $output = & $pdflatex --enable-installer -interaction=nonstopmode -halt-on-error -file-line-error technical_report.tex 2>&1
        $exitCode = $LASTEXITCODE
        $output | Out-File -FilePath $log -Append -Encoding utf8
        if ($exitCode -ne 0) {
            throw "pdflatex falló en la pasada $pass; consulte $log."
        }
        if ($pass -eq 1) {
            $output = & $bibtex technical_report 2>&1
            $exitCode = $LASTEXITCODE
            $output | Out-File -FilePath $log -Append -Encoding utf8
            if ($exitCode -ne 0) {
                throw "bibtex falló; consulte $log."
            }
        }
    }

    $pdf = Join-Path $docs "technical_report.pdf"
    if (-not (Test-Path $pdf) -or (Get-Item $pdf).Length -eq 0) {
        throw "La compilación terminó sin crear un PDF válido."
    }
    Write-Output "PDF generado: $pdf"
}
finally {
    $ErrorActionPreference = $previousErrorActionPreference
    Pop-Location
}
