# One-click LaTeX build script for paper
Write-Host "Compiling main.tex..." -ForegroundColor Cyan

pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nCompilation Successful! Generated main.pdf" -ForegroundColor Green
} else {
    Write-Host "`nCompilation Failed. Check main.log for details." -ForegroundColor Red
}

