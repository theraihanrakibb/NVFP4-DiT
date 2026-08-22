@echo off
REM Build script for NVFP4-DiT (IEEE TNNLS submission)
REM Requires MiKTeX / TeX Live on PATH.
set MAIN=NVFP4-DiT_TNNLS
set APP=appendix_TNNLS
echo Building main manuscript...
pdflatex -interaction=nonstopmode %MAIN%.tex
pdflatex -interaction=nonstopmode %MAIN%.tex
echo Building appendix...
pdflatex -interaction=nonstopmode %APP%.tex
pdflatex -interaction=nonstopmode %APP%.tex
echo Done. Outputs: %MAIN%.pdf and %APP%.pdf
