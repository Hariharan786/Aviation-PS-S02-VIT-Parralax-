
$ErrorActionPreference="Stop"
Set-Location $PSScriptRoot
if (!(Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
