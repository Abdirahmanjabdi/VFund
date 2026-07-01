# VFund — weekly paper-account update.
# Fetches a fresh broad universe from Binance and marks the paper account forward.
# Registered as a Windows Scheduled Task by scripts/install_task.ps1 (weekly).
#
# Run manually any time:  powershell -File scripts\update_paper.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot          # project root
$Py = Join-Path $Root ".venv\Scripts\vfund.exe"
$Data = Join-Path $Root "data\live.parquet"
$State = Join-Path $Root "data\paper.json"
$Log = Join-Path $Root "data\paper_log.txt"
$Today = Get-Date -Format "yyyy-MM-dd"

Set-Location $Root
"[$([DateTime]::Now)] fetching universe..." | Tee-Object -FilePath $Log -Append

# Full history each time keeps the signal's lookbacks intact (small daily data).
& $Py fetch-universe --top 60 --interval 1d --start 2021-01-01 --end $Today --out $Data *>> $Log
& $Py paper --data $Data --state $State --start-equity 100000 *>> $Log

"[$([DateTime]::Now)] done." | Tee-Object -FilePath $Log -Append
