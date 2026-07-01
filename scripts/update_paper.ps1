# VFund — weekly paper-account update (diversified 3-sleeve book).
# Fetches the broad universe, the DeFi price panel, and on-chain TVL, then marks
# the paper account forward. Registered as a weekly Scheduled Task by
# scripts/install_task.ps1.
#
# Run manually any time:  powershell -File scripts\update_paper.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\vfund.exe"
$Log = Join-Path $Root "data\paper_log.txt"
$Today = Get-Date -Format "yyyy-MM-dd"
Set-Location $Root

$defi = "AAVEUSDT","UNIUSDT","CRVUSDT","COMPUSDT","SUSHIUSDT","LDOUSDT","GMXUSDT",
        "PENDLEUSDT","CAKEUSDT","DYDXUSDT","1INCHUSDT","BALUSDT","YFIUSDT","RUNEUSDT",
        "JOEUSDT","LQTYUSDT","SPELLUSDT","CVXUSDT","RPLUSDT","STGUSDT","ENAUSDT",
        "ETHFIUSDT","JTOUSDT","JUPUSDT","RAYUSDT"

"[$([DateTime]::Now)] fetching data..." | Tee-Object -FilePath $Log -Append
& $Py fetch-universe --top 60 --interval 1d --start 2021-01-01 --end $Today --out data\live.parquet *>> $Log
& $Py fetch-universe --symbols @defi --interval 1d --start 2021-01-01 --end $Today --out data\live_defi.parquet *>> $Log
& $Py fetch-tvl --start 2021-01-01 --end $Today --out data\live_tvl.parquet *>> $Log

& $Py paper --three-sleeve --data data\live.parquet --defi-data data\live_defi.parquet `
    --tvl-data data\live_tvl.parquet --state data\paper.json --start-equity 100000 *>> $Log

"[$([DateTime]::Now)] done." | Tee-Object -FilePath $Log -Append
