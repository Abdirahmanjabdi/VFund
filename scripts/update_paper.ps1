# VFund - weekly paper-account update.
#
# Marks TWO forward accounts:
#   data/paper.json             the alpha engine alone (trend + size + on-chain).
#                               This is the original, unbroken forward record -
#                               it is NOT superseded, it is one half of the book
#                               below, and its history is worth preserving intact.
#   data/paper_two_engine.json  the full two-engine book: 4-sleeve alpha (adds
#                               the on-chain fees sleeve) + funding carry, 50/50
#                               capital. This is the leading candidate.
#
# Running both is deliberate: the pair isolates what the carry engine actually
# contributes forward, instead of leaving it to a backtest to claim.
#
# Registered as a weekly Scheduled Task by scripts/install_task.ps1.
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

# Carry trades liquid majors only - see vfund/live/carry.py MAJORS.
$majors = "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT",
          "AVAXUSDT","LINKUSDT","DOTUSDT","LTCUSDT","TRXUSDT","BCHUSDT"

"[$([DateTime]::Now)] fetching data..." | Tee-Object -FilePath $Log -Append
& $Py fetch-universe --top 60 --interval 1d --start 2021-01-01 --end $Today --out data\live.parquet *>> $Log
& $Py fetch-universe --symbols @defi --interval 1d --start 2021-01-01 --end $Today --out data\live_defi.parquet *>> $Log
& $Py fetch-tvl --start 2021-01-01 --end $Today --out data\live_tvl.parquet *>> $Log
& $Py fetch-fees --start 2021-01-01 --end $Today --out data\live_fees.parquet *>> $Log

# Carry inputs: perpetual bars (for the basis) and funding history.
& $Py fetch-universe --symbols @majors --interval 1d --start 2021-01-01 --end $Today --futures --out data\live_perp.parquet *>> $Log
& $Py fetch-funding --symbols @majors --start 2021-01-01 --end $Today --out data\live_funding.parquet *>> $Log

"[$([DateTime]::Now)] marking alpha-engine account..." | Tee-Object -FilePath $Log -Append
& $Py paper --three-sleeve --data data\live.parquet --defi-data data\live_defi.parquet `
    --tvl-data data\live_tvl.parquet --state data\paper.json --start-equity 100000 *>> $Log

"[$([DateTime]::Now)] marking two-engine account..." | Tee-Object -FilePath $Log -Append
& $Py paper --two-engine --data data\live.parquet --defi-data data\live_defi.parquet `
    --tvl-data data\live_tvl.parquet --fees-data data\live_fees.parquet `
    --perp-data data\live_perp.parquet --funding-data data\live_funding.parquet `
    --state data\paper_two_engine.json --start-equity 100000 *>> $Log

"[$([DateTime]::Now)] done." | Tee-Object -FilePath $Log -Append
