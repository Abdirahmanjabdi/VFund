# VFund - weekly paper-account update.
#
# Marks TWO forward accounts:
#   data/paper.json             the alpha engine alone (trend + size + on-chain).
#                               The original, unbroken forward record - NOT
#                               superseded, it is one half of the book below, and
#                               its history is worth preserving intact.
#   data/paper_two_engine.json  the full two-engine book: 4-sleeve alpha (adds
#                               the on-chain fees sleeve) + funding carry, 50/50.
#                               This is the leading candidate.
#
# Running both is deliberate: the alpha engine is common to each, so the gap
# between the two curves isolates what the carry engine actually contributes
# forward, instead of leaving it to a backtest to claim.
#
# Registered as a weekly Scheduled Task by scripts/install_task.ps1.
# Run manually any time:  powershell -File scripts\update_paper.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\vfund.exe"
$Log = Join-Path $Root "data\paper_log.txt"
$Today = Get-Date -Format "yyyy-MM-dd"
Set-Location $Root

# Append UTF-8, not PowerShell 5.1's default UTF-16LE. The old log was UTF-16,
# which made it unreadable to grep/tail and to every other tool you would reach
# for while diagnosing a failed run - exactly when you need it most.
function Write-Log([string]$Text) {
    $Text | Out-File -FilePath $Log -Append -Encoding utf8
}
function Invoke-Step([string[]]$VfundArgs) {
    & $Py @VfundArgs 2>&1 | Out-File -FilePath $Log -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Write-Log "[$([DateTime]::Now)] STEP FAILED (exit $LASTEXITCODE): vfund $($VfundArgs -join ' ')"
        throw "vfund $($VfundArgs[0]) failed with exit code $LASTEXITCODE"
    }
}

$defi = "AAVEUSDT","UNIUSDT","CRVUSDT","COMPUSDT","SUSHIUSDT","LDOUSDT","GMXUSDT",
        "PENDLEUSDT","CAKEUSDT","DYDXUSDT","1INCHUSDT","BALUSDT","YFIUSDT","RUNEUSDT",
        "JOEUSDT","LQTYUSDT","SPELLUSDT","CVXUSDT","RPLUSDT","STGUSDT","ENAUSDT",
        "ETHFIUSDT","JTOUSDT","JUPUSDT","RAYUSDT"

# Carry trades liquid majors only - see vfund/live/carry.py MAJORS.
$majors = "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT",
          "AVAXUSDT","LINKUSDT","DOTUSDT","LTCUSDT","TRXUSDT","BCHUSDT"

Write-Log "[$([DateTime]::Now)] fetching data..."
Invoke-Step @("fetch-universe","--top","60","--interval","1d","--start","2021-01-01","--end",$Today,"--out","data\live.parquet")
Invoke-Step (@("fetch-universe","--symbols") + $defi + @("--interval","1d","--start","2021-01-01","--end",$Today,"--out","data\live_defi.parquet"))
Invoke-Step @("fetch-tvl","--start","2021-01-01","--end",$Today,"--out","data\live_tvl.parquet")
Invoke-Step @("fetch-fees","--start","2021-01-01","--end",$Today,"--out","data\live_fees.parquet")

# Carry inputs: perpetual bars (for the basis) and funding history.
Invoke-Step (@("fetch-universe","--symbols") + $majors + @("--interval","1d","--start","2021-01-01","--end",$Today,"--futures","--out","data\live_perp.parquet"))
Invoke-Step (@("fetch-funding","--symbols") + $majors + @("--start","2021-01-01","--end",$Today,"--out","data\live_funding.parquet"))

Write-Log "[$([DateTime]::Now)] marking alpha-engine account..."
Invoke-Step @("paper","--three-sleeve","--data","data\live.parquet","--defi-data","data\live_defi.parquet",
              "--tvl-data","data\live_tvl.parquet","--state","data\paper.json","--start-equity","100000")

Write-Log "[$([DateTime]::Now)] marking two-engine account..."
Invoke-Step @("paper","--two-engine","--data","data\live.parquet","--defi-data","data\live_defi.parquet",
              "--tvl-data","data\live_tvl.parquet","--fees-data","data\live_fees.parquet",
              "--perp-data","data\live_perp.parquet","--funding-data","data\live_funding.parquet",
              "--state","data\paper_two_engine.json","--start-equity","100000")

# Health gate: `vfund status` exits non-zero if any account is stale, so a
# silent failure becomes a failed task instead of a gap noticed weeks later.
# A missed update cannot be back-filled - the tracker refuses stale gaps by
# design - so the data point would be lost permanently.
Write-Log "[$([DateTime]::Now)] checking account health..."
& $Py status 2>&1 | Out-File -FilePath $Log -Append -Encoding utf8
$health = $LASTEXITCODE
Write-Log "[$([DateTime]::Now)] done (health exit $health)."
exit $health
