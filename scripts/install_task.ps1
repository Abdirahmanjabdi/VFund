# Registers a Windows Scheduled Task that runs the weekly paper update.
# Run once:  powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1
# Remove:    Unregister-ScheduledTask -TaskName "VFundPaperUpdate" -Confirm:$false

$Root = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $Root "scripts\update_paper.ps1"
$TaskName = "VFundPaperUpdate"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Script`""
# Weekly, Monday 09:00. Change -DaysOfWeek / -At to taste.
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 9:00AM

# Every non-default below fixes an observed failure of the forward test. A
# missed week cannot be back-filled - the paper tracker refuses to mark a stale
# gap forward (paper.py MAX_GAP_DAYS) - so an ops failure permanently costs a
# data point from the only honest record this project has.
#
#   AllowStartIfOnBatteries / DontStopIfGoingOnBatteries
#       Windows defaults BOTH to "no". On a laptop that means the run is
#       skipped, or killed mid-flight with 0xC000013A. This silently ate the
#       2026-07-20 run.
#   WakeToRun
#       Default false: a sleeping machine simply misses the trigger.
#   ExecutionTimeLimit 2h
#       The update now fetches five feeds (spot, defi, tvl, fees, perp, funding)
#       and marks two accounts. 30 minutes left no margin for a slow API day.
#   RestartCount/Interval
#       A transient Binance timeout retries instead of losing the week.
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -WakeToRun `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "VFund: fetch data + update paper account weekly" `
    -Force | Out-Null
Write-Host "Registered scheduled task '$TaskName' (weekly, Mondays 9:00 AM)."
Write-Host "Run now to test:  Start-ScheduledTask -TaskName $TaskName"
Write-Host "Check it ran:     (Get-ScheduledTaskInfo $TaskName).LastTaskResult  # 0 = success"
