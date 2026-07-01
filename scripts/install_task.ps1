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
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "VFund: fetch data + update paper account weekly" `
    -Force | Out-Null
Write-Host "Registered scheduled task '$TaskName' (weekly, Mondays 9:00 AM)."
Write-Host "Run now to test:  Start-ScheduledTask -TaskName $TaskName"
