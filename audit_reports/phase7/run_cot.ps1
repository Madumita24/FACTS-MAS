# Phase 7 CoT holdout run (FINAL_CONFIG 11.4): gpt-5.6-luna, smoothed cut, lag 0 then lag 2.
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File "<repo>\audit_reports\phase7\run_cot.ps1"
# Optional: -Model <exact name the API reports>, if the first attempt stopped on a model-name mismatch.
# Shows one progress line per cell, stops at the first error, prints a summary, and pops up a message at the end.
# Each attempt writes to new folders (timestamped), so a stopped attempt is kept and never overwritten.
param(
    [string]$Model = "gpt-5.6-luna",
    [string]$StatedCutoff = "2026-02-16",
    [string]$Effort = "low"
)

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
$Py = "C:\Anaconda3\python.exe"
$Panel = "..\facts_mas_phase7_archive\archive_20261020b\panel_smoothed.csv"
$Stamp = Get-Date -Format "yyyyMMdd_HHmm"
$LogDir = "audit_reports\phase7\cot_logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

function Done($title, $text) {
    Write-Host "`n==== $title ====" -ForegroundColor Cyan
    Write-Host $text
    try { [console]::beep(880, 400); [console]::beep(660, 400) } catch {}
    try {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show($text, "FACTS-MAS CoT: $title") | Out-Null
    } catch {}
}

if (-not (Test-Path $Panel)) { Done "STOPPED" "Panel not found: $Panel"; exit 1 }

$summary = @()
foreach ($lag in 0, 2) {
    $out = "audit_reports\phase7\cot_${Stamp}_lag$lag"
    $log = "$LogDir\cot_${Stamp}_lag$lag.txt"
    Write-Host "`n==== lag $lag : $Model, effort $Effort, output $out ====" -ForegroundColor Cyan
    Write-Host "Started $(Get-Date -Format 'HH:mm:ss'). Full console log: $log`n"
    $cmd = "`"$Py`" audit_reports/phase7/cot_holdout.py --panel `"$Panel`" --model $Model " +
           "--stated-cutoff $StatedCutoff --reasoning-effort $Effort --pub-lag $lag --out `"$out`" 2>&1"
    cmd /c $cmd | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        $tail = (Get-Content $log -Tail 3) -join "`n"
        $hint = ""
        if ($tail -match "response.model '([^']+)'") {
            $hint = "`n`nThe API reports the model as '$($Matches[1])'. Re-run with:`n" +
                    "powershell -ExecutionPolicy Bypass -File audit_reports\phase7\run_cot.ps1 -Model $($Matches[1])"
        }
        Done "STOPPED (lag $lag, exit $code)" "Last lines:`n$tail$hint`n`nNothing was deleted. Tell Claude what this says."
        exit $code
    }
    $j = Get-Content "$out\cot_run_log_lag$lag.json" -Raw | ConvertFrom-Json
    $line = "lag ${lag}: failures $($j.n_failures)/$($j.n_cells) ($([math]::Round(100 * $j.failure_share, 2))%), " +
            "out_of_sample=$($j.out_of_sample), models seen=$($j.response_models_seen -join ','), " +
            "self-reported cutoff: $($j.self_reported_cutoff_verbatim)"
    Write-Host $line -ForegroundColor Green
    if ($j.rerun_allowed) {
        $line += "`n  Failure share > 5%: one re-run pass is allowed. Run:`n  `"$Py`" audit_reports/phase7/cot_holdout.py " +
                 "--panel `"$Panel`" --model $Model --stated-cutoff $StatedCutoff --reasoning-effort $Effort " +
                 "--pub-lag $lag --out `"$out`" --rerun-transport-failures"
    }
    $summary += $line
}
Done "FINISHED" (($summary -join "`n") + "`n`nOutput folders: audit_reports\phase7\cot_${Stamp}_lag0 and _lag2. Tell Claude 'CoT done'.")
