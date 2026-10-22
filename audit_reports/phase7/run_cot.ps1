# Phase 7 CoT holdout run (FINAL_CONFIG 11.4): gpt-5.6-luna, smoothed cut, lag 0 then lag 2.
#
# New run (from anywhere):
#   powershell -ExecutionPolicy Bypass -File "<repo>\audit_reports\phase7\run_cot.ps1"
#   1. Preflight: 2 real calls on pre-cap data (well under 1 cent). Checks connection, model name and parsing,
#      and prints a cost estimate. If it fails, nothing else is called.
#   2. Lag 0, then lag 2 (1,110 cells each). One progress line per cell. Every finished cell is saved to disk
#      at once, so a crash, a closed window or a lost connection never loses work already paid for.
# Continue after any stop (finished cells are never called again; a finished lag is skipped):
#   powershell -ExecutionPolicy Bypass -File "<repo>\audit_reports\phase7\run_cot.ps1" -Stamp <stamp shown on screen>
# Optional: -Model <exact name the API reports>, if the preflight stopped on a model-name mismatch.
param(
    [string]$Model = "gpt-5.6-luna",
    [string]$StatedCutoff = "2026-02-16",
    [string]$Effort = "low",
    [string]$Stamp = "",
    [string]$Py = "C:\Anaconda3\python.exe",
    [string]$OutRoot = "audit_reports\phase7",
    [switch]$NoPopup
)

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
$Panel = "..\facts_mas_phase7_archive\archive_20261020b\panel_smoothed.csv"
$Resuming = [bool]$Stamp
if (-not $Resuming) { $Stamp = Get-Date -Format "yyyyMMdd_HHmm" }
$LogDir = Join-Path $OutRoot "cot_logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$Self = "powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`""

function Done($title, $text) {
    Write-Host "`n==== $title ====" -ForegroundColor Cyan
    Write-Host $text
    try { [console]::beep(880, 400); [console]::beep(660, 400) } catch {}
    if (-not $NoPopup) {
        try {
            Add-Type -AssemblyName System.Windows.Forms
            [System.Windows.Forms.MessageBox]::Show($text, "FACTS-MAS CoT: $title") | Out-Null
        } catch {}
    }
}

function Run($argline, $log) {
    cmd /c "`"$Py`" audit_reports/phase7/cot_holdout.py $argline 2>&1" | Tee-Object -FilePath $log | Out-Host
    return $LASTEXITCODE
}

function MismatchHint($log) {
    $txt = (Get-Content $log -Raw)
    if ($txt -match "response.model '([^']+)'") {
        return "`n`nThe API reports the model as '$($Matches[1])'. Start again with:`n$Self -Model $($Matches[1])"
    }
    return ""
}

if (-not (Test-Path $Panel)) { Done "STOPPED" "Panel not found: $Panel"; exit 1 }
Write-Host "Run stamp: $Stamp   (to continue after any stop: $Self -Stamp $Stamp)" -ForegroundColor Yellow

if (-not $Resuming) {
    $log = Join-Path $LogDir "cot_${Stamp}_preflight.txt"
    Write-Host "`n==== preflight: $Model, effort $Effort (2 calls, pre-cap data) ====" -ForegroundColor Cyan
    $code = Run "--preflight --model $Model --reasoning-effort $Effort" $log
    if ($code -ne 0) {
        $tail = (Get-Content $log -Tail 3) -join "`n"
        Done "STOPPED at preflight (exit $code)" ("Nothing else was called (cost well under 1 cent).`nLast lines:`n$tail" +
            (MismatchHint $log) + "`n`nTell Claude what this says.")
        exit $code
    }
}

$summary = @()
$tokIn = 0; $tokOut = 0
foreach ($lag in 0, 2) {
    $out = Join-Path $OutRoot "cot_${Stamp}_lag$lag"
    $final = Join-Path $out "cot_run_log_lag$lag.json"
    $log = Join-Path $LogDir ("cot_${Stamp}_lag${lag}_" + (Get-Date -Format "HHmmss") + ".txt")
    if (Test-Path $final) {
        Write-Host "`n==== lag ${lag}: already finished, skipped ====" -ForegroundColor Cyan
    } else {
        $mode = ""
        if (Test-Path $out) { $mode = "--resume" }
        Write-Host "`n==== lag ${lag}: $Model, effort $Effort, output $out $mode ====" -ForegroundColor Cyan
        Write-Host "Started $(Get-Date -Format 'HH:mm:ss'). Console log: $log`n"
        $code = Run ("--panel `"$Panel`" --model $Model --stated-cutoff $StatedCutoff --reasoning-effort $Effort " +
                     "--pub-lag $lag --out `"$out`" $mode") $log
        if ($code -ne 0) {
            $tail = (Get-Content $log -Tail 3) -join "`n"
            Done "STOPPED (lag $lag, exit $code)" ("Every finished cell is saved. Last lines:`n$tail" + (MismatchHint $log) +
                "`n`nTo continue where it stopped, run:`n$Self -Stamp $Stamp")
            exit $code
        }
    }
    $j = Get-Content $final -Raw | ConvertFrom-Json
    $tokIn += [double]$j.prompt_tokens; $tokOut += [double]$j.completion_tokens
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
$cost = [math]::Round(($tokIn * 0.20 + $tokOut * 1.20) / 1e6, 2)
Done "FINISHED" (($summary -join "`n") + "`n`nTokens: $tokIn in, $tokOut out (about `$$cost at list price)." +
    "`nOutput folders: $OutRoot\cot_${Stamp}_lag0 and _lag2. Tell Claude 'CoT done'.")
