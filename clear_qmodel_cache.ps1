<#
.SYNOPSIS
    Clears Ultralytics disk-cache artifacts (.npy sidecars + *.cache label files)
    from QModel v6 channel datasets, with a dry-run report and confirmation.

.DESCRIPTION
    Ultralytics cache="disk" writes one .npy per source image plus train.cache /
    val.cache label files. These are regenerated automatically on the next run,
    so deleting them is safe AS LONG AS you are switching to CACHE_MODE="ram" or
    None (otherwise they just get rewritten). Source .png images and labels/*.txt
    files are NEVER touched.

.PARAMETER DatasetRoot
    Root folder containing the per-channel dataset dirs (default: data\datasets).

.PARAMETER Channel
    Optional. Limit cleanup to a single channel dir (e.g. ch_poi4_diff).
    Omit to clean ALL channels.

.PARAMETER Force
    Skip the confirmation prompt (for scripted/automated runs).

.EXAMPLE
    .\clear_qmodel_cache.ps1
    Dry-run report for all channels, then prompts before deleting.

.EXAMPLE
    .\clear_qmodel_cache.ps1 -Channel ch_poi4_diff
    Clean only the crashed channel.

.EXAMPLE
    .\clear_qmodel_cache.ps1 -Force
    Clean all channels with no prompt.
#>

[CmdletBinding()]
param(
    [string]$DatasetRoot = "data\datasets",
    [string]$Channel,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Write-Header($text) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 0. Resolve target path
# ---------------------------------------------------------------------------
if ($Channel) {
    $target = Join-Path $DatasetRoot $Channel
} else {
    $target = $DatasetRoot
}

if (-not (Test-Path $target)) {
    Write-Host "ERROR: path not found: $target" -ForegroundColor Red
    Write-Host "Pass -DatasetRoot pointing at your per-channel dataset folders." -ForegroundColor Red
    exit 1
}

$targetFull = (Resolve-Path $target).Path
Write-Header "QModel cache cleanup"
Write-Host "Target          : $targetFull"
Write-Host "Scope           : $(if ($Channel) { "single channel ($Channel)" } else { 'ALL channels' })"

# ---------------------------------------------------------------------------
# 1. Check for lingering python processes (zombie handles from a crash)
# ---------------------------------------------------------------------------
$pythons = Get-Process python -ErrorAction SilentlyContinue
if ($pythons) {
    Write-Host ""
    Write-Host "WARNING: python.exe processes are running. If any is a zombie from" -ForegroundColor Yellow
    Write-Host "the crashed training run it may hold file locks and block deletion:" -ForegroundColor Yellow
    $pythons | Select-Object Id, StartTime | Format-Table -AutoSize
    Write-Host "Kill stragglers with:  Stop-Process -Id <id>" -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 2. Free-space snapshot (before)
# ---------------------------------------------------------------------------
$driveLetter = $targetFull.Substring(0,1)
function Get-FreeGB($letter) {
    $d = Get-PSDrive -Name $letter
    [math]::Round($d.Free / 1GB, 2)
}
$freeBefore = Get-FreeGB $driveLetter

# ---------------------------------------------------------------------------
# 3. Dry-run inventory
# ---------------------------------------------------------------------------
Write-Header "Scanning for cache artifacts (this can take a moment)"

$npy   = Get-ChildItem $targetFull -Recurse -Filter *.npy   -File -ErrorAction SilentlyContinue
$cache = Get-ChildItem $targetFull -Recurse -Filter *.cache -File -ErrorAction SilentlyContinue

$npySum   = ($npy   | Measure-Object -Property Length -Sum).Sum
$cacheSum = ($cache | Measure-Object -Property Length -Sum).Sum
if (-not $npySum)   { $npySum   = 0 }
if (-not $cacheSum) { $cacheSum = 0 }

$npyGB   = [math]::Round($npySum   / 1GB, 2)
$cacheGB = [math]::Round($cacheSum / 1GB, 2)
$totalGB = [math]::Round(($npySum + $cacheSum) / 1GB, 2)

# Per-channel breakdown of .npy (the big stuff)
Write-Host ""
Write-Host "Per-channel .npy sidecar sizes:" -ForegroundColor White
if ($npy.Count -gt 0) {
    $npy |
        Group-Object { 
            # channel = first path segment under the dataset root
            $rel = $_.FullName.Substring($targetFull.Length).TrimStart('\')
            ($rel -split '\\')[0]
        } |
        ForEach-Object {
            [PSCustomObject]@{
                Channel = $_.Name
                Files   = $_.Count
                GB      = [math]::Round((($_.Group | Measure-Object Length -Sum).Sum)/1GB, 2)
            }
        } |
        Sort-Object GB -Descending |
        Format-Table -AutoSize
} else {
    Write-Host "  (none found - Ultralytics did not write .npy sidecars)" -ForegroundColor DarkGray
}

Write-Host "Label .cache files:" -ForegroundColor White
if ($cache.Count -gt 0) {
    $cache | Select-Object @{N='File';E={$_.FullName.Substring($targetFull.Length).TrimStart('\')}},
                           @{N='KB';E={[math]::Round($_.Length/1KB,1)}} |
             Format-Table -AutoSize
} else {
    Write-Host "  (none found)" -ForegroundColor DarkGray
}

Write-Header "Summary"
Write-Host ("  .npy sidecars : {0,8} files   {1,8} GB" -f $npy.Count,   $npyGB)
Write-Host ("  .cache files  : {0,8} files   {1,8} GB" -f $cache.Count, $cacheGB)
Write-Host ("  TOTAL reclaim : {0,30} GB" -f $totalGB) -ForegroundColor Green
Write-Host ("  Drive {0}: free now : {1,22} GB" -f $driveLetter, $freeBefore)
Write-Host ("  Drive {0}: free after: ~{1,21} GB" -f $driveLetter, ([math]::Round($freeBefore + $totalGB, 2))) -ForegroundColor Green

if (($npy.Count + $cache.Count) -eq 0) {
    Write-Host ""
    Write-Host "Nothing to delete. Exiting." -ForegroundColor Green
    exit 0
}

# ---------------------------------------------------------------------------
# 4. Confirm
# ---------------------------------------------------------------------------
if (-not $Force) {
    Write-Host ""
    Write-Host "This deletes ONLY *.npy and *.cache files. Source .png images and" -ForegroundColor Yellow
    Write-Host "labels/*.txt are NOT touched." -ForegroundColor Yellow
    $resp = Read-Host "Proceed with deletion? (type 'yes' to confirm)"
    if ($resp -ne "yes") {
        Write-Host "Aborted. No files deleted." -ForegroundColor Red
        exit 0
    }
}

# ---------------------------------------------------------------------------
# 5. Delete
# ---------------------------------------------------------------------------
Write-Header "Deleting"
$deleted = 0
$failed  = @()

foreach ($f in @($npy) + @($cache)) {
    try {
        Remove-Item -LiteralPath $f.FullName -Force
        $deleted++
    } catch {
        $failed += $f.FullName
    }
}

Write-Host ("Deleted {0} files." -f $deleted) -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host ("{0} file(s) could NOT be deleted (likely locked by a process):" -f $failed.Count) -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host "Kill any lingering python.exe and re-run." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 6. Free-space snapshot (after)
# ---------------------------------------------------------------------------
$freeAfter = Get-FreeGB $driveLetter
Write-Header "Done"
Write-Host ("  Drive {0}: free before : {1,8} GB" -f $driveLetter, $freeBefore)
Write-Host ("  Drive {0}: free after  : {1,8} GB" -f $driveLetter, $freeAfter)
Write-Host ("  Reclaimed            : {0,8} GB" -f ([math]::Round($freeAfter - $freeBefore, 2))) -ForegroundColor Green
Write-Host ""
Write-Host "Next: set CACHE_MODE='ram' and CACHE_RAM_LIMIT_GB=75 in config.py" -ForegroundColor Cyan
Write-Host "before relaunching, so these sidecars are not rewritten." -ForegroundColor Cyan