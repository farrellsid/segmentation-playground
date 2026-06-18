<#
.SYNOPSIS
  Resilient copy of the cross-worm VAST GT (project 280) off the flaky external
  E: drive into the repo, so eval runs never have to touch E: again.

.DESCRIPTION
  Copies the 851 *.vsseg_export_s*.png labelmaps (~76 MB) and the small
  VAST_segmentation_metadata.txt into:
      data/groundtruth/masks_p280_local/   (masks)
      data/groundtruth/                     (metadata, sibling of skeletons_p280)

  The E: drive mounts then drops within seconds, so this:
    * waits (polling) for the source dir to become visible,
    * runs robocopy in restartable mode (/Z) with per-file retries (/R /W),
    * loops the whole thing until all 851 PNGs are local (robocopy skips files
      already copied), or -MaxWaitMin elapses.

  Once this reports "851 PNGs local", config.GT_MASK_DIR_LOCAL /
  GT_METADATA_LOCAL are auto-preferred by GroundTruth.from_config() — no code
  change needed. Then run:  py -3 -m eval.registration

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\copy_gt_masks_local.ps1
#>
[CmdletBinding()]
param(
  [string]$SrcRoot   = "E:\ZhenLab\SEM DAUER 1",
  [int]$ExpectedPng  = 851,
  [int]$MaxWaitMin   = 30,     # give up if E: never shows for this long
  [int]$PollSec      = 5
)

$ErrorActionPreference = "Continue"

# Destination is resolved relative to this script (repo-portable). These live on
# an always-present drive, so Join-Path is safe.
$repo     = Split-Path -Parent $MyInvocation.MyCommand.Path
$gtDir    = Join-Path $repo "data\groundtruth"
$maskDst  = Join-Path $gtDir "masks_p280_local"
$metaDst  = Join-Path $gtDir "VAST_segmentation_metadata.txt"

# Source paths are on the flaky E: drive. Build them as PLAIN STRINGS, NOT with
# Join-Path: Join-Path validates the drive and THROWS DriveNotFoundException when
# E: is unmounted, which would leave these null and break the whole wait loop.
$srcMasks = $SrcRoot.TrimEnd('\') + "\Vast files"
$srcMeta  = $SrcRoot.TrimEnd('\') + "\VAST_segmentation_metadata.txt"

New-Item -ItemType Directory -Force -Path $maskDst | Out-Null

function Count-LocalPng {
  (Get-ChildItem $maskDst -Filter "*.vsseg_export_s*.png" -File -ErrorAction SilentlyContinue | Measure-Object).Count
}

$deadline = (Get-Date).AddMinutes($MaxWaitMin)
$round = 0
while ($true) {
  $have = Count-LocalPng
  if ($have -ge $ExpectedPng) {
    Write-Host "[copy] DONE: $have/$ExpectedPng PNGs local." -ForegroundColor Green
    break
  }
  if ((Get-Date) -gt $deadline) {
    Write-Host "[copy] TIMEOUT after $MaxWaitMin min: $have/$ExpectedPng PNGs local. Re-run to resume." -ForegroundColor Yellow
    break
  }

  if (-not (Test-Path $srcMasks)) {
    Write-Host ("[copy] waiting for E: ... ({0}/{1} local so far)" -f $have, $ExpectedPng)
    Start-Sleep -Seconds $PollSec
    continue
  }

  $round++
  Write-Host "[copy] round $round : E: visible, copying ($have/$ExpectedPng local)..." -ForegroundColor Cyan

  # metadata (tiny) first, so a single brief mount window still secures it.
  if ((Test-Path $srcMeta) -and -not (Test-Path $metaDst)) {
    Copy-Item $srcMeta $metaDst -Force -ErrorAction SilentlyContinue
    if (Test-Path $metaDst) { Write-Host "[copy] metadata copied." -ForegroundColor Green }
  }

  # /Z restartable, /R:2 /W:2 quick retries, /NP no per-file %, /NDL no dir list,
  # /XX don't flag extra local files. robocopy skips files already present+identical.
  robocopy "$srcMasks" "$maskDst" "*.vsseg_export_s*.png" /Z /R:2 /W:2 /NP /NDL /NJH /NJS /XX | Out-Null

  Start-Sleep -Seconds 1
}

# -- final report --
$have = Count-LocalPng
$mb = "{0:N1}" -f (((Get-ChildItem $maskDst -Filter "*.png" -File -ErrorAction SilentlyContinue) | Measure-Object Length -Sum).Sum / 1MB)
Write-Host "[copy] masks: $have/$ExpectedPng PNGs, $mb MB at $maskDst"
Write-Host "[copy] metadata local: $(Test-Path $metaDst)"
if ($have -ge $ExpectedPng -and (Test-Path $metaDst)) {
  Write-Host "[copy] GT is now fully local. Next: py -3 -m eval.registration" -ForegroundColor Green
}
