# Windows equivalent of run.sh (Docker Desktop with the NVIDIA runtime; MSF's g5.4xlarge target).
#   .\run.ps1 C:\data\aoi.tif C:\data\out --context arid_urban
param([Parameter(Mandatory)][string]$Image, [Parameter(Mandatory)][string]$Out,
      [Parameter(ValueFromRemainingArguments)][string[]]$Rest)
$img = (Resolve-Path $Image).Path
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$out = (Resolve-Path $Out).Path
$name = $env:IMAGE; if (-not $name) { $name = "msf-footprints:0.1" }
docker run --rm --gpus all --shm-size=8g -v "$(Split-Path $img):/in:ro" -v "${out}:/out" $name `
  --image "/in/$(Split-Path $img -Leaf)" --out /out @Rest
