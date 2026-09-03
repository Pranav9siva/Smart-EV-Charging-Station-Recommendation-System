$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$osm = Join-Path $root 'data/osm/bangalore_subarea.osm'
$out = Join-Path $root 'simulations/bangalore/network.net.xml'
$dir = Split-Path -Parent $out
New-Item -ItemType Directory -Force -Path $dir | Out-Null

if (-not (Test-Path $osm)) {
    Write-Error "OSM file not found at $osm. Run scripts/fetch_osm_network.py first."
    exit 1
}

netconvert --osm-files $osm --output-file $out --geometry.remove --roundabouts.guess --ramps.guess --junctions.join --tls.guess-signals --tls.discard-simple --tls.join
Write-Host "Wrote SUMO network to $out"
