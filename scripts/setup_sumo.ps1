param(
    [switch]$Persist
)

$sumoPath = 'C:\Program Files (x86)\Eclipse\Sumo'

# Set for current session
$env:SUMO_HOME = $sumoPath
$env:PATH = "$sumoPath\bin;" + $env:PATH
Write-Host "SUMO_HOME set to $env:SUMO_HOME"
Write-Host "SUMO bin prepended to PATH for this session"

if ($Persist) {
    Write-Host "Persisting SUMO_HOME to user environment (requires new shell)"
    setx SUMO_HOME "$sumoPath"
}
