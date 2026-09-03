$report = "D:\Smart_EV_Station_Recommandadtion_System\disk_report.txt"
"Report generated: $(Get-Date)" | Out-File $report
Get-Volume -DriveLetter D | Select-Object DriveLetter,Size,SizeRemaining | Format-List | Out-File $report -Append
$folders = '.venv','notebooks','sumo','data','src','.github','Documentation','tests','notebooks'
foreach ($f in $folders) {
    $p = Join-Path 'D:\Smart_EV_Station_Recommandadtion_System' $f
    if (Test-Path $p) {
        $s=(Get-ChildItem $p -Recurse -Force -ErrorAction SilentlyContinue | Where-Object {!$_.PSIsContainer} | Measure-Object Length -Sum).Sum
        "$f`t$s" | Out-File $report -Append
    } else {
        "$f`tMISSING" | Out-File $report -Append
    }
}
"--- Top files ---" | Out-File $report -Append
Get-ChildItem 'D:\Smart_EV_Station_Recommandadtion_System' -Recurse -Force -ErrorAction SilentlyContinue | Where-Object { -not $_.PSIsContainer } | Sort-Object Length -Descending | Select-Object FullName,Length -First 40 | ForEach-Object { "$($_.FullName)`t$($_.Length)" | Out-File $report -Append }
