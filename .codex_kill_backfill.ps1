Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -and $_.CommandLine.Contains('tse_backfill_2025_notion.py') } |
  Sort-Object ProcessId -Descending |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Write-Output 'killed'
