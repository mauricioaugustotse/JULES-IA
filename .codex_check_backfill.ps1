Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -and $_.CommandLine.Contains('tse_backfill_2025_notion.py') } |
  Select-Object ProcessId, ParentProcessId, CommandLine |
  Sort-Object ProcessId |
  Format-Table -Wrap -AutoSize
