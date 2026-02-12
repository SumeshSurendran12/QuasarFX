param(
  [string]$RemoteHost = "192.222.55.197",
  [string]$KeyPath = "$env:USERPROFILE\.ssh\lambda_a100",
  [string]$RemoteDir = "/home/ubuntu/forex-trading-bot/models",
  [string]$AltRemoteDir = "/home/ubuntu/forex-trading-bot/modules/models",
  [string]$LocalDir = "C:\Users\sumes\OneDrive - UTHealth Houston\Desktop\FX\Forex-trading-bot\models\downloads",
  [int]$WatchSeconds = 0
)

New-Item -ItemType Directory -Path $LocalDir -Force | Out-Null

$files = @(
  "best_model.zip",
  "best_model_metadata.json",
  "final_model.zip"
)

do {
  foreach ($f in $files) {
    $localPath = Join-Path $LocalDir $f
    scp -i $KeyPath "ubuntu@${RemoteHost}:$RemoteDir/$f" "$localPath"
    if ($LASTEXITCODE -ne 0 -and $AltRemoteDir) {
      scp -i $KeyPath "ubuntu@${RemoteHost}:$AltRemoteDir/$f" "$localPath"
    }
  }
  if ($WatchSeconds -gt 0) {
    Start-Sleep -Seconds $WatchSeconds
  }
} while ($WatchSeconds -gt 0)
