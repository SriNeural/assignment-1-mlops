# Turn on Docker Desktop's built-in Kubernetes by editing its settings file.
# A copy of each file is saved beside it as *.bak-cardiascreen before any write.
$ErrorActionPreference = "Stop"
$paths = @(
    (Join-Path $env:APPDATA "Docker\settings-store.json"),
    (Join-Path $env:APPDATA "Docker\settings.json")
)
$seen = $false
foreach ($path in $paths) {
    if (-not (Test-Path -LiteralPath $path)) {
        continue
    }
    $seen = $true
    $raw = [IO.File]::ReadAllText($path)
    $updated = [regex]::Replace($raw, '"kubernetesEnabled"\s*:\s*false', '"kubernetesEnabled": true')
    $updated = [regex]::Replace($updated, '"KubernetesEnabled"\s*:\s*false', '"KubernetesEnabled": true')
    if ($updated -notmatch '(?i)"kubernetesEnabled"') {
        $trimmed = $updated.TrimEnd()
        if ($trimmed.EndsWith("}")) {
            $body = $trimmed.Substring(0, $trimmed.Length - 1).TrimEnd().TrimEnd(",")
            if ($body.Trim() -eq "{") {
                $updated = "{`r`n  `"kubernetesEnabled`": true`r`n}"
            } else {
                $updated = $body + ",`r`n  `"kubernetesEnabled`": true`r`n}"
            }
        }
    }
    if ($updated -eq $raw) {
        Write-Output "Kubernetes is already enabled in $path"
        continue
    }
    Copy-Item -LiteralPath $path -Destination ($path + ".bak-cardiascreen") -Force
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [IO.File]::WriteAllText($path, $updated, $utf8)
    Write-Output "Enabled Kubernetes in $path"
}
if (-not $seen) {
    Write-Output "Docker Desktop has no settings file yet. Start it once, then run setup again."
    exit 2
}
exit 0
