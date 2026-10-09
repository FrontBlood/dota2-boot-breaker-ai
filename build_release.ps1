param(
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildRoot = Join-Path $Root "build"
$DistRoot = Join-Path $Root "dist"
$ReleaseRoot = Join-Path $Root "release"
$PackageName = "Dota2-Boot-Breaker-AI-v$Version-win64"
$PackageDir = Join-Path $ReleaseRoot $PackageName
$Archive = Join-Path $ReleaseRoot "$PackageName.zip"
$ConfigData = "$(Join-Path $Root 'boot_config.json');."
$GuideName = -join @([char]0x4F7F, [char]0x7528, [char]0x6307, [char]0x5357, ".md")

foreach ($Path in @($BuildRoot, $DistRoot, $PackageDir)) {
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}
if (Test-Path -LiteralPath $Archive) {
    Remove-Item -LiteralPath $Archive -Force
}

python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --console `
    --name "Dota2-Boot-Breaker-AI" `
    --add-data $ConfigData `
    --distpath $DistRoot `
    --workpath (Join-Path $BuildRoot "pyinstaller") `
    --specpath $BuildRoot `
    (Join-Path $Root "boot_breaker\__main__.py")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

New-Item -ItemType Directory -Path $PackageDir -Force | Out-Null
Copy-Item -Path (Join-Path $DistRoot "Dota2-Boot-Breaker-AI\*") -Destination $PackageDir -Recurse -Force
Copy-Item -LiteralPath (Join-Path $Root "RELEASE_README.md") -Destination (Join-Path $PackageDir $GuideName)
Copy-Item -Path (Join-Path $Root "release_launchers\*.bat") -Destination $PackageDir

# cmd.exe is unreliable with UTF-8/LF batch files. Launchers contain ASCII
# only and are normalized to native Windows CRLF before publishing.
$Ascii = [System.Text.Encoding]::ASCII
Get-ChildItem -LiteralPath $PackageDir -Filter "*.bat" | ForEach-Object {
    $Text = [System.IO.File]::ReadAllText($_.FullName)
    $Text = [System.Text.RegularExpressions.Regex]::Replace($Text, "\r?\n", "`r`n")
    [System.IO.File]::WriteAllText($_.FullName, $Text, $Ascii)
}

Compress-Archive -Path $PackageDir -DestinationPath $Archive -CompressionLevel Optimal
Write-Host "Release directory: $PackageDir"
Write-Host "Release archive:   $Archive"
