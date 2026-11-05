$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath (Join-Path $PSScriptRoot 'frontend')
try {
    npm.cmd run dev
} finally {
    Pop-Location
}
