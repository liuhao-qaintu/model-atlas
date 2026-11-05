param(
    [string]$PythonPath = '',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$taskBackendPython = $PythonPath
if (-not $taskBackendPython) {
    if ($env:MODEL_ATLAS_PYTHON) {
        $taskBackendPython = $env:MODEL_ATLAS_PYTHON
    } elseif (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')) {
        $taskBackendPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    } else {
        throw 'Specify the Python interpreter with -PythonPath or MODEL_ATLAS_PYTHON.'
    }
}
Get-Command -Name $taskBackendPython -CommandType Application -ErrorAction Stop | Out-Null
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($CheckOnly) {
        & $taskBackendPython -c "import importlib.util, sys; required = ['uvicorn', 'fastapi', 'sqlalchemy', 'pymysql', 'numpy', 'pypdf', 'jwt', 'bcrypt', 'backend.main']; missing = [name for name in required if importlib.util.find_spec(name) is None]; print('Python:', sys.executable); print('Missing modules:', ', '.join(missing) if missing else 'none'); sys.exit(1 if missing else 0)"
        if ($LASTEXITCODE -ne 0) { throw 'Backend dependency check failed.' }
        return
    }
    & $taskBackendPython -m uvicorn backend.main:app --host 127.0.0.1 --port 8002 --reload --reload-dir backend --timeout-graceful-shutdown 15
    if ($LASTEXITCODE -ne 0) { throw 'Backend stopped with an error; inspect the preceding traceback.' }
} finally {
    Pop-Location
}
