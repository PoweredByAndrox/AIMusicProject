[CmdletBinding()]
param(
    [string]$MacHost = "192.168.0.138",
    [string]$MacUser = "andrejviluha",
    [string]$SshKey = "",
    [string]$MacProjectRoot = "/Users/andrejviluha/AIMusicProject_win",
    [string]$RemoteCommand = "",
    [switch]$NoMountCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-DefaultSshKey {
    if (-not [string]::IsNullOrWhiteSpace($SshKey)) {
        return $SshKey
    }
    return (Join-Path $env:USERPROFILE ".ssh\id_ed25519_phone")
}

function ConvertTo-RemoteSingleQuoted {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'""'""'") + "'"
}

if (-not (Get-Command "ssh" -ErrorAction SilentlyContinue)) {
    throw "Required command is missing from PATH: ssh"
}

$sshKeyPath = Resolve-DefaultSshKey
if (-not (Test-Path -LiteralPath $sshKeyPath)) {
    throw "SSH key not found: $sshKeyPath"
}

$target = "$MacUser@$MacHost"
$sshArgs = @(
    "-i", $sshKeyPath,
    "-o", "IdentitiesOnly=yes",
    "-o", "BatchMode=yes",
    $target
)

if (-not $NoMountCheck) {
    $mount = ConvertTo-RemoteSingleQuoted -Value $MacProjectRoot
    $probeCommand = "MP=$mount; mount | grep -F -- `"`$MP`" >/dev/null 2>&1 && test -d `"`$MP/ml_worker_improved`" && mkdir -p `"`$MP/.aimusic_mount_probe`" && test -w `"`$MP/.aimusic_mount_probe`""
    & ssh @sshArgs $probeCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Mac project mount is not writable at $MacProjectRoot. Run scripts\macos-bootstrap-workspace-mount.ps1 first."
    }
}

if ([string]::IsNullOrWhiteSpace($RemoteCommand)) {
    Write-Host ("ssh -i `"{0}`" -o IdentitiesOnly=yes -o BatchMode=yes {1}" -f $sshKeyPath, $target)
    & ssh @sshArgs
    exit $LASTEXITCODE
}

$remoteRoot = ConvertTo-RemoteSingleQuoted -Value $MacProjectRoot
& ssh @sshArgs "cd $remoteRoot && $RemoteCommand"
exit $LASTEXITCODE
