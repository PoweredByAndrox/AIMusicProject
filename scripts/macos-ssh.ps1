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

function Join-RemotePosixPath {
    param(
        [string]$Root,
        [string]$Child
    )
    return $Root.TrimEnd("/") + "/" + $Child.TrimStart("/")
}

$ForbiddenMacLocalArtifactRoots = @(
    "/Users/andrejviluha/aimusic_local_artifacts",
    "/Users/andrejviluha/aimusic_local_artifacts/host_artifacts",
    "/Users/andrejviluha/aimusic_local_artifacts/mac_api",
    '$HOME/aimusic_local_artifacts',
    '${HOME}/aimusic_local_artifacts',
    '~/aimusic_local_artifacts',
    "/Users/andrejviluha/win_artifacts",
    "/Users/andrejviluha/artifacts",
    "/tmp/artifacts"
)

function Get-ForbiddenMacLocalArtifactRoot {
    param([string]$Command)
    foreach ($root in $ForbiddenMacLocalArtifactRoots) {
        if ($Command.Contains($root)) {
            return $root
        }
    }
    return $null
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

$forbiddenProjectRoot = Get-ForbiddenMacLocalArtifactRoot -Command $MacProjectRoot
if (-not [string]::IsNullOrWhiteSpace($forbiddenProjectRoot)) {
    throw "Forbidden Mac-local project root: $forbiddenProjectRoot. Use the mounted Windows project root, normally /Users/andrejviluha/AIMusicProject_win."
}

$MacArtifactRoot = Join-RemotePosixPath -Root $MacProjectRoot -Child "artifacts"
$MacLabArtifactsRoot = Join-RemotePosixPath -Root $MacProjectRoot -Child "lab_artifacts"
$MacArtifactRagRoot = Join-RemotePosixPath -Root $MacProjectRoot -Child "artifact_rag"

foreach ($derivedRoot in @($MacArtifactRoot, $MacLabArtifactsRoot, $MacArtifactRagRoot)) {
    $forbiddenDerivedRoot = Get-ForbiddenMacLocalArtifactRoot -Command $derivedRoot
    if (-not [string]::IsNullOrWhiteSpace($forbiddenDerivedRoot)) {
        throw "Forbidden Mac-local derived artifact root: $forbiddenDerivedRoot. Write result artifacts directly under $MacProjectRoot/artifacts, $MacProjectRoot/lab_artifacts, or $MacProjectRoot/artifact_rag."
    }
}

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

$forbiddenRoot = Get-ForbiddenMacLocalArtifactRoot -Command $RemoteCommand
if (-not [string]::IsNullOrWhiteSpace($forbiddenRoot)) {
    throw "Forbidden Mac-local result artifact root in RemoteCommand: $forbiddenRoot. Write result artifacts directly under $MacProjectRoot/artifacts, $MacProjectRoot/lab_artifacts, or $MacProjectRoot/artifact_rag."
}

$remoteRoot = ConvertTo-RemoteSingleQuoted -Value $MacProjectRoot
$remoteArtifactRoot = ConvertTo-RemoteSingleQuoted -Value $MacArtifactRoot
$remoteLabArtifactsRoot = ConvertTo-RemoteSingleQuoted -Value $MacLabArtifactsRoot
$remoteArtifactRagRoot = ConvertTo-RemoteSingleQuoted -Value $MacArtifactRagRoot
& ssh @sshArgs "cd $remoteRoot && export AIMUSIC_MAC_PROJECT_ROOT=$remoteRoot && export AIMUSIC_MAC_ARTIFACT_ROOT=$remoteArtifactRoot && export AIMUSIC_MAC_LAB_ARTIFACTS_ROOT=$remoteLabArtifactsRoot && export AIMUSIC_MAC_ARTIFACT_RAG_ROOT=$remoteArtifactRagRoot && $RemoteCommand"
exit $LASTEXITCODE
