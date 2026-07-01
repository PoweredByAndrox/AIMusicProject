# scripts/macos-bootstrap-and-ssh.ps1

[CmdletBinding()]
param(
    [string]$MacHost = "192.168.0.138",
    [string]$MacUser = "andrejviluha",
    [string]$SshKey = (Join-Path $env:USERPROFILE ".ssh\id_ed25519_phone"),
    [string]$WindowsProjectRoot = "D:\AIMusicProject",
    [string]$WindowsHostIp = "",
    [string]$WindowsShareName = "AIMusicProject",
    [string]$WindowsArtifactShareName = "AIMusicArtifacts",
    [string]$SmbUser = "MacAudioApi",
    [string]$MacMountPath = "/Users/andrejviluha/AIMusicProject_win",
    [switch]$UseExistingCredential,
    [switch]$ForceCredentialReset,
    [switch]$SkipAppleContainerCheck,
    [switch]$Connect,
    [string]$RemoteCommand = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-CommandAvailable {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command is missing from PATH: $Name"
    }
}

function Add-OptionalArg {
    param(
        [System.Collections.Generic.List[string]]$ArgsList,
        [string]$Name,
        [string]$Value
    )
    if (-not [string]::IsNullOrWhiteSpace($Value)) {
        $ArgsList.Add($Name)
        $ArgsList.Add($Value)
    }
}

function Invoke-PowerShellScript {
    param([string[]]$Arguments)
    & powershell @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: powershell $($Arguments -join ' ')"
    }
}

Assert-CommandAvailable -Name "powershell"

$scriptRoot = Split-Path -Parent $PSCommandPath
$bootstrapScript = Join-Path $scriptRoot "macos-bootstrap-workspace-mount.ps1"
$sshScript = Join-Path $scriptRoot "macos-ssh.ps1"

if (-not (Test-Path -LiteralPath $bootstrapScript)) {
    throw "Missing bootstrap script: $bootstrapScript"
}
if (-not (Test-Path -LiteralPath $sshScript)) {
    throw "Missing SSH script: $sshScript"
}

$bootstrapArgs = [System.Collections.Generic.List[string]]::new()
$bootstrapArgs.Add("-ExecutionPolicy")
$bootstrapArgs.Add("Bypass")
$bootstrapArgs.Add("-File")
$bootstrapArgs.Add($bootstrapScript)
Add-OptionalArg -ArgsList $bootstrapArgs -Name "-MacHost" -Value $MacHost
Add-OptionalArg -ArgsList $bootstrapArgs -Name "-MacUser" -Value $MacUser
Add-OptionalArg -ArgsList $bootstrapArgs -Name "-SshKey" -Value $SshKey
Add-OptionalArg -ArgsList $bootstrapArgs -Name "-WindowsProjectRoot" -Value $WindowsProjectRoot
Add-OptionalArg -ArgsList $bootstrapArgs -Name "-WindowsHostIp" -Value $WindowsHostIp
Add-OptionalArg -ArgsList $bootstrapArgs -Name "-WindowsShareName" -Value $WindowsShareName
Add-OptionalArg -ArgsList $bootstrapArgs -Name "-WindowsArtifactShareName" -Value $WindowsArtifactShareName
Add-OptionalArg -ArgsList $bootstrapArgs -Name "-SmbUser" -Value $SmbUser
Add-OptionalArg -ArgsList $bootstrapArgs -Name "-MacMountPath" -Value $MacMountPath
if ($UseExistingCredential) {
    $bootstrapArgs.Add("-UseExistingCredential")
}
if ($ForceCredentialReset) {
    $bootstrapArgs.Add("-ForceCredentialReset")
}

Write-Host "Bootstrapping Mac SSH + Windows workspace mount:"
Write-Host "powershell $($bootstrapArgs -join ' ')"
Invoke-PowerShellScript -Arguments $bootstrapArgs.ToArray()

$sshBaseArgs = [System.Collections.Generic.List[string]]::new()
$sshBaseArgs.Add("-ExecutionPolicy")
$sshBaseArgs.Add("Bypass")
$sshBaseArgs.Add("-File")
$sshBaseArgs.Add($sshScript)
Add-OptionalArg -ArgsList $sshBaseArgs -Name "-MacHost" -Value $MacHost
Add-OptionalArg -ArgsList $sshBaseArgs -Name "-MacUser" -Value $MacUser
Add-OptionalArg -ArgsList $sshBaseArgs -Name "-SshKey" -Value $SshKey
Add-OptionalArg -ArgsList $sshBaseArgs -Name "-MacProjectRoot" -Value $MacMountPath

if (-not $SkipAppleContainerCheck) {
    $readyCommand = "/usr/local/bin/container --version && pwd && test -d ml_worker_improved && test -w artifacts && printf 'mac_ready\n'"
    $readyArgs = [System.Collections.Generic.List[string]]::new()
    $readyArgs.AddRange($sshBaseArgs)
    $readyArgs.Add("-RemoteCommand")
    $readyArgs.Add($readyCommand)
    Write-Host "Verifying Apple container and mounted Windows workspace:"
    Write-Host "powershell $($readyArgs -join ' ')"
    Invoke-PowerShellScript -Arguments $readyArgs.ToArray()
}

Write-Host "Mac bootstrap is ready."
Write-Host "From-zero command:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\macos-bootstrap-and-ssh.ps1"
Write-Host "Future connect command:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1"
Write-Host "Future remote command example:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1 -RemoteCommand '/usr/local/bin/container --version && pwd'"

if (-not [string]::IsNullOrWhiteSpace($RemoteCommand)) {
    $remoteArgs = [System.Collections.Generic.List[string]]::new()
    $remoteArgs.AddRange($sshBaseArgs)
    $remoteArgs.Add("-RemoteCommand")
    $remoteArgs.Add($RemoteCommand)
    Invoke-PowerShellScript -Arguments $remoteArgs.ToArray()
} elseif ($Connect) {
    Invoke-PowerShellScript -Arguments $sshBaseArgs.ToArray()
}
