[CmdletBinding()]
param(
    [string]$MacHost = "192.168.0.138",
    [string]$MacUser = "andrejviluha",
    [string]$SshKey = "",
    [string]$WindowsProjectRoot = "",
    [string]$WindowsHostIp = "",
    [string]$WindowsShareName = "AIMusicProject",
    [string]$WindowsArtifactShareName = "AIMusicArtifacts",
    [string]$SmbUser = "MacAudioApi",
    [string]$MacMountPath = "/Users/andrejviluha/AIMusicProject_win",
    [System.Management.Automation.PSCredential]$SmbCredential = $null,
    [switch]$UseExistingCredential,
    [switch]$ForceCredentialReset
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ("[{0}] {1}" -f (Get-Date).ToString("s"), $Message)
}

function Resolve-DefaultProjectRoot {
    if (-not [string]::IsNullOrWhiteSpace($WindowsProjectRoot)) {
        return (Resolve-Path -LiteralPath $WindowsProjectRoot).Path
    }
    return (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
}

function Resolve-DefaultSshKey {
    if (-not [string]::IsNullOrWhiteSpace($SshKey)) {
        return $SshKey
    }
    return (Join-Path $env:USERPROFILE ".ssh\id_ed25519_phone")
}

function Get-DefaultWindowsHostIp {
    if (-not [string]::IsNullOrWhiteSpace($WindowsHostIp)) {
        return $WindowsHostIp
    }

    $candidate = Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object {
            $_.IPAddress -like "192.168.*" -and
            $_.IPAddress -ne "127.0.0.1" -and
            $_.PrefixOrigin -ne "WellKnown" -and
            $_.AddressState -eq "Preferred"
        } |
        Select-Object -First 1 -ExpandProperty IPAddress

    if ([string]::IsNullOrWhiteSpace($candidate)) {
        throw "Could not auto-detect a 192.168.x.x Windows IPv4 address. Pass -WindowsHostIp explicitly."
    }
    return $candidate
}

function New-StrongAsciiPassword {
    $chars = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789".ToCharArray()
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $rng.GetBytes($bytes)
    } finally {
        $rng.Dispose()
    }
    $passwordChars = for ($i = 0; $i -lt $bytes.Length; $i++) {
        $chars[$bytes[$i] % $chars.Length]
    }
    return -join $passwordChars
}

function Assert-CommandAvailable {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command is missing from PATH: $Name"
    }
}

function Ensure-LocalSmbUser {
    param(
        [string]$Name,
        [securestring]$Password
    )

    $existing = Get-LocalUser -Name $Name -ErrorAction SilentlyContinue
    if ($null -eq $existing) {
        New-LocalUser `
            -Name $Name `
            -Password $Password `
            -PasswordNeverExpires `
            -UserMayNotChangePassword `
            -Description "AIMusicProject Mac SMB artifact writer" | Out-Null
    } else {
        Enable-LocalUser -Name $Name
        Set-LocalUser -Name $Name -Password $Password
        Set-LocalUser -Name $Name -PasswordNeverExpires $true -UserMayChangePassword $false
    }
}

function Ensure-SmbShareWithChangeAccess {
    param(
        [string]$Name,
        [string]$Path,
        [string]$Account
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }

    $share = Get-SmbShare -Name $Name -ErrorAction SilentlyContinue
    if ($null -eq $share) {
        New-SmbShare -Name $Name -Path $Path -ChangeAccess $Account | Out-Null
    } else {
        Grant-SmbShareAccess -Name $Name -AccountName $Account -AccessRight Change -Force | Out-Null
    }
}

function Ensure-MacSmbFirewallRule {
    param(
        [string]$RemoteAddress
    )

    $ruleName = "AIMusicProject-Mac-SMB-In"
    $displayName = "AIMusicProject Mac SMB"
    $existing = Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue
    if ($null -eq $existing) {
        New-NetFirewallRule `
            -Name $ruleName `
            -DisplayName $displayName `
            -Direction Inbound `
            -Action Allow `
            -Protocol TCP `
            -LocalPort 445 `
            -RemoteAddress $RemoteAddress `
            -Profile Any | Out-Null
    } else {
        Set-NetFirewallRule -Name $ruleName -Enabled True -Direction Inbound -Action Allow -Profile Any
        Get-NetFirewallRule -Name $ruleName | Get-NetFirewallPortFilter | Set-NetFirewallPortFilter -Protocol TCP -LocalPort 445
        Get-NetFirewallRule -Name $ruleName | Get-NetFirewallAddressFilter | Set-NetFirewallAddressFilter -RemoteAddress $RemoteAddress
    }
}

function Grant-PathAccess {
    param(
        [string]$Path,
        [string]$Account,
        [string]$Rights
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
    $aclText = (& icacls $Path 2>&1) -join "`n"
    $hasAccountRule = $aclText -match [regex]::Escape($Account)
    $hasModifyEquivalent = $aclText -match [regex]::Escape($Account) -and $aclText -match "\((?:[^\)]*,)?[MF](?:,[^\)]*)?\)"
    if ($hasAccountRule -and ($Rights -eq "RX" -or $hasModifyEquivalent)) {
        return
    }

    $rule = "{0}:(OI)(CI)({1})" -f $Account, $Rights
    & icacls $Path /grant $rule | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "icacls failed for $Path"
    }
}

function ConvertTo-RemoteSingleQuoted {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'""'""'") + "'"
}

function Invoke-MacShellScript {
    param(
        [string]$ScriptText,
        [string]$Target,
        [string]$KeyPath
    )

    $leaf = "aimp_mac_" + ([guid]::NewGuid().ToString("N")) + ".sh"
    $localPath = Join-Path ([System.IO.Path]::GetTempPath()) $leaf
    $remotePath = "/tmp/$leaf"
    Set-Content -LiteralPath $localPath -Value $ScriptText -Encoding ascii -NoNewline
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & scp -i $KeyPath -o IdentitiesOnly=yes -o BatchMode=yes $localPath "${Target}:$remotePath" 2>&1 |
            ForEach-Object { Write-Host $_ }
        $scpExitCode = $LASTEXITCODE
        if ($scpExitCode -ne 0) {
            throw "scp failed while copying bootstrap script to Mac"
        }

        $remoteQuoted = ConvertTo-RemoteSingleQuoted -Value $remotePath
        & ssh -i $KeyPath -o IdentitiesOnly=yes -o BatchMode=yes $Target "chmod 700 $remoteQuoted && $remoteQuoted; RC=`$?; rm -f $remoteQuoted; exit `$RC" 2>&1 |
            ForEach-Object { Write-Host $_ }
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
        Remove-Item -LiteralPath $localPath -Force -ErrorAction SilentlyContinue
    }
}

Assert-CommandAvailable -Name "ssh"
Assert-CommandAvailable -Name "scp"

$projectRoot = Resolve-DefaultProjectRoot
$sshKeyPath = Resolve-DefaultSshKey
$windowsIp = Get-DefaultWindowsHostIp
$target = "$MacUser@$MacHost"
$account = "$env:COMPUTERNAME\$SmbUser"

if (-not (Test-Path -LiteralPath $sshKeyPath)) {
    throw "SSH key not found: $sshKeyPath"
}

Write-Step "Verifying Mac SSH with explicit key: $target"
& ssh -i $sshKeyPath -o IdentitiesOnly=yes -o BatchMode=yes $target "printf 'ssh_ok\n'"
if ($LASTEXITCODE -ne 0) {
    throw "Mac SSH failed. Do not use hostname fallback; fix explicit-key access first."
}

Write-Step "Ensuring Windows SMB shares and ACLs"
Ensure-SmbShareWithChangeAccess -Name $WindowsShareName -Path $projectRoot -Account $account
Ensure-SmbShareWithChangeAccess -Name $WindowsArtifactShareName -Path (Join-Path $projectRoot "artifacts") -Account $account
Ensure-MacSmbFirewallRule -RemoteAddress $MacHost
Grant-PathAccess -Path $projectRoot -Account $account -Rights "RX"
foreach ($relativePath in @("artifacts", "lab_artifacts", "artifact_rag", ".aimusic_mount_probe")) {
    Grant-PathAccess -Path (Join-Path $projectRoot $relativePath) -Account $account -Rights "M"
}

$probeScript = @"
#!/bin/sh
set -eu
MP='$MacMountPath'
if ! mount | grep -F -- "`$MP" >/dev/null 2>&1; then
  printf 'mount_missing=%s\n' "`$MP"
  exit 78
fi
python3 - "`$MP" <<'PY'
import pathlib
import signal
import sys
import time

def _timeout(_signum, _frame):
    raise TimeoutError("mount probe timed out")

mount = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGALRM, _timeout)
signal.alarm(12)
try:
    if not (mount / "ml_worker_improved").is_dir():
        print(f"mount_missing={mount}")
        sys.exit(78)
    probe_dir = mount / ".aimusic_mount_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_dir / f"bootstrap_probe_{time.time_ns()}.txt"
    payload = f"aimusic_mount_probe:{time.time_ns()}"
    probe.write_text(payload, encoding="utf-8")
    readback = probe.read_text(encoding="utf-8")
    probe.unlink()
    if readback != payload:
        print(f"probe_failed={mount}")
        sys.exit(79)
    print(f"mount_ok={mount}")
    print("probe=write_read_delete_passed")
finally:
    signal.alarm(0)
PY
"@

if (-not $ForceCredentialReset) {
    Write-Step "Checking existing Mac mount before resetting credentials"
    $probeRc = Invoke-MacShellScript -ScriptText $probeScript -Target $target -KeyPath $sshKeyPath
    if ($probeRc -eq 0) {
        Write-Step "Existing mount is writable; no SMB credential reset needed"
        Write-Host "Future connect command:"
        Write-Host "powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1"
        exit 0
    }
}

$plainPassword = $null
if ($UseExistingCredential) {
    if ($null -eq $SmbCredential) {
        $SmbCredential = Get-Credential -UserName $account -Message "Enter the Windows SMB credential for the Mac mount."
    }
    $plainPassword = $SmbCredential.GetNetworkCredential().Password
} else {
    $plainPassword = New-StrongAsciiPassword
    Write-Step "Creating/resetting local SMB account password without printing it"
    Ensure-LocalSmbUser -Name $SmbUser -Password (ConvertTo-SecureString $plainPassword -AsPlainText -Force)
}

$mountScript = @"
#!/bin/sh
set -eu
MP='$MacMountPath'
WINDOWS_IP='$windowsIp'
SHARE='$WindowsShareName'
SMB_USER='$SmbUser'
SMB_PASS='$plainPassword'
if [ '$ForceCredentialReset' = 'True' ]; then
  if mount | grep -F -- "`$MP" >/dev/null 2>&1; then
    umount "`$MP" >/dev/null 2>&1 || true
  fi
fi
if ! mount | grep -F -- "`$MP" >/dev/null 2>&1; then
  mkdir -p "`$MP"
  mount_smbfs "//`$SMB_USER:`$SMB_PASS@`$WINDOWS_IP/`$SHARE" "`$MP" 2>/tmp/aimp_mount_smbfs.err
  SMB_RC=`$?
  if [ "`$SMB_RC" -ne 0 ]; then
    printf 'mount_smbfs_failed=%s\n' "`$SMB_RC"
    sed "s/`$SMB_PASS/***/g" /tmp/aimp_mount_smbfs.err 2>/dev/null || true
    exit 77
  fi
fi
if ! mount | grep -F -- "`$MP" >/dev/null 2>&1; then
  printf 'stable_mount_path_missing=%s\n' "`$MP"
  printf 'mounted_volumes:\n'
  mount | grep AIMusicProject 2>/dev/null || true
  exit 78
fi
python3 - "`$MP" <<'PY'
import pathlib
import signal
import sys
import time

def _timeout(_signum, _frame):
    raise TimeoutError("mount probe timed out")

mount = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGALRM, _timeout)
signal.alarm(12)
try:
    if not (mount / "ml_worker_improved").is_dir():
        print(f"mount_missing={mount}")
        sys.exit(78)
    probe_dir = mount / ".aimusic_mount_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_dir / f"bootstrap_probe_{time.time_ns()}.txt"
    payload = f"aimusic_mount_probe:{time.time_ns()}"
    probe.write_text(payload, encoding="utf-8")
    readback = probe.read_text(encoding="utf-8")
    probe.unlink()
    if readback != payload:
        print(f"probe_failed={mount}")
        sys.exit(79)
    print(f"mount_ok={mount}")
    print("probe=write_read_delete_passed")
finally:
    signal.alarm(0)
PY
printf 'windows_share=\\\\%s\\%s\n' "`$WINDOWS_IP" "`$SHARE"
/usr/local/bin/container --version 2>/dev/null || true
"@

Write-Step "Mounting Windows project share on Mac and running write/read/delete probe"
$mountRc = Invoke-MacShellScript -ScriptText $mountScript -Target $target -KeyPath $sshKeyPath
if ($mountRc -ne 0) {
    throw "Mac workspace mount bootstrap failed with exit code $mountRc"
}

Write-Step "Bootstrap completed"
Write-Host "Mac mount path: $MacMountPath"
Write-Host "Windows project root: $projectRoot"
Write-Host "Future connect command:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1"
