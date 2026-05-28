[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$VirtualDjExe = "",
    [string]$VirtualDjDataRoot = "",
    [string]$ExportDir = "",
    [string]$ManifestOutput = "/app/lab_artifacts/virtualdj/baseline_manifest.json",
    [string]$FullMix = "",
    [string]$OutputDir = "/app/lab_artifacts/virtualdj/nine_thou_quality_gate",
    [switch]$LaunchVirtualDj,
    [switch]$Visible,
    [string]$OpenTrack = "",
    [int]$WaitForExportsSeconds = 0,
    [switch]$RunQualityGate,
    [switch]$ProbeNetworkControl,
    [string]$NetworkControlHost = "127.0.0.1",
    [int[]]$NetworkControlPorts = @(80, 8080),
    [string]$NetworkControlBearer = "",
    [int]$WaitForNetworkControlSeconds = 0,
    [switch]$LoadOpenTrackViaNetworkControl,
    [switch]$CloseVirtualDjWhenDone,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GateArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$AudioExtensions = @(".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aif", ".aiff")

function Resolve-ProjectRoot {
    param([string]$Root)

    if (-not [string]::IsNullOrWhiteSpace($Root)) {
        return (Resolve-Path -LiteralPath $Root).Path
    }

    $scriptRoot = $PSScriptRoot
    if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
        $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    return (Resolve-Path -LiteralPath (Join-Path $scriptRoot "..")).Path
}

function Get-AudioFiles {
    param([string]$Root)

    if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
        return @()
    }

    return @(
        Get-ChildItem -LiteralPath $Root -Recurse -File |
            Where-Object { $AudioExtensions -contains $_.Extension.ToLowerInvariant() }
    )
}

function Convert-AppLabPathToHostPath {
    param(
        [string]$ContainerPath,
        [string]$Root
    )

    $normalizedContainerPath = $ContainerPath.Replace("\", "/")
    if (-not $normalizedContainerPath.StartsWith("/app/lab_artifacts/")) {
        throw "ManifestOutput must be under /app/lab_artifacts so the ml-worker volume writes it back to the host."
    }

    $relativeParts = @(
        $normalizedContainerPath.Substring("/app/".Length).Split("/") |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and $_ -ne "." }
    )
    if ($relativeParts.Count -eq 0 -or $relativeParts -contains "..") {
        throw "ManifestOutput must not contain empty, current-directory, or parent-directory path segments: $ContainerPath"
    }

    $resolvedRoot = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Root).Path).TrimEnd("\", "/")
    $labArtifactsRoot = [System.IO.Path]::GetFullPath((Join-Path $resolvedRoot "lab_artifacts")).TrimEnd("\", "/")
    $hostPath = Join-Path $resolvedRoot ($relativeParts -join [System.IO.Path]::DirectorySeparatorChar)
    $resolvedHostPath = [System.IO.Path]::GetFullPath($hostPath)
    $labArtifactsRootWithSeparator = $labArtifactsRoot + [System.IO.Path]::DirectorySeparatorChar
    $isInsideLabArtifacts = $resolvedHostPath.StartsWith(
        $labArtifactsRootWithSeparator,
        [System.StringComparison]::OrdinalIgnoreCase
    )
    if (-not $isInsideLabArtifacts) {
        throw "ManifestOutput must resolve inside ProjectRoot lab_artifacts: $resolvedHostPath"
    }

    return $resolvedHostPath
}

function Convert-HostProjectPathToContainerPath {
    param(
        [string]$HostPath,
        [string]$Root
    )

    $resolvedPath = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $HostPath).Path)
    $resolvedRoot = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Root).Path).TrimEnd("\", "/")
    $rootWithSeparator = $resolvedRoot + [System.IO.Path]::DirectorySeparatorChar
    $isRootPath = [string]::Equals($resolvedPath, $resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)
    $isChildPath = $resolvedPath.StartsWith($rootWithSeparator, [System.StringComparison]::OrdinalIgnoreCase)
    if (-not ($isRootPath -or $isChildPath)) {
        throw "Path must be inside ProjectRoot for /host_project mapping: $resolvedPath"
    }

    $relative = $resolvedPath.Substring($resolvedRoot.Length).TrimStart("\", "/").Replace("\", "/")
    return "/host_project/$relative"
}

function Get-VirtualDjProcessInfo {
    $processes = @(Get-Process -Name "virtualdj" -ErrorAction SilentlyContinue)
    $items = @()

    foreach ($process in $processes) {
        $connections = @()
        $startTime = $null
        try {
            $connections = @(
                Get-NetTCPConnection -OwningProcess $process.Id -ErrorAction Stop |
                    Select-Object LocalAddress, LocalPort, RemoteAddress, RemotePort, State
            )
        }
        catch {
            $connections = @()
        }
        try {
            $startTime = $process.StartTime.ToString("o")
        }
        catch {
            $startTime = $null
        }

        $items += [pscustomobject]@{
            id = $process.Id
            process_name = $process.ProcessName
            path = $process.Path
            start_time = $startTime
            tcp_connections = $connections
        }
    }

    return $items
}

function Get-VirtualDjPluginEvidence {
    param([string]$ResolvedVirtualDjDataRoot)

    $pluginsRoot = Join-Path $ResolvedVirtualDjDataRoot "Plugins64"
    $networkControlFiles = @()
    if (Test-Path -LiteralPath $pluginsRoot -PathType Container) {
        $networkControlFiles = @(
            Get-ChildItem -LiteralPath $pluginsRoot -Recurse -File -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.Name -match "Network" -or
                    $_.Name -match "Control" -or
                    $_.DirectoryName -match "Network" -or
                    $_.DirectoryName -match "Control"
                } |
                Select-Object FullName, Length, LastWriteTime
        )
    }

    return [pscustomobject]@{
        plugins64_root = $pluginsRoot
        plugins64_root_exists = Test-Path -LiteralPath $pluginsRoot -PathType Container
        network_control_candidate_files = $networkControlFiles
    }
}

function Invoke-VirtualDjNetworkControlCommand {
    param(
        [string]$HostName,
        [int]$Port,
        [ValidateSet("query", "execute")]
        [string]$Mode,
        [string]$Script,
        [string]$Bearer = ""
    )

    $encodedScript = [System.Uri]::EscapeDataString($Script)
    $uri = "http://${HostName}:${Port}/${Mode}?script=${encodedScript}"
    $headers = @{}
    if (-not [string]::IsNullOrWhiteSpace($Bearer)) {
        $headers["Authorization"] = "Bearer $Bearer"
    }

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $uri -Headers $headers -TimeoutSec 3 -ErrorAction Stop
        $stopwatch.Stop()
        return [pscustomobject]@{
            status = "ok"
            mode = $Mode
            script = $Script
            endpoint = "http://${HostName}:${Port}/${Mode}"
            status_code = [int]$response.StatusCode
            response_text = [string]$response.Content
            error_message = $null
            elapsed_ms = [int]$stopwatch.ElapsedMilliseconds
        }
    }
    catch {
        $stopwatch.Stop()
        $statusCode = $null
        $body = $null
        if ($_.Exception.Response) {
            try {
                $statusCode = [int]$_.Exception.Response.StatusCode
            }
            catch {
                $statusCode = $null
            }
            try {
                $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
                $body = $reader.ReadToEnd()
                $reader.Dispose()
            }
            catch {
                $body = $null
            }
        }

        $status = if ($statusCode) { "http_error" } else { "connection_failed" }
        return [pscustomobject]@{
            status = $status
            mode = $Mode
            script = $Script
            endpoint = "http://${HostName}:${Port}/${Mode}"
            status_code = $statusCode
            response_text = $body
            error_message = $_.Exception.Message
            elapsed_ms = [int]$stopwatch.ElapsedMilliseconds
        }
    }
}

function ConvertTo-VdjScriptQuotedPath {
    param([string]$Path)

    $resolvedPath = (Resolve-Path -LiteralPath $Path).Path
    return '"' + $resolvedPath.Replace('"', '\"') + '"'
}

function Test-VirtualDjNetworkControlHandshake {
    param(
        [object]$FolderProbe,
        [object]$VersionProbe
    )

    if ($FolderProbe.status -ne "ok" -or $VersionProbe.status -ne "ok") {
        return $false
    }

    $folderText = ([string]$FolderProbe.response_text).Trim()
    $versionText = ([string]$VersionProbe.response_text).Trim()
    if ([string]::IsNullOrWhiteSpace($folderText) -or [string]::IsNullOrWhiteSpace($versionText)) {
        return $false
    }

    $folderLooksLikeVirtualDj = $folderText -match "VirtualDJ" -or $folderText -match "virtualdj"
    $versionLooksLikeVirtualDj = $versionText -match "\d+(\.\d+)*" -or $versionText -match "VirtualDJ" -or $versionText -match "virtualdj"
    return $folderLooksLikeVirtualDj -and $versionLooksLikeVirtualDj
}

function Invoke-VirtualDjNetworkControlProbe {
    param(
        [string]$HostName,
        [int[]]$Ports,
        [string]$Bearer = "",
        [int]$WaitSeconds = 0,
        [string]$TrackToLoad = "",
        [bool]$LoadTrack = $false
    )

    if ($Ports.Count -eq 0) {
        throw "NetworkControlPorts must include at least one port."
    }
    if ($WaitSeconds -lt 0) {
        throw "WaitForNetworkControlSeconds must not be negative."
    }

    $attempts = @()
    $activePort = $null
    $deadline = (Get-Date).AddSeconds($WaitSeconds)

    do {
        foreach ($port in $Ports) {
            $folderProbe = Invoke-VirtualDjNetworkControlCommand `
                -HostName $HostName `
                -Port $port `
                -Mode "query" `
                -Script "get_vdj_folder" `
                -Bearer $Bearer
            $attempts += $folderProbe

            $versionProbe = Invoke-VirtualDjNetworkControlCommand `
                -HostName $HostName `
                -Port $port `
                -Mode "query" `
                -Script "get_version" `
                -Bearer $Bearer
            $attempts += $versionProbe

            if (Test-VirtualDjNetworkControlHandshake -FolderProbe $folderProbe -VersionProbe $versionProbe) {
                $activePort = $port
                break
            }
        }

        if ($activePort -ne $null -or $WaitSeconds -eq 0) {
            break
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)

    $actions = @()
    if ($activePort -ne $null) {
        $infoQueries = @("get_version", "get_build", "get_license", "get_vdj_folder")
        foreach ($script in $infoQueries) {
            $actions += Invoke-VirtualDjNetworkControlCommand `
                -HostName $HostName `
                -Port $activePort `
                -Mode "query" `
                -Script $script `
                -Bearer $Bearer
        }

        if ($LoadTrack) {
            if ([string]::IsNullOrWhiteSpace($TrackToLoad)) {
                throw "LoadOpenTrackViaNetworkControl requires OpenTrack."
            }
            $quotedPath = ConvertTo-VdjScriptQuotedPath -Path $TrackToLoad
            $actions += Invoke-VirtualDjNetworkControlCommand `
                -HostName $HostName `
                -Port $activePort `
                -Mode "execute" `
                -Script "deck 1 load $quotedPath" `
                -Bearer $Bearer
            Start-Sleep -Seconds 2

            $loadedTrackQueries = @(
                "deck 1 get_filepath",
                "deck 1 get_title",
                "deck 1 get_artist",
                "deck 1 get_bpm",
                "deck 1 get_key",
                "deck 1 has_stems",
                "deck 1 has_stems `"ready`""
            )
            foreach ($script in $loadedTrackQueries) {
                $actions += Invoke-VirtualDjNetworkControlCommand `
                    -HostName $HostName `
                    -Port $activePort `
                    -Mode "query" `
                    -Script $script `
                    -Bearer $Bearer
            }
        }
    }

    return [pscustomobject]@{
        requested = $true
        host = $HostName
        ports = @($Ports)
        wait_seconds = $WaitSeconds
        bearer_supplied = -not [string]::IsNullOrWhiteSpace($Bearer)
        available = $activePort -ne $null
        active_endpoint = if ($activePort -ne $null) { "http://${HostName}:${activePort}" } else { $null }
        attempts = $attempts
        actions = $actions
        status = if ($activePort -ne $null) { "available" } else { "unavailable" }
        unavailable_reason = if ($activePort -ne $null) { $null } else { "network_control_http_endpoint_not_found_or_not_enabled" }
        official_contract = [pscustomobject]@{
            plugin = "VirtualDJ Network Control"
            endpoints = @("/query", "/execute")
            script_language = "VDJScript"
            requires = "VirtualDJ 2023 or later and a Pro license; plugin must be installed and enabled in Auto-Start"
            source = "https://www.virtualdj.com/wiki/NetworkControlPlugin.html"
        }
    }
}

function Write-HostProbe {
    param(
        [string]$ProbePath,
        [string]$ResolvedVirtualDjExe,
        [string]$ResolvedVirtualDjDataRoot,
        [string]$ResolvedExportDir,
        [string]$ResolvedProjectRoot,
        [bool]$LaunchRequested,
        [object]$NetworkControlProbe = $null
    )

    $exeItem = if (Test-Path -LiteralPath $ResolvedVirtualDjExe -PathType Leaf) {
        Get-Item -LiteralPath $ResolvedVirtualDjExe
    } else {
        $null
    }

    $versionInfo = if ($exeItem) { $exeItem.VersionInfo } else { $null }
    $payload = [pscustomobject]@{
        schema_version = 1
        generated_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        tool = "scripts/run-virtualdj-baseline-gate.ps1"
        project_root = $ResolvedProjectRoot
        virtualdj = [pscustomobject]@{
            exe_path = $ResolvedVirtualDjExe
            exe_exists = [bool]$exeItem
            product_version = if ($versionInfo) { $versionInfo.ProductVersion } else { $null }
            file_version = if ($versionInfo) { $versionInfo.FileVersion } else { $null }
            file_size_bytes = if ($exeItem) { $exeItem.Length } else { $null }
            data_root = $ResolvedVirtualDjDataRoot
            data_root_exists = Test-Path -LiteralPath $ResolvedVirtualDjDataRoot -PathType Container
            launch_requested = $LaunchRequested
            plugin_evidence = Get-VirtualDjPluginEvidence -ResolvedVirtualDjDataRoot $ResolvedVirtualDjDataRoot
            processes = @(Get-VirtualDjProcessInfo)
            network_control = if ($NetworkControlProbe) {
                $NetworkControlProbe
            } else {
                [pscustomobject]@{
                    requested = $false
                    status = "not_requested"
                }
            }
        }
        export_dir = [pscustomobject]@{
            path = $ResolvedExportDir
            exists = Test-Path -LiteralPath $ResolvedExportDir -PathType Container
            audio_file_count = @(Get-AudioFiles -Root $ResolvedExportDir).Count
        }
    }

    $parent = Split-Path -Parent $ProbePath
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $ProbePath -Encoding UTF8
}

$ProjectRoot = Resolve-ProjectRoot -Root $ProjectRoot

if ([string]::IsNullOrWhiteSpace($VirtualDjExe)) {
    $VirtualDjExe = Join-Path ${env:ProgramFiles} "VirtualDJ\virtualdj.exe"
}
if ([string]::IsNullOrWhiteSpace($VirtualDjDataRoot)) {
    $VirtualDjDataRoot = Join-Path $env:LOCALAPPDATA "VirtualDJ"
}
if ([string]::IsNullOrWhiteSpace($ExportDir)) {
    $ExportDir = Join-Path $ProjectRoot "lab_artifacts\virtualdj\exports\nine_thou"
}
if ([string]::IsNullOrWhiteSpace($FullMix)) {
    $FullMix = Join-Path $ProjectRoot "Nine Thou (Grant Mohrman Superstars Remix) (320).mp3"
}

$VirtualDjExe = if (Test-Path -LiteralPath $VirtualDjExe) { (Resolve-Path -LiteralPath $VirtualDjExe).Path } else { $VirtualDjExe }
$VirtualDjDataRoot = (Resolve-Path -LiteralPath $VirtualDjDataRoot).Path
New-Item -ItemType Directory -Force -Path $ExportDir | Out-Null
$ExportDir = (Resolve-Path -LiteralPath $ExportDir).Path

$preLaunchVirtualDjProcessIds = @(Get-Process -Name "virtualdj" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
$virtualDjLaunchStartedAt = $null
$startedVirtualDjProcess = $null
if ($LaunchVirtualDj) {
    if (-not (Test-Path -LiteralPath $VirtualDjExe -PathType Leaf)) {
        throw "VirtualDJ executable not found: $VirtualDjExe"
    }

    $argumentList = @()
    if (-not [string]::IsNullOrWhiteSpace($OpenTrack)) {
        $argumentList += (Resolve-Path -LiteralPath $OpenTrack).Path
    }

    $windowStyle = if ($Visible) { "Normal" } else { "Hidden" }
    $virtualDjLaunchStartedAt = Get-Date
    $startedVirtualDjProcess = Start-Process -FilePath $VirtualDjExe -ArgumentList $argumentList -WindowStyle $windowStyle -PassThru
    Start-Sleep -Seconds 3
}

if ($WaitForExportsSeconds -gt 0) {
    $deadline = (Get-Date).AddSeconds($WaitForExportsSeconds)
    while ((Get-Date) -lt $deadline -and @(Get-AudioFiles -Root $ExportDir).Count -eq 0) {
        Start-Sleep -Seconds 2
    }
}

$networkControlProbe = $null
if ($ProbeNetworkControl) {
    $networkControlProbe = Invoke-VirtualDjNetworkControlProbe `
        -HostName $NetworkControlHost `
        -Ports $NetworkControlPorts `
        -Bearer $NetworkControlBearer `
        -WaitSeconds $WaitForNetworkControlSeconds `
        -TrackToLoad $OpenTrack `
        -LoadTrack ([bool]$LoadOpenTrackViaNetworkControl)
}

$hostProbePath = Join-Path $ProjectRoot "lab_artifacts\virtualdj\host_probe.json"
Write-HostProbe `
    -ProbePath $hostProbePath `
    -ResolvedVirtualDjExe $VirtualDjExe `
    -ResolvedVirtualDjDataRoot $VirtualDjDataRoot `
    -ResolvedExportDir $ExportDir `
    -ResolvedProjectRoot $ProjectRoot `
    -LaunchRequested ([bool]$LaunchVirtualDj) `
    -NetworkControlProbe $networkControlProbe

$manifestHostPath = Convert-AppLabPathToHostPath -ContainerPath $ManifestOutput -Root $ProjectRoot
$hostProbeContainerPath = Convert-HostProjectPathToContainerPath -HostPath $hostProbePath -Root $ProjectRoot

$manifestDockerArgs = @(
    "compose",
    "run",
    "--rm",
    "--no-deps",
    "-v",
    "${ProjectRoot}:/host_project:ro",
    "-v",
    "${VirtualDjDataRoot}:/virtualdj_data:ro",
    "-v",
    "${ExportDir}:/virtualdj_export:ro",
    "--entrypoint",
    "python",
    "ml-worker",
    "/app/scripts/virtualdj_baseline_manifest.py",
    "--virtualdj-data-root",
    "/virtualdj_data",
    "--export-dir",
    "/virtualdj_export",
    "--output-json",
    $ManifestOutput,
    "--host-probe-json",
    $hostProbeContainerPath
)

Push-Location $ProjectRoot
try {
    & docker @manifestDockerArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    if ($RunQualityGate) {
        if (-not (Test-Path -LiteralPath $manifestHostPath -PathType Leaf)) {
            throw "VirtualDJ manifest was not written: $manifestHostPath"
        }

        $manifest = Get-Content -Raw -LiteralPath $manifestHostPath | ConvertFrom-Json
        $referenceMinus = $manifest.quality_gate_hint.reference_minus_docker_path
        if ([string]::IsNullOrWhiteSpace($referenceMinus)) {
            throw "No VirtualDJ instrumental export found. Put an Instrumental/Karaoke/Minus audio file into ExportDir first: $ExportDir"
        }

        $fullMixContainerPath = Convert-HostProjectPathToContainerPath -HostPath $FullMix -Root $ProjectRoot
        $gateDockerArgs = @(
            "compose",
            "run",
            "--rm",
            "--no-deps",
            "-v",
            "${ProjectRoot}:/host_project:ro",
            "-v",
            "${ExportDir}:/virtualdj_export:ro",
            "--entrypoint",
            "python",
            "ml-worker",
            "/app/scripts/run_nine_thou_quality_gate.py",
            "--full-mix",
            $fullMixContainerPath,
            "--reference-minus",
            $referenceMinus,
            "--output-dir",
            $OutputDir
        )

        if ($GateArgs) {
            $gateDockerArgs += $GateArgs
        }

        & docker @gateDockerArgs
        exit $LASTEXITCODE
    }

    Write-Host "VirtualDJ manifest: $manifestHostPath"
    Write-Host "VirtualDJ host probe: $hostProbePath"
    Write-Host "Export folder: $ExportDir"
    exit 0
}
finally {
    Pop-Location
    if ($CloseVirtualDjWhenDone -and $LaunchVirtualDj) {
        try {
            $candidateProcessIds = @()
            if ($startedVirtualDjProcess -ne $null) {
                $candidateProcessIds += $startedVirtualDjProcess.Id
            }

            if ($preLaunchVirtualDjProcessIds.Count -eq 0 -and $virtualDjLaunchStartedAt -ne $null) {
                $safeChildCandidates = @(
                    Get-Process -Name "virtualdj" -ErrorAction SilentlyContinue |
                        Where-Object {
                            $preLaunchVirtualDjProcessIds -notcontains $_.Id -and
                            $_.Path -eq $VirtualDjExe -and
                            $_.StartTime -ge $virtualDjLaunchStartedAt.AddSeconds(-2)
                        }
                )
                if ($safeChildCandidates.Count -eq 1) {
                    $candidateProcessIds += $safeChildCandidates[0].Id
                }
                elseif ($safeChildCandidates.Count -gt 1) {
                    Write-Warning "Not closing VirtualDJ child process because ownership is ambiguous: $($safeChildCandidates.Id -join ', ')"
                }
            }

            foreach ($processId in ($candidateProcessIds | Select-Object -Unique)) {
                $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
                if (-not $process) {
                    continue
                }

                $closed = $process.CloseMainWindow()
                if ($closed) {
                    Wait-Process -Id $process.Id -Timeout 15 -ErrorAction SilentlyContinue
                }
                $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
                if ($process) {
                    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                }
            }
        }
        catch {
            Write-Warning "Failed to close VirtualDJ process started by this wrapper: $($_.Exception.Message)"
        }
    }
}
