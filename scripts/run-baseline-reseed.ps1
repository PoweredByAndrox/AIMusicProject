param(
    [string]$ApiBaseUrl = 'http://localhost:8000/api',
    [int]$PilotTimeoutMinutes = 20,
    [int]$ImportTimeoutMinutes = 180,
    [int]$AnalysisTimeoutMinutes = 180,
    [int]$TrainingTimeoutMinutes = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$RunStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$ArtifactDir = Join-Path $RepoRoot "lab_artifacts\baseline_reseed_$RunStamp"
$RunLogPath = Join-Path $ArtifactDir 'run.log'
$StatusPath = Join-Path $ArtifactDir 'status.json'
$YouTubeLogPath = Join-Path $RepoRoot 'storage\logs\youtube-download.log'
$AudioRoot = Join-Path $RepoRoot 'docker_data\storage\audio'
$StorageRoot = Join-Path $RepoRoot 'docker_data\storage'

New-Item -ItemType Directory -Path $ArtifactDir -Force | Out-Null

$script:State = [ordered]@{
    run_stamp = $RunStamp
    artifact_dir = $ArtifactDir
    phase = 'starting'
    started_at = (Get-Date).ToString('o')
    completed_at = $null
    error = $null
    pilot = $null
    reset = $null
    reseed = $null
    evaluation = $null
}

$script:ServicesStopped = $false
$script:ServicesRestarted = $false
$script:AdminCredentials = $null

function Write-RunLog {
    param(
        [string]$Message,
        [string]$Level = 'INFO'
    )

    $line = "[{0}] [{1}] {2}" -f (Get-Date).ToString('o'), $Level.ToUpperInvariant(), $Message
    $line | Out-File -FilePath $RunLogPath -Encoding utf8 -Append
    Write-Host $line
}

function Write-State {
    param(
        [string]$Phase,
        [hashtable]$Extra = @{}
    )

    $script:State.phase = $Phase
    foreach ($key in $Extra.Keys) {
        $script:State[$key] = $Extra[$key]
    }
    $script:State | ConvertTo-Json -Depth 20 | Out-File -FilePath $StatusPath -Encoding utf8
}

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

function Get-FileCount {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return 0
    }

    return (Get-ChildItem -Path $Path -File -Recurse -Force | Measure-Object).Count
}

function Get-LineCount {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return 0
    }

    return (Get-Content -Path $Path | Measure-Object -Line).Lines
}

function Normalize-YouTubeUrl {
    param([string]$Url)

    $trimmed = $Url.Trim()
    if ($trimmed -match '^https?://(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{11})(?:\?.*)?$') {
        return "https://www.youtube.com/watch?v=$($matches[1])"
    }

    return $trimmed
}

function Normalize-UrlList {
    param([string]$RawBlock)

    $result = New-Object System.Collections.Generic.List[string]
    $seen = @{}

    foreach ($line in ($RawBlock -split "`r?`n")) {
        $trimmed = $line.Trim()
        if ($trimmed -eq '') {
            continue
        }

        $normalized = Normalize-YouTubeUrl -Url $trimmed
        if ($seen.ContainsKey($normalized)) {
            continue
        }

        $seen[$normalized] = $true
        $result.Add($normalized)
    }

    return ,$result.ToArray()
}

function Get-VideoIdFromUrl {
    param([string]$Url)

    $patterns = @(
        '(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([A-Za-z0-9_-]{11})'
    )

    foreach ($pattern in $patterns) {
        if ($Url -match $pattern) {
            return $matches[1]
        }
    }

    throw "Could not extract video id from URL: $Url"
}

function New-RandomPassword {
    return "Codex!{0}aA" -f ([guid]::NewGuid().ToString('N').Substring(0, 20))
}

function New-CredentialObject {
    param([string]$Prefix)

    $token = ([guid]::NewGuid().ToString('N').Substring(0, 12)).ToLowerInvariant()
    return [ordered]@{
        name = "$Prefix-$token"
        email = "$Prefix-$token@local.test"
        password = New-RandomPassword
    }
}

function Invoke-ComposeText {
    param([string[]]$ComposeArgs)

    Write-RunLog ("docker compose " + ($ComposeArgs -join ' '))
    $stdoutPath = Join-Path $ArtifactDir ("compose_stdout_" + [guid]::NewGuid().ToString('N') + '.log')
    $stderrPath = Join-Path $ArtifactDir ("compose_stderr_" + [guid]::NewGuid().ToString('N') + '.log')

    try {
        $dockerArgs = @('compose') + $ComposeArgs
        $proc = Start-Process -FilePath 'docker' `
            -ArgumentList $dockerArgs `
            -WorkingDirectory $RepoRoot `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath

        $stdout = if (Test-Path $stdoutPath) { Get-Content $stdoutPath } else { @() }
        $stderr = if (Test-Path $stderrPath) { Get-Content $stderrPath } else { @() }

        foreach ($line in @($stdout + $stderr)) {
            if (-not [string]::IsNullOrWhiteSpace([string]$line)) {
                Write-RunLog $line 'DOCKER'
            }
        }

        if ($proc.ExitCode -ne 0) {
            throw "docker compose failed with exit code $($proc.ExitCode): $($ComposeArgs -join ' ')"
        }

        return (@($stdout + $stderr) -join "`n").Trim()
    } finally {
        Remove-Item $stdoutPath, $stderrPath -ErrorAction SilentlyContinue
    }
}

function Invoke-PsqlText {
    param([string]$Sql)

    $output = & docker compose exec -T postgres psql -U postgres -d music_recommendations -At -c $Sql 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "psql failed with exit code ${exitCode}: $($output -join "`n")"
    }

    return ($output -join "`n").Trim()
}

function ConvertFrom-JsonSafe {
    param([string]$Raw)

    if ([string]::IsNullOrWhiteSpace($Raw)) {
        return $null
    }

    return $Raw | ConvertFrom-Json
}

function Invoke-PsqlJson {
    param([string]$Sql)
    return ConvertFrom-JsonSafe -Raw (Invoke-PsqlText -Sql $Sql)
}

function Quote-SqlLiteral {
    param([string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

function Join-SqlStringList {
    param([string[]]$Values)
    return ($Values | ForEach-Object { Quote-SqlLiteral -Value $_ }) -join ', '
}

function Invoke-ApiRequest {
    param(
        [ValidateSet('GET', 'POST', 'PUT', 'DELETE')]
        [string]$Method,
        [string]$Path,
        [object]$Body = $null,
        [string]$Token = $null
    )

    $uri = ($ApiBaseUrl.TrimEnd('/') + '/' + $Path.TrimStart('/'))
    $headers = @{
        Accept = 'application/json'
    }

    if ($Token) {
        $headers.Authorization = "Bearer $Token"
    }

    $params = @{
        Method = $Method
        Uri = $uri
        Headers = $headers
        UseBasicParsing = $true
    }

    if ($null -ne $Body) {
        $params.ContentType = 'application/json'
        $params.Body = ($Body | ConvertTo-Json -Depth 20)
    }

    Write-RunLog ("HTTP {0} {1}" -f $Method, $uri)

    try {
        $response = Invoke-WebRequest @params
        $bodyObject = $null
        if (-not [string]::IsNullOrWhiteSpace($response.Content)) {
            $bodyObject = $response.Content | ConvertFrom-Json
        }

        return [pscustomobject]@{
            StatusCode = [int]$response.StatusCode
            Body = $bodyObject
            Raw = $response.Content
        }
    } catch {
        if ($_.Exception.Response) {
            $response = $_.Exception.Response
            $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
            $raw = $reader.ReadToEnd()
            $bodyObject = $null
            if (-not [string]::IsNullOrWhiteSpace($raw)) {
                try {
                    $bodyObject = $raw | ConvertFrom-Json
                } catch {
                    $bodyObject = $null
                }
            }

            return [pscustomobject]@{
                StatusCode = [int]$response.StatusCode
                Body = $bodyObject
                Raw = $raw
            }
        }

        throw
    }
}

function Wait-Until {
    param(
        [string]$Name,
        [int]$TimeoutSeconds,
        [int]$SleepSeconds,
        [scriptblock]$Probe
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $last = $null

    while ((Get-Date) -lt $deadline) {
        $last = & $Probe
        if ($last.Success) {
            Write-RunLog ("Wait succeeded for {0}: {1}" -f $Name, (($last.State | ConvertTo-Json -Compress -Depth 10)))
            return $last
        }

        Write-RunLog ("Wait pending for {0}: {1}" -f $Name, (($last.State | ConvertTo-Json -Compress -Depth 10)))
        Start-Sleep -Seconds $SleepSeconds
    }

    throw ("Timeout while waiting for {0}. Last state: {1}" -f $Name, (($last.State | ConvertTo-Json -Compress -Depth 10)))
}

function Get-AppendedJsonLogEntries {
    param(
        [string]$Path,
        [int]$SkipLines
    )

    if (-not (Test-Path $Path)) {
        return @()
    }

    $allLines = Get-Content -Path $Path
    if ($allLines.Count -le $SkipLines) {
        return @()
    }

    $entries = New-Object System.Collections.Generic.List[object]
    foreach ($line in $allLines[$SkipLines..($allLines.Count - 1)]) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        try {
            $entries.Add(($line | ConvertFrom-Json))
        } catch {
        }
    }

    return @($entries.ToArray())
}

function Export-Json {
    param(
        [string]$Path,
        [object]$Data
    )

    $Data | ConvertTo-Json -Depth 40 | Out-File -FilePath $Path -Encoding utf8
}

function Export-JsonLines {
    param(
        [string]$Path,
        [object[]]$Entries
    )

    $lines = foreach ($entry in $Entries) {
        $entry | ConvertTo-Json -Compress -Depth 40
    }
    $lines | Out-File -FilePath $Path -Encoding utf8
}

function Wait-For-Health {
    Wait-Until -Name 'api_health' -TimeoutSeconds 180 -SleepSeconds 3 -Probe {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri ($ApiBaseUrl.TrimEnd('/') + '/health')
            $raw = $response.Content | ConvertFrom-Json
            return @{
                Success = ([int]$response.StatusCode -eq 200 -and $raw.status -eq 'ok')
                State = @{
                    status_code = [int]$response.StatusCode
                    body = $raw
                }
            }
        } catch {
            return @{
                Success = $false
                State = @{
                    error = $_.Exception.Message
                }
            }
        }
    } | Out-Null
}

function Get-TrackRowsByVideoIds {
    param([string[]]$VideoIds)

    if ($VideoIds.Count -eq 0) {
        return @()
    }

    $sql = @"
SELECT COALESCE(json_agg(row_to_json(t))::text, '[]')
FROM (
    SELECT id, title, artist, source, youtube_url, youtube_video_id, youtube_channel, file_path, analysis_status, embedding_path, created_at, updated_at
    FROM tracks
    WHERE youtube_video_id IN ($(Join-SqlStringList -Values $VideoIds))
    ORDER BY id
) t;
"@

    $rows = Invoke-PsqlJson -Sql $sql
    if ($null -eq $rows) {
        return @()
    }

    return @($rows)
}

function Get-TrackSummaryByVideoIds {
    param([string[]]$VideoIds)

    if ($VideoIds.Count -eq 0) {
        return [ordered]@{}
    }

    $sql = @"
SELECT json_build_object(
    'matched_tracks', COUNT(*),
    'distinct_video_ids', COUNT(DISTINCT youtube_video_id),
    'pending', COUNT(*) FILTER (WHERE analysis_status = 'pending'),
    'processing', COUNT(*) FILTER (WHERE analysis_status = 'processing'),
    'completed', COUNT(*) FILTER (WHERE analysis_status = 'completed'),
    'failed', COUNT(*) FILTER (WHERE analysis_status = 'failed'),
    'with_embeddings', COUNT(*) FILTER (WHERE embedding_path IS NOT NULL)
)::text
FROM tracks
WHERE youtube_video_id IN ($(Join-SqlStringList -Values $VideoIds));
"@

    return Invoke-PsqlJson -Sql $sql
}

function Get-AllTrackSummary {
    $sql = @"
SELECT json_build_object(
    'total_tracks', COUNT(*),
    'pending', COUNT(*) FILTER (WHERE analysis_status = 'pending'),
    'processing', COUNT(*) FILTER (WHERE analysis_status = 'processing'),
    'completed', COUNT(*) FILTER (WHERE analysis_status = 'completed'),
    'failed', COUNT(*) FILTER (WHERE analysis_status = 'failed'),
    'with_embeddings', COUNT(*) FILTER (WHERE embedding_path IS NOT NULL)
)::text
FROM tracks;
"@

    return Invoke-PsqlJson -Sql $sql
}

function Get-FailedQueueTasks {
    $sql = @"
SELECT COALESCE(json_agg(row_to_json(t))::text, '[]')
FROM (
    SELECT id, task_type, status, payload, error_message, created_at
    FROM queue_tasks
    WHERE status = 'failed'
    ORDER BY id
) t;
"@
    $rows = Invoke-PsqlJson -Sql $sql
    if ($null -eq $rows) {
        return @()
    }
    return @($rows)
}

function Get-UserSummary {
    $sql = @"
SELECT COALESCE(json_agg(row_to_json(t))::text, '[]')
FROM (
    SELECT id, name, email, role, created_at
    FROM users
    ORDER BY id
) t;
"@
    $rows = Invoke-PsqlJson -Sql $sql
    if ($null -eq $rows) {
        return @()
    }
    return @($rows)
}

function Get-SidecarPathForTrack {
    param([psobject]$TrackRow)

    $basename = [System.IO.Path]::GetFileName([string]$TrackRow.file_path)
    return (Join-Path $AudioRoot ($basename + '.meta.json'))
}

function Get-AudioPathForTrack {
    param([psobject]$TrackRow)

    $basename = [System.IO.Path]::GetFileName([string]$TrackRow.file_path)
    return (Join-Path $AudioRoot $basename)
}

function Get-ResetDirectoryState {
    param([string[]]$DirectoryNames)

    $state = [ordered]@{}
    foreach ($name in $DirectoryNames) {
        $path = Join-Path $StorageRoot $name
        $state[$name] = [ordered]@{
            path = $path
            exists = (Test-Path $path)
            file_count = (Get-FileCount -Path $path)
        }
    }

    return $state
}

$EtalonUrls = Normalize-UrlList -RawBlock @'
https://www.youtube.com/watch?v=MhM7Dp5cYlA
https://www.youtube.com/watch?v=Cq1B3TP-hFU
https://www.youtube.com/watch?v=QUIpbeTuzCs
https://www.youtube.com/watch?v=4krJsLGxipk
https://www.youtube.com/watch?v=Xbo-xCRSk1Y
https://www.youtube.com/watch?v=QsBShN9701c
https://www.youtube.com/watch?v=cYyrfI_B_rc
https://www.youtube.com/watch?v=mEFzMNUFIr0
https://www.youtube.com/watch?v=QksUPPKJsSE
https://www.youtube.com/watch?v=rYMEQoEYubE
'@

$RandomUrls = Normalize-UrlList -RawBlock @'
https://www.youtube.com/watch?v=_UMyEXDUVuQ
https://www.youtube.com/watch?v=dyVDs9szSiI
https://www.youtube.com/watch?v=sV1WclMga40
https://www.youtube.com/watch?v=lWy15941dpc
https://www.youtube.com/watch?v=pRSGlzU7eZU
https://www.youtube.com/watch?v=W0_MvSXdKrg
https://www.youtube.com/watch?v=lTRUVzIKhOI
https://www.youtube.com/watch?v=rc1nW1F9yQ0
https://www.youtube.com/watch?v=2tzFqdgWcb4
https://www.youtube.com/watch?v=pjOro9JGDSk
https://www.youtube.com/watch?v=QoTkIWZb64I
https://www.youtube.com/watch?v=oWVpIzbIyqQ
https://www.youtube.com/watch?v=FjCXJW5ofPI
https://www.youtube.com/watch?v=saIFdkXk-ew
https://www.youtube.com/watch?v=aFtNKbkrc0I
https://www.youtube.com/watch?v=Rv7TnlqySMM
https://www.youtube.com/watch?v=PARuttPMNqU
https://www.youtube.com/watch?v=hbl2wb_ZH2k
https://www.youtube.com/watch?v=S4135GccPWM
https://www.youtube.com/watch?v=m4gRlgU99YE
https://www.youtube.com/watch?v=lpItUsvfWLU
https://www.youtube.com/watch?v=GSBxf3W2sWs
https://www.youtube.com/watch?v=sILsB0oaQHk
https://www.youtube.com/watch?v=GPDKOWx3uGI
https://www.youtube.com/watch?v=UPG8Oy_qoYA
https://www.youtube.com/watch?v=-aft6MS1e2k
https://www.youtube.com/watch?v=fkVHBmqRMWg
https://www.youtube.com/watch?v=WlcHI4nV9hY
https://www.youtube.com/watch?v=py7QZa1XaJM
https://www.youtube.com/shorts/rxHtcEoyFao
https://www.youtube.com/watch?v=xLV-5KXOo7c
https://www.youtube.com/watch?v=gdS352ujN7I
https://www.youtube.com/watch?v=PTLRNSukPyQ
https://www.youtube.com/watch?v=71FGUXEVDZE
https://www.youtube.com/watch?v=RBeDlavEuYk
https://www.youtube.com/watch?v=Vn68_i3zt5g
https://www.youtube.com/watch?v=5YjhTrskCbM
https://www.youtube.com/watch?v=99uuu6CfPzI
https://www.youtube.com/watch?v=ScHgUyeRRf8
https://www.youtube.com/watch?v=LLVC6CQk_fU
https://www.youtube.com/watch?v=dEh8wf327WI
https://www.youtube.com/watch?v=Aj8UPSndYhU
https://www.youtube.com/watch?v=lcyb2PWG4RU
https://www.youtube.com/watch?v=vFNGBS5vh5w
https://www.youtube.com/watch?v=c_b8ohunUis
https://www.youtube.com/watch?v=LxFBO82eryk
https://www.youtube.com/watch?v=uKGg_yf5ljE
https://www.youtube.com/watch?v=v5XL7ZZpk94
https://www.youtube.com/watch?v=Ws_ZKmAU2RQ
https://www.youtube.com/watch?v=X6ZoLnSJ_GE
https://www.youtube.com/watch?v=yJFZeCNKSKM
https://www.youtube.com/watch?v=o8aUb2ftX5k
https://www.youtube.com/watch?v=HoH1PyqdhiY
https://www.youtube.com/watch?v=4jTSPhTe79Y
https://www.youtube.com/watch?v=LikQOoQ-g30
https://www.youtube.com/watch?v=FGvYM6q9WmY
https://www.youtube.com/watch?v=bPukSnz7XJM
https://www.youtube.com/watch?v=wAMwe1FYeQg
https://www.youtube.com/watch?v=GupDaKN-16Q
https://www.youtube.com/watch?v=PHeoYcglrjQ
https://www.youtube.com/watch?v=kKmKQoR5Dck
https://www.youtube.com/watch?v=bwyzhWHFydM
https://www.youtube.com/watch?v=uPQLJTSp3hs
https://www.youtube.com/watch?v=MLNCIh7XmIU
https://www.youtube.com/watch?v=ZImirsnTtKI
https://www.youtube.com/watch?v=jtA6OC7X3IE
https://www.youtube.com/watch?v=cHIU2nLGyrE
https://www.youtube.com/watch?v=orCdlQTrPKg
https://www.youtube.com/watch?v=RuR6YE9sgiI
https://www.youtube.com/watch?v=XBT6OXztPPc
https://www.youtube.com/watch?v=KWM0tppKIKs
https://www.youtube.com/watch?v=SopiJ0t_C8I
https://www.youtube.com/watch?v=2dmQdISvYt8
https://www.youtube.com/watch?v=nD2OdniU6Hc
https://www.youtube.com/watch?v=aQLVUsuQ9dA
https://www.youtube.com/watch?v=VSUSSHCirCE
https://www.youtube.com/watch?v=Gvu-vEGJBm8
https://www.youtube.com/watch?v=l3l9tN6xkeA
https://www.youtube.com/watch?v=suxxGMe9nlI
https://www.youtube.com/watch?v=pPX8lgW1qsY
https://www.youtube.com/watch?v=3Kc7D563_s4
https://www.youtube.com/watch?v=2yyW6JzIfXg
https://www.youtube.com/watch?v=ZaRtf9OTHfI
https://www.youtube.com/watch?v=5KTPGDMqKBQ
https://www.youtube.com/watch?v=tWVGXIkGWB4
https://www.youtube.com/watch?v=DtVQa7Q6pZU
https://www.youtube.com/watch?v=56NjLjIlquU
https://www.youtube.com/watch?v=Rs7fA8qGUlk
https://www.youtube.com/watch?v=sOFYc23IUoU
https://www.youtube.com/watch?v=ZrYd6WS3m0E
https://www.youtube.com/watch?v=M2gyq41A-_s
https://www.youtube.com/watch?v=Kn807OEQ2G8
https://www.youtube.com/watch?v=tYVRzl5YFdA
https://www.youtube.com/watch?v=pobh5ZvzTrE
https://www.youtube.com/watch?v=EWW4Yy-CCIk
https://www.youtube.com/watch?v=NIVgXfoBHFk
https://www.youtube.com/watch?v=yd-eNov8t8A
https://www.youtube.com/watch?v=knkunvJBpNg
https://www.youtube.com/watch?v=2rfx7o5_kfk
https://www.youtube.com/watch?v=06wjEXKesQY
'@

$AllUrls = @($EtalonUrls + $RandomUrls)
$AllVideoIds = @($AllUrls | ForEach-Object { Get-VideoIdFromUrl -Url $_ })
$PilotUrl = $EtalonUrls[0]
$PilotVideoId = Get-VideoIdFromUrl -Url $PilotUrl
$EtalonVideoIds = @($EtalonUrls | ForEach-Object { Get-VideoIdFromUrl -Url $_ })
$ResetDirectoryNames = @(
    'audio',
    'embeddings',
    'patterns',
    'analysis_cache',
    'public_radio_hls',
    'public_radio_variants',
    'public_radio_waveforms',
    'tmp'
)

Assert-True ($EtalonUrls.Count -eq 10) "Expected 10 etalon URLs after normalization, got $($EtalonUrls.Count)"
Assert-True ($RandomUrls.Count -eq 100) "Expected 100 random URLs after normalization, got $($RandomUrls.Count)"
Assert-True ($AllUrls.Count -eq 110) "Expected 110 total URLs after dedupe, got $($AllUrls.Count)"
Assert-True ((@($AllVideoIds | Select-Object -Unique)).Count -eq 110) 'Expected 110 unique video ids after normalization'

Export-Json -Path (Join-Path $ArtifactDir 'url_manifest.json') -Data ([ordered]@{
    pilot_url = $PilotUrl
    etalon_urls = $EtalonUrls
    random_urls = $RandomUrls
    all_urls = $AllUrls
    etalon_video_ids = $EtalonVideoIds
    all_video_ids = $AllVideoIds
})

$EtalonUrls | Out-File -FilePath (Join-Path $ArtifactDir 'etalon_urls.txt') -Encoding utf8
$RandomUrls | Out-File -FilePath (Join-Path $ArtifactDir 'random_urls.txt') -Encoding utf8
$AllUrls | Out-File -FilePath (Join-Path $ArtifactDir 'all_urls_normalized.txt') -Encoding utf8

Write-RunLog "Artifacts directory: $ArtifactDir"
Write-State -Phase 'preflight' -Extra @{
    url_manifest = [ordered]@{
        pilot_url = $PilotUrl
        etalon_count = $EtalonUrls.Count
        random_count = $RandomUrls.Count
        total_count = $AllUrls.Count
    }
}

try {
    Write-RunLog 'Checking API health'
    Wait-For-Health

    $cachePreCount = Get-FileCount -Path (Join-Path $StorageRoot 'cache')
    $logsPreCount = Get-FileCount -Path (Join-Path $StorageRoot 'logs')

    if (Test-Path $YouTubeLogPath) {
        Copy-Item -Path $YouTubeLogPath -Destination (Join-Path $ArtifactDir 'youtube-download.pre-pilot.log') -Force
        Write-RunLog 'Archived current youtube-download.log before pilot'
    }

    $pilotUser = New-CredentialObject -Prefix 'pilot'
    Export-Json -Path (Join-Path $ArtifactDir 'pilot_user_credentials.json') -Data $pilotUser

    $pilotRegisterResponse = Invoke-ApiRequest -Method POST -Path 'auth/register' -Body @{
        name = $pilotUser.name
        email = $pilotUser.email
        password = $pilotUser.password
        password_confirmation = $pilotUser.password
    }

    Export-Json -Path (Join-Path $ArtifactDir 'pilot_register_response.json') -Data $pilotRegisterResponse
    Assert-True ($pilotRegisterResponse.StatusCode -eq 201) "Pilot register failed with status $($pilotRegisterResponse.StatusCode)"
    Assert-True (-not [string]::IsNullOrWhiteSpace([string]$pilotRegisterResponse.Body.token)) 'Pilot register did not return token'

    Write-State -Phase 'pilot_import_request' -Extra @{
        pilot = [ordered]@{
            user_email = $pilotUser.email
            register_status_code = $pilotRegisterResponse.StatusCode
        }
    }

    $pilotLogSkip = Get-LineCount -Path $YouTubeLogPath
    $pilotImportResponse = Invoke-ApiRequest -Method POST -Path 'tracks/youtube' -Token $pilotRegisterResponse.Body.token -Body @{
        url = $PilotUrl
        like_all = $false
    }
    Export-Json -Path (Join-Path $ArtifactDir 'pilot_import_response.json') -Data $pilotImportResponse
    Assert-True ($pilotImportResponse.StatusCode -eq 202) "Pilot import failed with status $($pilotImportResponse.StatusCode)"

    Wait-Until -Name 'pilot_track_and_logs' -TimeoutSeconds ($PilotTimeoutMinutes * 60) -SleepSeconds 5 -Probe {
        $entries = @(Get-AppendedJsonLogEntries -Path $YouTubeLogPath -SkipLines $pilotLogSkip |
            Where-Object { $_.context.url -eq $PilotUrl })

        $messages = @($entries | ForEach-Object { [string]$_.message })
        $trackRows = @(Get-TrackRowsByVideoIds -VideoIds @($PilotVideoId))
        $trackRow = if ($trackRows.Count -gt 0) { $trackRows[-1] } else { $null }

        $audioExists = $false
        $sidecarExists = $false
        if ($null -ne $trackRow) {
            $audioExists = Test-Path (Get-AudioPathForTrack -TrackRow $trackRow)
            $sidecarExists = Test-Path (Get-SidecarPathForTrack -TrackRow $trackRow)
        }

        $success = (
            ($messages -contains 'job_started') -and
            ($messages -contains 'video_import_started') -and
            (($messages -contains 'track_created') -or ($messages -contains 'duplicate_by_video_id') -or ($messages -contains 'duplicate_by_content')) -and
            ($messages -contains 'track_analysis_dispatched') -and
            ($null -ne $trackRow) -and
            ([string]$trackRow.source -eq 'youtube') -and
            ([string]$trackRow.youtube_video_id -eq $PilotVideoId) -and
            (-not [string]::IsNullOrWhiteSpace([string]$trackRow.youtube_url)) -and
            (-not [string]::IsNullOrWhiteSpace([string]$trackRow.file_path)) -and
            $audioExists -and
            $sidecarExists -and
            (-not ([string]$trackRow.title).StartsWith('YouTube Import '))
        )

        return @{
            Success = $success
            State = @{
                messages = $messages
                track_count = $trackRows.Count
                audio_exists = $audioExists
                sidecar_exists = $sidecarExists
                latest_track = $trackRow
            }
        }
    } | Out-Null

    $pilotEntries = @(Get-AppendedJsonLogEntries -Path $YouTubeLogPath -SkipLines $pilotLogSkip |
        Where-Object { $_.context.url -eq $PilotUrl })
    $pilotTrack = @(Get-TrackRowsByVideoIds -VideoIds @($PilotVideoId))[-1]
    $pilotSidecarPath = Get-SidecarPathForTrack -TrackRow $pilotTrack

    Export-Json -Path (Join-Path $ArtifactDir 'pilot_track.json') -Data $pilotTrack
    Export-JsonLines -Path (Join-Path $ArtifactDir 'pilot_youtube_log.jsonl') -Entries $pilotEntries
    if (Test-Path $pilotSidecarPath) {
        Copy-Item -Path $pilotSidecarPath -Destination (Join-Path $ArtifactDir 'pilot_track.meta.json') -Force
    }

    Write-State -Phase 'pilot_passed' -Extra @{
        pilot = [ordered]@{
            import_status_code = $pilotImportResponse.StatusCode
            track = $pilotTrack
            log_event_messages = @($pilotEntries | ForEach-Object { $_.message })
            sidecar_path = $pilotSidecarPath
        }
    }

    Write-RunLog 'Pilot gate passed'

    $preResetDirState = Get-ResetDirectoryState -DirectoryNames $ResetDirectoryNames
    $preResetUsers = @(Get-UserSummary)
    $preResetTrackSummary = Get-AllTrackSummary
    Export-Json -Path (Join-Path $ArtifactDir 'pre_reset_directory_state.json') -Data $preResetDirState
    Export-Json -Path (Join-Path $ArtifactDir 'pre_reset_users.json') -Data $preResetUsers
    Export-Json -Path (Join-Path $ArtifactDir 'pre_reset_track_summary.json') -Data $preResetTrackSummary

    Write-State -Phase 'reset_stopping_services' -Extra @{
        reset = [ordered]@{
            pre_reset_directory_state = $preResetDirState
            pre_reset_track_summary = $preResetTrackSummary
        }
    }

    Invoke-ComposeText -ComposeArgs @('stop', 'app', 'ml-worker', 'ml-worker-2', 'ml-worker-3')
    $script:ServicesStopped = $true

    Invoke-ComposeText -ComposeArgs @('exec', '-T', 'redis', 'redis-cli', '-n', '0', 'FLUSHDB')

    foreach ($directoryName in $ResetDirectoryNames) {
        $path = Join-Path $StorageRoot $directoryName
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Path $path -Force | Out-Null
        }

        foreach ($child in (Get-ChildItem -Path $path -Force)) {
            Remove-Item -Path $child.FullName -Recurse -Force
        }
    }

    $postWipeDirState = Get-ResetDirectoryState -DirectoryNames $ResetDirectoryNames
    Export-Json -Path (Join-Path $ArtifactDir 'post_wipe_directory_state.json') -Data $postWipeDirState
    foreach ($directoryName in $ResetDirectoryNames) {
        Assert-True ($postWipeDirState[$directoryName].file_count -eq 0) "Directory $directoryName is not empty after wipe"
    }

    Write-State -Phase 'reset_migrate_fresh'
    Invoke-ComposeText -ComposeArgs @('run', '--rm', 'app', 'php', '/var/www/html/artisan', 'migrate:fresh', '--force')

    Write-State -Phase 'reset_starting_services'
    Invoke-ComposeText -ComposeArgs @('up', '-d', 'app', 'ml-worker', 'ml-worker-2', 'ml-worker-3')
    $script:ServicesRestarted = $true

    Wait-For-Health

    $postResetUsers = @(Get-UserSummary)
    $postResetTracks = Get-AllTrackSummary
    $cachePostCount = Get-FileCount -Path (Join-Path $StorageRoot 'cache')
    $logsPostCount = Get-FileCount -Path (Join-Path $StorageRoot 'logs')

    Export-Json -Path (Join-Path $ArtifactDir 'post_reset_users.json') -Data $postResetUsers
    Export-Json -Path (Join-Path $ArtifactDir 'post_reset_track_summary.json') -Data $postResetTracks

    Assert-True ($postResetUsers.Count -eq 0) "Expected zero users after migrate:fresh, got $($postResetUsers.Count)"
    Assert-True ([int]$postResetTracks.total_tracks -eq 0) "Expected zero tracks after migrate:fresh, got $($postResetTracks.total_tracks)"
    Assert-True (Test-Path (Join-Path $StorageRoot 'cache')) 'Cache directory missing after reset'
    Assert-True (Test-Path (Join-Path $StorageRoot 'logs')) 'Runtime storage log directory missing after reset'
    Assert-True (Test-Path (Join-Path $RepoRoot 'storage\logs')) 'Application log directory missing after reset'
    Assert-True (Test-Path $YouTubeLogPath) 'youtube-download.log missing after reset'
    if ($cachePreCount -gt 0) {
        Assert-True ($cachePostCount -gt 0) 'Cache files were present before reset but disappeared after reset'
    }
    if ($logsPreCount -gt 0) {
        Assert-True ($logsPostCount -gt 0) 'Runtime log files were present before reset but disappeared after reset'
    }

    $adminUser = New-CredentialObject -Prefix 'admin'
    $script:AdminCredentials = $adminUser
    Export-Json -Path (Join-Path $ArtifactDir 'admin_credentials.json') -Data $adminUser

    $adminRegisterResponse = Invoke-ApiRequest -Method POST -Path 'auth/register' -Body @{
        name = $adminUser.name
        email = $adminUser.email
        password = $adminUser.password
        password_confirmation = $adminUser.password
    }
    Export-Json -Path (Join-Path $ArtifactDir 'admin_register_response.json') -Data $adminRegisterResponse
    Assert-True ($adminRegisterResponse.StatusCode -eq 201) "Admin register failed with status $($adminRegisterResponse.StatusCode)"
    Assert-True ([string]$adminRegisterResponse.Body.user.role -eq 'admin') "Expected first registered user to be admin, got $($adminRegisterResponse.Body.user.role)"

    $adminToken = [string]$adminRegisterResponse.Body.token
    Assert-True (-not [string]::IsNullOrWhiteSpace($adminToken)) 'Admin register did not return token'

    Write-State -Phase 'reseed_batch_import_request' -Extra @{
        reset = [ordered]@{
            post_reset_track_summary = $postResetTracks
            cache_file_count_before = $cachePreCount
            cache_file_count_after = $cachePostCount
            log_file_count_before = $logsPreCount
            log_file_count_after = $logsPostCount
        }
        reseed = [ordered]@{
            admin_email = $adminUser.email
            total_urls = $AllUrls.Count
        }
    }

    $reseedLogSkip = Get-LineCount -Path $YouTubeLogPath
    $batchResponse = Invoke-ApiRequest -Method POST -Path 'tracks/youtube/batch' -Token $adminToken -Body @{
        urls = ($AllUrls -join "`n")
        like_all = $false
    }
    Export-Json -Path (Join-Path $ArtifactDir 'batch_import_response.json') -Data $batchResponse
    Assert-True ($batchResponse.StatusCode -eq 202) "Batch import failed with status $($batchResponse.StatusCode)"
    Assert-True ([int]$batchResponse.Body.queued -eq 110) "Expected queued=110, got $($batchResponse.Body.queued)"
    Assert-True ([int]$batchResponse.Body.invalid -eq 0) "Expected invalid=0, got $($batchResponse.Body.invalid)"

    Wait-Until -Name 'reseed_import_tracks' -TimeoutSeconds ($ImportTimeoutMinutes * 60) -SleepSeconds 10 -Probe {
        $summary = Get-TrackSummaryByVideoIds -VideoIds $AllVideoIds
        $entries = @(Get-AppendedJsonLogEntries -Path $YouTubeLogPath -SkipLines $reseedLogSkip |
            Where-Object { $_.context.url -and ($AllUrls -contains [string]$_.context.url) })
        $failedQueueTasks = @(Get-FailedQueueTasks)

        $failedEvents = @($entries | Where-Object { $_.message -eq 'job_failed' })
        $state = @{
            summary = $summary
            failed_events = $failedEvents.Count
            logged_events = $entries.Count
            failed_queue_tasks = $failedQueueTasks.Count
        }

        if ($failedEvents.Count -gt 0) {
            throw "Reseed import encountered job_failed events before reaching 110 tracks"
        }
        if ($failedQueueTasks.Count -gt 0) {
            $firstFailedTask = $failedQueueTasks[0]
            throw "Reseed import encountered failed queue task $($firstFailedTask.id) ($($firstFailedTask.task_type)): $($firstFailedTask.error_message)"
        }
        if ([int]$summary.failed -gt 0) {
            throw "Reseed import encountered tracks with analysis_status=failed before reaching full set"
        }

        return @{
            Success = ([int]$summary.matched_tracks -eq 110 -and [int]$summary.distinct_video_ids -eq 110)
            State = $state
        }
    } | Out-Null

    Wait-Until -Name 'reseed_analysis_complete' -TimeoutSeconds ($AnalysisTimeoutMinutes * 60) -SleepSeconds 15 -Probe {
        $summary = Get-TrackSummaryByVideoIds -VideoIds $AllVideoIds
        $audioFileCount = (Get-ChildItem -Path $AudioRoot -File -Filter '*.mp3' -Force | Measure-Object).Count
        $sidecarFileCount = (Get-ChildItem -Path $AudioRoot -File -Filter '*.meta.json' -Force | Measure-Object).Count

        if ([int]$summary.failed -gt 0) {
            throw "Analysis produced failed tracks before completion"
        }

        return @{
            Success = (
                ([int]$summary.matched_tracks -eq 110) -and
                ([int]$summary.completed -eq 110) -and
                ([int]$summary.with_embeddings -eq 110) -and
                ([int]$summary.pending -eq 0) -and
                ([int]$summary.processing -eq 0) -and
                ($audioFileCount -eq 110) -and
                ($sidecarFileCount -eq 110)
            )
            State = @{
                summary = $summary
                audio_file_count = $audioFileCount
                sidecar_file_count = $sidecarFileCount
            }
        }
    } | Out-Null

    $reseedEntries = @(Get-AppendedJsonLogEntries -Path $YouTubeLogPath -SkipLines $reseedLogSkip |
        Where-Object { $_.context.url -and ($AllUrls -contains [string]$_.context.url) })
    Export-JsonLines -Path (Join-Path $ArtifactDir 'reseed_youtube_log.jsonl') -Entries $reseedEntries

    $importedTracks = @(Get-TrackRowsByVideoIds -VideoIds $AllVideoIds)
    Export-Json -Path (Join-Path $ArtifactDir 'imported_tracks.json') -Data $importedTracks

    $terminalEvents = @('track_created', 'duplicate_by_video_id', 'duplicate_by_content', 'job_failed')
    $terminalCoverage = [ordered]@{}
    foreach ($url in $AllUrls) {
        $eventsForUrl = @($reseedEntries | Where-Object { $_.context.url -eq $url -and ($terminalEvents -contains [string]$_.message) })
        $terminalCoverage[$url] = [ordered]@{
            terminal_event_count = $eventsForUrl.Count
            terminal_messages = @($eventsForUrl | ForEach-Object { $_.message })
        }
    }
    Export-Json -Path (Join-Path $ArtifactDir 'reseed_terminal_event_coverage.json') -Data $terminalCoverage

    foreach ($url in $AllUrls) {
        Assert-True ($terminalCoverage[$url].terminal_event_count -ge 1) "Missing terminal log event for URL $url"
    }

    $etalonTracks = @(Get-TrackRowsByVideoIds -VideoIds $EtalonVideoIds)
    Assert-True ($etalonTracks.Count -eq 10) "Expected 10 etalon tracks after reseed, got $($etalonTracks.Count)"
    Export-Json -Path (Join-Path $ArtifactDir 'etalon_tracks.json') -Data $etalonTracks

    $ratingResponses = New-Object System.Collections.Generic.List[object]
    foreach ($track in $etalonTracks) {
        $ratingResponse = Invoke-ApiRequest -Method POST -Path ("tracks/{0}/rating" -f $track.id) -Token $adminToken -Body @{
            rating = 5
        }
        $ratingResponses.Add([ordered]@{
            track_id = $track.id
            youtube_video_id = $track.youtube_video_id
            status_code = $ratingResponse.StatusCode
            body = $ratingResponse.Body
        })
        Assert-True ($ratingResponse.StatusCode -eq 200) "Rating failed for track $($track.id) with status $($ratingResponse.StatusCode)"
    }
    Export-Json -Path (Join-Path $ArtifactDir 'etalon_rating_responses.json') -Data @($ratingResponses.ToArray())

    $statusBeforeTraining = Invoke-ApiRequest -Method GET -Path 'preference-model/status' -Token $adminToken
    Export-Json -Path (Join-Path $ArtifactDir 'preference_model_status_before.json') -Data $statusBeforeTraining

    $trainResponse = Invoke-ApiRequest -Method POST -Path 'preference-model/train' -Token $adminToken
    Export-Json -Path (Join-Path $ArtifactDir 'preference_model_train_response.json') -Data $trainResponse
    Assert-True ($trainResponse.StatusCode -eq 202) "Training request failed with status $($trainResponse.StatusCode)"

    $trainedStatus = Wait-Until -Name 'preference_model_trained' -TimeoutSeconds ($TrainingTimeoutMinutes * 60) -SleepSeconds 10 -Probe {
        $statusResponse = Invoke-ApiRequest -Method GET -Path 'preference-model/status' -Token $adminToken
        return @{
            Success = (
                $statusResponse.StatusCode -eq 200 -and
                [string]$statusResponse.Body.status -eq 'trained' -and
                -not [string]::IsNullOrWhiteSpace([string]$statusResponse.Body.model_path) -and
                -not [string]::IsNullOrWhiteSpace([string]$statusResponse.Body.trained_at)
            )
            State = @{
                status_code = $statusResponse.StatusCode
                body = $statusResponse.Body
            }
        }
    }
    Export-Json -Path (Join-Path $ArtifactDir 'preference_model_status_after.json') -Data $trainedStatus.State

    $recommendationResponse = Invoke-ApiRequest -Method GET -Path 'recommendations?limit=20' -Token $adminToken
    Export-Json -Path (Join-Path $ArtifactDir 'recommendations_limit_20.json') -Data $recommendationResponse
    $recommendations = @($recommendationResponse.Body.recommendations)
    Assert-True ($recommendationResponse.StatusCode -eq 200) "Recommendations request failed with status $($recommendationResponse.StatusCode)"
    Assert-True ($recommendations.Count -gt 0) 'Recommendations response is empty'

    $recommendedTrackIds = @($recommendations | ForEach-Object { [int]$_.track.id })
    $etalonTrackIds = @($etalonTracks | ForEach-Object { [int]$_.id })
    $intersections = @($recommendedTrackIds | Where-Object { $etalonTrackIds -contains $_ })
    Assert-True ($intersections.Count -eq 0) 'Rated etalon tracks were re-served as recommendations'

    $finalTrackSummary = Get-AllTrackSummary
    Export-Json -Path (Join-Path $ArtifactDir 'final_track_summary.json') -Data $finalTrackSummary

    Write-State -Phase 'completed' -Extra @{
        completed_at = (Get-Date).ToString('o')
        pilot = [ordered]@{
            video_id = $PilotVideoId
            track_id = $pilotTrack.id
            source = $pilotTrack.source
            sidecar_path = $pilotSidecarPath
        }
        reset = [ordered]@{
            cache_file_count_before = $cachePreCount
            cache_file_count_after = $cachePostCount
            log_file_count_before = $logsPreCount
            log_file_count_after = $logsPostCount
        }
        reseed = [ordered]@{
            queued = $batchResponse.Body.queued
            invalid = $batchResponse.Body.invalid
            total_tracks = $finalTrackSummary.total_tracks
            completed_tracks = $finalTrackSummary.completed
            embeddings = $finalTrackSummary.with_embeddings
        }
        evaluation = [ordered]@{
            rated_etalon_count = $ratingResponses.Count
            trained_status = $trainedStatus.State.body.status
            recommendation_count = $recommendations.Count
            recommendation_track_ids = $recommendedTrackIds
        }
    }

    Write-RunLog 'Baseline reseed run completed successfully'
} catch {
    $script:State.error = $_.Exception.Message
    $script:State.completed_at = (Get-Date).ToString('o')
    Write-State -Phase 'failed'
    Write-RunLog $_.Exception.Message 'ERROR'
    throw
} finally {
    if ($script:ServicesStopped -and -not $script:ServicesRestarted) {
        try {
            Write-RunLog 'Recovery: restarting app and ML worker services after failure'
            Invoke-ComposeText -ComposeArgs @('up', '-d', 'app', 'ml-worker', 'ml-worker-2', 'ml-worker-3')
        } catch {
            Write-RunLog ("Recovery restart failed: " + $_.Exception.Message) 'ERROR'
        }
    }
}
