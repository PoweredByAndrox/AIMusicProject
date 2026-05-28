param(
    [int]$Limit = 10,
    [string]$AcoustIdClient = '',
    [switch]$DryRun,
    [switch]$AllowMetadataOnly
)

$ErrorActionPreference = 'Stop'

if ($Limit -lt 1) {
    throw 'Limit must be >= 1.'
}

if ([string]::IsNullOrWhiteSpace($AcoustIdClient)) {
    throw 'AcoustIdClient is required.'
}

$projectRoot = 'D:\AIMusicProject'
$audioMount = 'D:\AIMusicProject\docker_data\storage\audio:/audio'
$helperName = 'codex-acoustid-helper'

function Normalize-Text([string]$value) {
    if ([string]::IsNullOrWhiteSpace($value)) {
        return ''
    }

    $normalized = $value.ToLowerInvariant()
    $normalized = [regex]::Replace($normalized, '[^a-z0-9]+', ' ')
    return ($normalized -replace '\s+', ' ').Trim()
}

function Get-Tokens([string]$value) {
    $normalized = Normalize-Text $value
    if ([string]::IsNullOrWhiteSpace($normalized)) {
        return @()
    }

    return @(
        $normalized.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries) |
            Where-Object { $_.Length -ge 3 } |
            Select-Object -Unique
    )
}

function Get-Flags([string]$value) {
    $flags = @()
    $normalized = Normalize-Text $value

    foreach ($flag in @('remix', 'mix', 'edit', 'live', 'vip', 'bootleg', 'instrumental')) {
        if ($normalized -match "(^| )$flag( |$)") {
            $flags += $flag
        }
    }

    return @($flags | Select-Object -Unique)
}

function Test-TokenCoverage([string[]]$needles, [string]$haystack) {
    if ($needles.Count -eq 0) {
        return $false
    }

    $normalizedHaystack = Normalize-Text $haystack
    $matched = 0
    foreach ($needle in $needles) {
        if ($normalizedHaystack -match "(^| )$([regex]::Escape($needle))( |$)") {
            $matched++
        }
    }

    return ($matched / $needles.Count) -ge 0.6
}

function Escape-Sql([string]$value) {
    return $value.Replace("'", "''")
}

function Start-HelperContainer {
    $existing = & docker ps -aq --filter "name=^${helperName}$"
    if (-not [string]::IsNullOrWhiteSpace(($existing | Select-Object -First 1))) {
        & docker rm -f $helperName | Out-Null
    }
    & docker run -d --rm --name $helperName -v $audioMount alpine:3.20 sh -lc "apk add --no-cache chromaprint ffmpeg >/dev/null && sleep 36000" | Out-Null

    for ($i = 0; $i -lt 180; $i++) {
        $ready = & docker exec $helperName sh -lc "command -v fpcalc" 2>$null
        if (-not [string]::IsNullOrWhiteSpace(($ready | Select-Object -First 1))) {
            return
        }
        Start-Sleep -Seconds 1
    }

    throw 'Helper container did not become ready in time.'
}

function Stop-HelperContainer {
    $existing = & docker ps -aq --filter "name=^${helperName}$"
    if (-not [string]::IsNullOrWhiteSpace(($existing | Select-Object -First 1))) {
        & docker rm -f $helperName | Out-Null
    }
}

function Get-PlaceholderTracks([int]$limit) {
    $sql = @"
COPY (
  SELECT id, file_path, duration
  FROM tracks
  WHERE title LIKE 'YouTube Import %'
  ORDER BY id
  LIMIT $limit
) TO STDOUT WITH CSV HEADER;
"@

    $rows = $sql | docker compose exec -T postgres psql -U postgres -d music_recommendations
    return @($rows | ConvertFrom-Csv)
}

function Get-FingerprintMatch([string]$basename, [string]$client) {
    $out = & docker exec $helperName sh -lc "fpcalc '/audio/$basename'"
    $durationLine = $out | Where-Object { $_ -like 'DURATION=*' } | Select-Object -First 1
    $fingerprintLine = $out | Where-Object { $_ -like 'FINGERPRINT=*' } | Select-Object -First 1

    if ($null -eq $durationLine -or $null -eq $fingerprintLine) {
        return $null
    }

    $duration = [int](($durationLine -split '=', 2)[1])
    $fingerprint = $fingerprintLine.Substring(12)
    $uri = 'https://api.acoustid.org/v2/lookup?client=' + $client + '&meta=recordings+releasegroups+compress&duration=' + $duration + '&fingerprint=' + [uri]::EscapeDataString($fingerprint)
    $resp = $null
    $lastError = $null
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        try {
            $resp = Invoke-RestMethod -Uri $uri -Method Get
            break
        }
        catch {
            $lastError = $_
            Start-Sleep -Seconds ([Math]::Min($attempt * 2, 8))
        }
    }

    if ($null -eq $resp) {
        throw $lastError
    }

    $first = $resp.results | Select-Object -First 1

    if ($null -eq $first) {
        return $null
    }

    $recording = $first.recordings | Select-Object -First 1
    if ($null -eq $recording -or [string]::IsNullOrWhiteSpace($recording.title)) {
        return $null
    }

    $artist = $null
    if ($recording.artists) {
        $artist = ($recording.artists | Select-Object -First 1).name
    }

    if ([string]::IsNullOrWhiteSpace($artist)) {
        return $null
    }

    return [pscustomobject]@{
        score    = [double]$first.score
        title    = [string]$recording.title
        artist   = [string]$artist
        duration = $duration
    }
}

function Get-YouTubeCandidates([string]$query) {
    $lines = $query | docker compose exec -T app python -c "import sys, subprocess; q=sys.stdin.read().rstrip('\n'); subprocess.run(['yt-dlp','--flat-playlist','--no-warnings','--skip-download','ytsearch10:' + q,'--print','%(title)s|%(channel)s|%(duration)s|%(id)s'], check=True)"
    $candidates = @()

    foreach ($line in $lines) {
        $parts = $line -split '\|', 4
        if ($parts.Count -lt 4) {
            continue
        }

        $duration = 0.0
        if (-not [double]::TryParse(
            $parts[2].Trim(),
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$duration
        )) {
            continue
        }

        $candidates += [pscustomobject]@{
            title    = $parts[0].Trim()
            channel  = $parts[1].Trim()
            duration = [int][math]::Round($duration)
            id       = $parts[3].Trim()
        }
    }

    return @($candidates)
}

function Select-YouTubeCandidate($match, $candidates) {
    $titleTokens = Get-Tokens $match.title
    $artistTokens = Get-Tokens $match.artist
    $requiredFlags = Get-Flags $match.title
    $best = $null

    foreach ($candidate in $candidates) {
        $diff = [math]::Abs($candidate.duration - $match.duration)
        if ($diff -gt 2) {
            continue
        }

        $candidateTitle = [string]$candidate.title
        $candidateChannel = [string]$candidate.channel

        if (-not (Test-TokenCoverage $titleTokens $candidateTitle)) {
            continue
        }

        if (-not ((Test-TokenCoverage $artistTokens $candidateTitle) -or (Test-TokenCoverage $artistTokens $candidateChannel))) {
            continue
        }

        $candidateFlags = Get-Flags $candidateTitle
        $missingFlags = @($requiredFlags | Where-Object { $_ -notin $candidateFlags })
        if ($missingFlags.Count -gt 0) {
            continue
        }

        $unexpectedFlags = @($candidateFlags | Where-Object { $_ -notin $requiredFlags })
        if ($unexpectedFlags.Count -gt 0) {
            continue
        }

        $score = 1000 - ($diff * 100)
        if ($candidateChannel -match [regex]::Escape($match.artist)) {
            $score += 25
        }

        $selected = [pscustomobject]@{
            title       = $candidateTitle
            channel     = $candidateChannel
            duration    = $candidate.duration
            id          = $candidate.id
            diff        = $diff
            ranking     = $score
            youtube_url = 'https://www.youtube.com/watch?v=' + $candidate.id
        }

        if ($null -eq $best -or $selected.ranking -gt $best.ranking) {
            $best = $selected
        }
    }

    return $best
}

function Update-Track([int]$id, $match, $candidate) {
    $sql = @"
UPDATE tracks
SET title = '$(Escape-Sql $match.title)',
    artist = '$(Escape-Sql $match.artist)',
    source = 'youtube',
    youtube_url = '$(Escape-Sql $candidate.youtube_url)',
    youtube_video_id = '$(Escape-Sql $candidate.id)',
    youtube_channel = '$(Escape-Sql $candidate.channel)'
WHERE id = $id;
"@

    $sql | docker compose exec -T postgres psql -U postgres -d music_recommendations | Out-Null
}

function Update-TrackMetadataOnly([int]$id, $match) {
    $sql = @"
UPDATE tracks
SET title = '$(Escape-Sql $match.title)',
    artist = '$(Escape-Sql $match.artist)'
WHERE id = $id;
"@

    $sql | docker compose exec -T postgres psql -U postgres -d music_recommendations | Out-Null
}

function Add-ReportEntry($entry) {
    $script:report += $entry

    $parts = @(
        [string]$entry.action,
        ('id=' + [string]$entry.id),
        ('file=' + [string]$entry.file)
    )

    if (-not [string]::IsNullOrWhiteSpace([string]$entry.reason)) {
        $parts += 'reason=' + [string]$entry.reason
    }

    if (-not [string]::IsNullOrWhiteSpace([string]$entry.title)) {
        $parts += 'title=' + [string]$entry.title
    }

    if (-not [string]::IsNullOrWhiteSpace([string]$entry.artist)) {
        $parts += 'artist=' + [string]$entry.artist
    }

    if (-not [string]::IsNullOrWhiteSpace([string]$entry.youtube)) {
        $parts += 'youtube=' + [string]$entry.youtube
    }

    Write-Output ($parts -join ' | ')
}

$report = @()

try {
    Start-HelperContainer
    $tracks = Get-PlaceholderTracks $Limit

    foreach ($track in $tracks) {
        try {
            $basename = [System.IO.Path]::GetFileName($track.file_path)
            $match = Get-FingerprintMatch $basename $AcoustIdClient

            if ($null -eq $match -or $match.score -lt 0.97) {
                Add-ReportEntry ([pscustomobject]@{
                    id       = $track.id
                    file     = $basename
                    action   = 'skip'
                    reason   = 'no_high_confidence_fingerprint_match'
                    title    = ''
                    artist   = ''
                    youtube  = ''
                })
                continue
            }

            $candidates = Get-YouTubeCandidates ($match.artist + ' ' + $match.title)
            $selected = Select-YouTubeCandidate $match $candidates

            if ($null -eq $selected) {
                if ($AllowMetadataOnly) {
                    if (-not $DryRun) {
                        Update-TrackMetadataOnly ([int]$track.id) $match
                    }

                    Add-ReportEntry ([pscustomobject]@{
                        id       = $track.id
                        file     = $basename
                        action   = if ($DryRun) { 'dry_run_metadata_only' } else { 'metadata_only' }
                        reason   = 'no_safe_youtube_candidate'
                        title    = $match.title
                        artist   = $match.artist
                        youtube  = ''
                    })
                    continue
                }

                Add-ReportEntry ([pscustomobject]@{
                    id       = $track.id
                    file     = $basename
                    action   = 'skip'
                    reason   = 'no_safe_youtube_candidate'
                    title    = $match.title
                    artist   = $match.artist
                    youtube  = ''
                })
                continue
            }

            if (-not $DryRun) {
                Update-Track ([int]$track.id) $match $selected
            }

            Add-ReportEntry ([pscustomobject]@{
                id       = $track.id
                file     = $basename
                action   = if ($DryRun) { 'dry_run' } else { 'updated' }
                reason   = ''
                title    = $match.title
                artist   = $match.artist
                youtube  = $selected.youtube_url
            })
        }
        catch {
            Add-ReportEntry ([pscustomobject]@{
                id       = $track.id
                file     = [System.IO.Path]::GetFileName($track.file_path)
                action   = 'error'
                reason   = $_.Exception.Message
                title    = ''
                artist   = ''
                youtube  = ''
            })
        }
    }
}
finally {
    Stop-HelperContainer
}

$report | Format-Table -Wrap -AutoSize
