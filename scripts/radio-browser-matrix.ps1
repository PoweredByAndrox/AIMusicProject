param(
    [string] $Url = 'http://localhost:8000/radio/test',
    [int] $VirtualTimeBudgetMs = 12000
)

$browserCandidates = @(
    @{
        Name = 'Chrome'
        Path = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
    },
    @{
        Name = 'Edge'
        Path = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    }
)

function Get-BodyAttributeValue {
    param(
        [string] $Html,
        [string] $AttributeName
    )

    $match = [regex]::Match($Html, "$AttributeName=""([^""]*)""", [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $match.Success) {
        return $null
    }

    return $match.Groups[1].Value
}

$results = @()

foreach ($browser in $browserCandidates) {
    if (-not (Test-Path $browser.Path)) {
        continue
    }

    $html = (& $browser.Path `
        --headless=new `
        --disable-gpu `
        --autoplay-policy=no-user-gesture-required `
        --virtual-time-budget=$VirtualTimeBudgetMs `
        --dump-dom `
        $Url 2>$null | Out-String)

    $results += [pscustomobject]@{
        browser = $browser.Name
        playback_capability = Get-BodyAttributeValue -Html $html -AttributeName 'data-radio-playback-capability'
        playback_mode = Get-BodyAttributeValue -Html $html -AttributeName 'data-radio-playback-mode'
        waveform_transport = Get-BodyAttributeValue -Html $html -AttributeName 'data-radio-waveform-transport'
        waveform_track_id = Get-BodyAttributeValue -Html $html -AttributeName 'data-radio-waveform-track-id'
        bitrate = Get-BodyAttributeValue -Html $html -AttributeName 'data-radio-bitrate'
    }
}

if ($results.Count -eq 0) {
    throw 'No supported browser executable was found for the radio browser matrix check.'
}

$results | ConvertTo-Json -Depth 4
