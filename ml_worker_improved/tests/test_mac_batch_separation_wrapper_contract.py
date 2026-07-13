from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_SCRIPT = REPO_ROOT / "scripts" / "run-mac-batch-separation.ps1"


def _script() -> str:
    return BATCH_SCRIPT.read_text(encoding="utf-8-sig")


def test_batch_wrapper_supports_the_mt017_c3_mac_api_profile() -> None:
    script = _script()

    assert '[ValidateSet("quick", "mt017-c3")]' in script
    assert 'http://127.0.0.1:8765/health' in script
    assert 'Join-Path $WindowsProjectRoot "run_gradusi_test.ps1"' in script
    assert '-InputAudioPath $file.FullName' in script
    assert '-OutputAlias $trackAlias' in script


def test_batch_wrapper_uses_short_numeric_aliases_and_keeps_provenance_in_manifest() -> None:
    script = _script()

    assert 'return "{0}t{1:D3}" -f $BatchAlias, $Index' in script
    assert '$file.BaseName.Replace' not in script
    assert 'file_name = $file.Name' in script
    assert 'file_path = $file.FullName' in script
    assert 'schema = "mac_batch_separation.v2"' in script
    assert 'delivery_allowed = $false' in script


def test_batch_wrapper_is_resumable_and_writes_manifest_atomically() -> None:
    script = _script()

    assert '[switch]$Resume' in script
    assert 'Test-CompletedTrackOutput' in script
    assert 'skipped_completed' in script
    assert '$tempPath = "$Path.tmp"' in script
    assert 'Move-Item -LiteralPath $tempPath -Destination $Path -Force' in script
