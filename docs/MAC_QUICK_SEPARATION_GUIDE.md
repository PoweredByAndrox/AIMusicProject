# Mac Quick Separation Guide

## Обзор

Два новых PowerShell скрипта для быстрого запуска vocal + minus separation через Mac SSH:

1. **`run-mac-quick-separation.ps1`** — простой скрипт для базового separation
2. **`run-mac-separation-with-rank.ps1`** — расширенный скрипт с опциональным rank scoring

Оба скрипта:
- Используют Mac MPS (Apple Metal) для GPU acceleration
- Автоматически bootstrap SSH + Windows workspace mount
- Генерируют короткие Windows-safe имена файлов
- Создают structured manifests с provenance
- Соблюдают diagnostic-only contract (не accepted delivery)

---

## Скрипт 1: `run-mac-quick-separation.ps1`

### Быстрый старт

**Минимальная команда:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 -InputAudioPath "D:\AIMusicProject\testing_stuff_music\track.mp3"
```

**С custom alias:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
    -InputAudioPath "D:\AIMusicProject\testing_stuff_music\my_song.mp3" `
    -OutputAlias "my_song_v1"
```

**Без bootstrap (если уже настроено):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
    -InputAudioPath "D:\path\to\audio.mp3" `
    -SkipBootstrap `
    -OpenOutputFolder
```

### Параметры

| Параметр | Обязательный | Значение по умолчанию | Описание |
|----------|--------------|------------------------|-----------|
| `-InputAudioPath` | ✓ | — | Полный путь к input audio файлу (MP3/WAV/FLAC) |
| `-OutputAlias` | ✗ | `t_<timestamp>` | Короткий alias для output (≤32 chars) |
| `-MacHost` | ✗ | `192.168.0.138` | IP Mac машины |
| `-MacUser` | ✗ | `andrejviluha` | Mac SSH user |
| `-SshKey` | ✗ | `%USERPROFILE%\.ssh\id_ed25519_phone` | SSH private key |
| `-WindowsProjectRoot` | ✗ | `D:\AIMusicProject` | Windows project root |
| `-MacMountPath` | ✗ | `/Users/andrejviluha/AIMusicProject_win` | Mac mount path |
| `-SkipBootstrap` | ✗ | `$false` | Пропустить SSH + mount setup |
| `-OpenOutputFolder` | ✗ | `$false` | Открыть output папку после completion |

### Output структура

```
D:\AIMusicProject\artifacts\mac_quick\<alias>\
├── vocals.flac                 # Vocal stem
├── instrumental.flac           # Minus stem (minus product role)
├── raw_ref.flac               # Raw reference (unused в простом route)
├── stem_manifest.json         # Full separation manifest
└── alias_map.json             # Short alias → provenance mapping
```

### Примеры

**1. Базовая separation:**
```powershell
cd D:\AIMusicProject
powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
    -InputAudioPath "testing_stuff_music\track_001.mp3"
```

Output: `artifacts\mac_quick\t_20260705_143022\`

**2. Batch processing (loop):**
```powershell
$tracks = Get-ChildItem "D:\AIMusicProject\testing_stuff_music\*.mp3"
foreach ($track in $tracks) {
    $alias = "t_" + $track.BaseName.Replace(" ", "_").Substring(0, [Math]::Min(28, $track.BaseName.Length))
    powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
        -InputAudioPath $track.FullName `
        -OutputAlias $alias `
        -SkipBootstrap
}
```

**3. С автоматическим открытием результата:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
    -InputAudioPath "D:\Music\my_track.wav" `
    -OutputAlias "my_track_test" `
    -OpenOutputFolder
```

---

## Скрипт 2: `run-mac-separation-with-rank.ps1`

### Быстрый старт

**Только separation (без rank):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-separation-with-rank.ps1 `
    -InputAudioPath "D:\AIMusicProject\testing_stuff_music\track.mp3" `
    -SkipRank
```

**Separation + rank с reference stems:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-separation-with-rank.ps1 `
    -InputAudioPath "D:\AIMusicProject\testing_stuff_music\mix.mp3" `
    -ReferenceVocalPath "D:\AIMusicProject\testing_stuff_music\reference_vocal.flac" `
    -ReferenceMinusPath "D:\AIMusicProject\testing_stuff_music\reference_minus.flac" `
    -OutputAlias "track_with_rank"
```

### Дополнительные параметры

| Параметр | Описание |
|----------|-----------|
| `-ReferenceVocalPath` | Optional reference vocal для rank comparison |
| `-ReferenceMinusPath` | Optional reference minus для rank comparison |
| `-SkipRank` | Пропустить rank phase (только separation) |
| `-UseDockerRank` | Использовать Docker CUDA для rank (не реализовано) |

### Output структура

```
D:\AIMusicProject\artifacts\mac_rank\<alias>\
├── sep\                        # Separation phase
│   ├── vocals.flac
│   ├── instrumental.flac
│   ├── raw_ref.flac
│   └── stem_manifest.json
├── rank\                       # Rank phase (если не -SkipRank)
│   └── rank_manifest.json      # Rank metrics (placeholder в текущей версии)
└── summary.json               # Top-level summary
```

### Примеры

**1. Separation без rank:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-separation-with-rank.ps1 `
    -InputAudioPath "testing_stuff_music\track.mp3" `
    -SkipRank
```

**2. Separation + rank с vocal reference:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-separation-with-rank.ps1 `
    -InputAudioPath "testing_stuff_music\mix.mp3" `
    -ReferenceVocalPath "testing_stuff_music\golden\vocal.flac" `
    -OutputAlias "test_vocal_rank"
```

**3. Full pipeline с обоими references:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-separation-with-rank.ps1 `
    -InputAudioPath "testing_stuff_music\full_mix.wav" `
    -ReferenceVocalPath "testing_stuff_music\stems\vocal.flac" `
    -ReferenceMinusPath "testing_stuff_music\stems\minus.flac" `
    -OutputAlias "full_test_001" `
    -OpenOutputFolder
```

---

## Контракты и ограничения

### Diagnostic-only status

⚠️ **Все outputs являются diagnostic-only и не approved для delivery:**

- `delivery_allowed = false`
- `delivery_status = "diagnostic_mac_quick_separation_only"` или `"diagnostic_mac_separation_rank_only"`
- Mac routes не проходят полный production contract (CUDA Docker, full gates, provenance-safe reuse)

Для **production accepted delivery** используйте Docker CUDA routes:
- `export_instrumental_resurrection_minus.py` (minus)
- `adaptive_vocal_isolation.py` (vocal)

### Windows-safe naming

Все output directories и files используют короткие Windows-safe names:
- Output alias ≤32 символа
- Stems: `vocals.flac`, `instrumental.flac` (не raw track titles)
- Manifests: `stem_manifest.json`, `alias_map.json`, `summary.json`

### Model route

**MelBand RoFormer** (`melband_roformer_vocals`):
- Один model route для vocal + instrumental pair
- **Не** то же самое что production resurrection minus
- Instrumental output является **alias для minus product role**, но не production minus route

### Mac MPS acceleration

- Native PyTorch MPS (Apple Metal)
- venv: `/Users/andrejviluha/aimusic_mps_venv`
- device: `mps`
- num_overlap: `4` (default в скрипте)

### SSH + mount contract

Скрипты автоматически вызывают:
1. `macos-bootstrap-and-ssh.ps1` (mount + verify)
2. `macos-ssh.ps1` (remote command execution)

Required mount:
- Windows share: `\\<windows-ip>\AIMusicProject` → `D:\AIMusicProject`
- Mac mount: `/Users/andrejviluha/AIMusicProject_win`
- SMB user: `MacAudioApi`

---

## Troubleshooting

### Ошибка: "Input audio file not found"

Проверьте, что путь существует и правильно экранирован:
```powershell
Test-Path "D:\AIMusicProject\testing_stuff_music\track.mp3"
```

### Ошибка: "stem_manifest.json not found"

Separation failed на Mac. Проверьте:
1. Mac MPS venv активирован: `/Users/andrejviluha/aimusic_mps_venv`
2. Input audio path доступен через mount
3. Mac имеет write access к output dir

Debug command:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1 `
    -RemoteCommand "source /Users/andrejviluha/aimusic_mps_venv/bin/activate && python3 --version && ls -lh /Users/andrejviluha/AIMusicProject_win/testing_stuff_music/"
```

### Ошибка: SSH connection failed

Проверьте SSH key и target:
```powershell
ssh -i %USERPROFILE%\.ssh\id_ed25519_phone -o IdentitiesOnly=yes andrejviluha@192.168.0.138 "echo mac_ready"
```

Если не работает, запустите full bootstrap:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-bootstrap-and-ssh.ps1 `
    -MacHost 192.168.0.138 `
    -MacUser andrejviluha
```

### Output alias too long error

Alias должен быть ≤32 символа (Windows-safe contract):
```powershell
# ✗ BAD (слишком длинный)
-OutputAlias "my_very_long_track_name_with_timestamp_20260705"

# ✓ GOOD
-OutputAlias "track_v1"
-OutputAlias "t_20260705_143022"
```

### Mac mount not writable

Проверьте mount status:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1 `
    -RemoteCommand "mount | grep AIMusicProject && test -w /Users/andrejviluha/AIMusicProject_win/artifacts && echo writable || echo not_writable"
```

Если mount отсутствует, запустите:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-bootstrap-workspace-mount.ps1
```

---

## Production route comparison

| | Mac Quick Separation | Production Docker |
|---|---------------------|-------------------|
| **Separation model** | MelBand RoFormer | Resurrection IR (minus) + Adaptive vocal candidates |
| **Acceleration** | Mac MPS (diagnostic) | CUDA GPU |
| **Minus route** | MelBand instrumental output (alias) | Dedicated `export_instrumental_resurrection_minus.py` |
| **Vocal route** | MelBand vocals output | `adaptive_vocal_isolation.py` multi-candidate + gates |
| **Delivery status** | `delivery_allowed=false` | После explicit gates → `delivery_allowed=true` |
| **Use case** | Quick diagnostic/preview | Production accepted delivery |

---

## Следующие шаги

1. **Для quick preview/diagnostic:** используйте `run-mac-quick-separation.ps1`
2. **Для rank experiments:** используйте `run-mac-separation-with-rank.ps1` с reference stems
3. **Для production delivery:** переключайтесь на Docker CUDA routes после diagnostic phase

**Production delivery example:**
```powershell
# Step 1: Quick Mac diagnostic
powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
    -InputAudioPath "testing_stuff_music\track.mp3" `
    -OutputAlias "quick_preview"

# Step 2: Listen to diagnostic stems
# (manual step)

# Step 3: Production Docker route (если diagnostic OK)
docker compose run --rm --no-deps ml-worker `
    python /app/tools/export_instrumental_resurrection_minus.py `
    /host_project/testing_stuff_music/track.mp3 `
    /app/lab_artifacts/production/track_minus

docker compose run --rm --no-deps ml-worker `
    python /app/tools/adaptive_vocal_isolation.py `
    /host_project/testing_stuff_music/track.mp3 `
    /app/lab_artifacts/production/track_vocal `
    --ir-instrumental-minus-path /app/lab_artifacts/production/track_minus/instrumental_resurrection_minus.flac
```

---

---

## Скрипт 3: `run-mac-batch-separation.ps1` (Batch processing)

### Обзор

Обработка нескольких audio файлов одной командой.

### Быстрый старт

**Dry run (preview без execution):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "D:\AIMusicProject\testing_stuff_music" `
    -DryRun
```

**Реальная batch обработка:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "D:\AIMusicProject\testing_stuff_music" `
    -FilePattern "*.mp3"
```

**С лимитом файлов:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "D:\Music\MyAlbum" `
    -FilePattern "*.wav" `
    -MaxFiles 5 `
    -ContinueOnError
```

### Параметры

| Параметр | Обязательный | Описание |
|----------|--------------|-----------|
| `-InputFolder` | ✓ | Папка с audio файлами |
| `-FilePattern` | ✗ | Glob pattern (default: `*.mp3`) |
| `-OutputRootAlias` | ✗ | Batch alias (default: `batch_<timestamp>`) |
| `-MaxFiles` | ✗ | Максимальное количество файлов (0 = без лимита) |
| `-DryRun` | ✗ | Preview режим без реального execution |
| `-ContinueOnError` | ✗ | Продолжить batch при ошибке на одном файле |

### Output структура

```
D:\AIMusicProject\artifacts\mac_batch\<batch_alias>\
└── batch_manifest.json         # Batch summary + results

D:\AIMusicProject\artifacts\mac_quick\
├── t_track1\                   # Individual track outputs
│   ├── vocals.flac
│   ├── instrumental.flac
│   └── stem_manifest.json
├── t_track2\
│   └── ...
└── t_track3\
    └── ...
```

### Примеры

**1. Обработать все MP3 в папке:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "D:\AIMusicProject\testing_stuff_music"
```

**2. Обработать только WAV файлы:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "D:\MyMusic" `
    -FilePattern "*.wav"
```

**3. Test batch на 3 файлах:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "testing_stuff_music" `
    -MaxFiles 3 `
    -OutputRootAlias "test_batch_001"
```

**4. Production batch с error handling:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "D:\Albums\MyAlbum" `
    -FilePattern "*.flac" `
    -ContinueOnError `
    -OutputRootAlias "album_batch_v1"
```

### Batch manifest

После completion создаётся `batch_manifest.json`:

```json
{
  "schema": "mac_batch_separation.v1",
  "batch_alias": "batch_20260705_143022",
  "input_folder": "D:\\AIMusicProject\\testing_stuff_music",
  "file_pattern": "*.mp3",
  "total_files": 10,
  "success_count": 9,
  "fail_count": 1,
  "skipped_count": 0,
  "timestamp": "2026-07-05T14:30:45.1234567+00:00",
  "results": [
    {
      "index": 1,
      "file_name": "track1.mp3",
      "track_alias": "t_track1",
      "status": "success",
      "output_dir": "D:\\AIMusicProject\\artifacts\\mac_quick\\t_track1",
      "elapsed_sec": 45.3
    },
    ...
  ]
}
```

---

## Related docs

- `docs/MACOS_8GB_AUDIO_RUNTIME.md` — Mac MPS API полная документация
- `docs/OPERATIONS.md` — Production audio loop operations
- `ml_worker_improved/docs/MACOS_REMOTE_API_COMMAND_README.md` — Mac remote command contract
- `scripts/macos-ssh.ps1` — SSH wrapper script
- `scripts/macos-bootstrap-and-ssh.ps1` — Bootstrap + mount setup
