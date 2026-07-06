# Mac Separation — Quick Start

## TL;DR

```powershell
# Один файл → vocal + minus
powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
    -InputAudioPath "D:\path\to\track.mp3"

# Batch (все MP3 в папке)
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "D:\path\to\folder" `
    -DryRun  # убрать для реального запуска
```

**Output:** `D:\AIMusicProject\artifacts\mac_quick\<alias>\vocals.flac` + `instrumental.flac`

---

## Доступные скрипты

| Скрипт | Назначение | Use case |
|--------|------------|----------|
| **`run-mac-quick-separation.ps1`** | Один audio файл → vocal + minus | Быстрый preview/diagnostic |
| **`run-mac-separation-with-rank.ps1`** | Separation + опциональный rank | Experiments с reference stems |
| **`run-mac-batch-separation.ps1`** | Batch processing множества файлов | Album/folder batch |

---

## Примеры использования

### 1. Базовая separation одного файла

```powershell
cd D:\AIMusicProject

powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
    -InputAudioPath "testing_stuff_music\my_track.mp3"
```

**Output:**
```
D:\AIMusicProject\artifacts\mac_quick\t_20260705_143022\
├── vocals.flac
├── instrumental.flac
├── stem_manifest.json
└── alias_map.json
```

**Время выполнения:** ~30-60 секунд для 3-4 минутного трека (Mac MPS).

---

### 2. Batch processing папки

```powershell
# Dry run (preview)
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "testing_stuff_music" `
    -DryRun

# Реальный запуск
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "testing_stuff_music"
```

**Output:**
```
D:\AIMusicProject\artifacts\mac_batch\batch_20260705_143022\
└── batch_manifest.json

D:\AIMusicProject\artifacts\mac_quick\
├── t_track1\
├── t_track2\
└── t_track3\
```

---

### 3. Separation с custom alias

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
    -InputAudioPath "D:\Music\my_song.wav" `
    -OutputAlias "my_song_v1" `
    -OpenOutputFolder
```

---

### 4. Separation с reference stems (для rank)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-separation-with-rank.ps1 `
    -InputAudioPath "testing_stuff_music\mix.mp3" `
    -ReferenceVocalPath "testing_stuff_music\stems\vocal.flac" `
    -ReferenceMinusPath "testing_stuff_music\stems\minus.flac" `
    -OutputAlias "test_rank_001"
```

**Output:**
```
D:\AIMusicProject\artifacts\mac_rank\test_rank_001\
├── sep\               # Separation outputs
│   ├── vocals.flac
│   └── instrumental.flac
├── rank\              # Rank metrics (placeholder)
│   └── rank_manifest.json
└── summary.json
```

---

## Первый запуск — setup

### Проверка SSH доступа

```powershell
ssh -i %USERPROFILE%\.ssh\id_ed25519_phone -o IdentitiesOnly=yes andrejviluha@192.168.0.138 "echo mac_ready"
```

Если ошибка → запустите bootstrap:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-bootstrap-and-ssh.ps1 `
    -MacHost 192.168.0.138 `
    -MacUser andrejviluha
```

### Проверка Mac MPS venv

```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1 `
    -RemoteCommand "source /Users/andrejviluha/aimusic_mps_venv/bin/activate && python3 --version && python3 -c 'import torch; print(torch.backends.mps.is_available())'"
```

Expected output: `True` (MPS available)

### Проверка Windows mount

```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1 `
    -RemoteCommand "test -w /Users/andrejviluha/AIMusicProject_win/artifacts && echo writable || echo not_writable"
```

Expected output: `writable`

---

## Ключевые параметры

### Все скрипты (общие)

| Параметр | Default | Описание |
|----------|---------|-----------|
| `-MacHost` | `192.168.0.138` | IP Mac машины |
| `-MacUser` | `andrejviluha` | SSH user |
| `-SshKey` | `%USERPROFILE%\.ssh\id_ed25519_phone` | SSH private key |
| `-WindowsProjectRoot` | `D:\AIMusicProject` | Windows project root |
| `-MacMountPath` | `/Users/andrejviluha/AIMusicProject_win` | Mac mount path |

### `run-mac-quick-separation.ps1`

| Параметр | Required | Default |
|----------|----------|---------|
| `-InputAudioPath` | ✓ | — |
| `-OutputAlias` | ✗ | `t_<timestamp>` |
| `-SkipBootstrap` | ✗ | `$false` |
| `-OpenOutputFolder` | ✗ | `$false` |

### `run-mac-batch-separation.ps1`

| Параметр | Required | Default |
|----------|----------|---------|
| `-InputFolder` | ✓ | — |
| `-FilePattern` | ✗ | `*.mp3` |
| `-MaxFiles` | ✗ | `0` (без лимита) |
| `-DryRun` | ✗ | `$false` |
| `-ContinueOnError` | ✗ | `$false` |

---

## Важные контракты

### ⚠️ Diagnostic-only status

Все Mac outputs являются **diagnostic-only**:
- `delivery_allowed = false`
- Не approved для production delivery
- Использовать только для preview/experiments

Для **production accepted delivery** → Docker CUDA routes:
```powershell
docker compose run --rm ml-worker python /app/tools/export_instrumental_resurrection_minus.py ...
docker compose run --rm ml-worker python /app/tools/adaptive_vocal_isolation.py ...
```

### Windows-safe naming

- Output alias ≤32 символа
- Stems: `vocals.flac`, `instrumental.flac`
- Manifests: короткие имена (`stem_manifest.json`, `alias_map.json`)

### Model route

- **MelBand RoFormer** (`melband_roformer_vocals`)
- Один model route для vocal + instrumental pair
- Instrumental output = alias для **minus product role**
- **Не** production resurrection minus route

---

## Troubleshooting

### "Input audio file not found"

Проверьте путь:
```powershell
Test-Path "D:\path\to\file.mp3"
```

### "stem_manifest.json not found"

Separation failed на Mac. Проверьте:
1. Mac SSH доступен
2. Windows mount доступен на Mac
3. MPS venv активирован

Debug:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1 `
    -RemoteCommand "ls -lh /Users/andrejviluha/AIMusicProject_win/testing_stuff_music/"
```

### "Output alias too long"

Alias должен быть ≤32 символа:
```powershell
# ✗ BAD
-OutputAlias "my_very_long_track_name_2026"

# ✓ GOOD
-OutputAlias "track_v1"
```

### SSH connection failed

Проверьте SSH + bootstrap:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-bootstrap-and-ssh.ps1
```

---

## Next steps

1. **Quick test:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
       -InputAudioPath "testing_stuff_music\track.mp3"
   ```

2. **Listen to outputs:**
   - `artifacts\mac_quick\t_<timestamp>\vocals.flac`
   - `artifacts\mac_quick\t_<timestamp>\instrumental.flac`

3. **Batch processing:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
       -InputFolder "testing_stuff_music" `
       -MaxFiles 3
   ```

4. **Production delivery:**
   - После diagnostic OK → переключиться на Docker CUDA routes
   - См. `docs/MAC_QUICK_SEPARATION_GUIDE.md` для production examples

---

## Full documentation

- **`docs/MAC_QUICK_SEPARATION_GUIDE.md`** — Полная документация всех скриптов
- **`docs/MACOS_8GB_AUDIO_RUNTIME.md`** — Mac MPS API reference
- **`docs/OPERATIONS.md`** — Production audio loop operations
