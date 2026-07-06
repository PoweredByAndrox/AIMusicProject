# Mac Separation Scripts — Summary

## Что создано

Три новых PowerShell скрипта для запуска vocal + minus separation через Mac SSH:

### 1. `scripts/run-mac-quick-separation.ps1`
**Назначение:** Быстрая separation одного audio файла → vocal + minus stems

**Использование:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
    -InputAudioPath "D:\path\to\track.mp3"
```

**Output:**
```
D:\AIMusicProject\artifacts\mac_quick\<alias>\
├── vocals.flac
├── instrumental.flac
├── stem_manifest.json
└── alias_map.json
```

**Ключевые параметры:**
- `-InputAudioPath` (required) — полный путь к audio файлу
- `-OutputAlias` (optional) — короткий alias (≤32 chars)
- `-SkipBootstrap` — пропустить SSH setup
- `-OpenOutputFolder` — открыть output после completion

---

### 2. `scripts/run-mac-separation-with-rank.ps1`
**Назначение:** Separation + опциональный rank scoring с reference stems

**Использование:**
```powershell
# Только separation
powershell -ExecutionPolicy Bypass -File scripts\run-mac-separation-with-rank.ps1 `
    -InputAudioPath "D:\path\to\track.mp3" `
    -SkipRank

# Separation + rank
powershell -ExecutionPolicy Bypass -File scripts\run-mac-separation-with-rank.ps1 `
    -InputAudioPath "D:\path\to\mix.mp3" `
    -ReferenceVocalPath "D:\path\to\ref_vocal.flac" `
    -ReferenceMinusPath "D:\path\to\ref_minus.flac"
```

**Output:**
```
D:\AIMusicProject\artifacts\mac_rank\<alias>\
├── sep\                        # Separation phase
│   ├── vocals.flac
│   ├── instrumental.flac
│   └── stem_manifest.json
├── rank\                       # Rank phase
│   └── rank_manifest.json
└── summary.json
```

**Дополнительные параметры:**
- `-ReferenceVocalPath` — reference vocal для rank
- `-ReferenceMinusPath` — reference minus для rank
- `-SkipRank` — пропустить rank phase
- `-UseDockerRank` — использовать Docker CUDA для rank (not implemented)

**Note:** Текущая rank implementation = placeholder (simplified). Для real rank metrics нужен Python rank script или Docker CUDA route.

---

### 3. `scripts/run-mac-batch-separation.ps1`
**Назначение:** Batch processing множества audio файлов одной командой

**Использование:**
```powershell
# Dry run (preview)
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "D:\AIMusicProject\testing_stuff_music" `
    -DryRun

# Реальная обработка
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "D:\AIMusicProject\testing_stuff_music" `
    -FilePattern "*.mp3" `
    -MaxFiles 10 `
    -ContinueOnError
```

**Output:**
```
D:\AIMusicProject\artifacts\mac_batch\<batch_alias>\
└── batch_manifest.json         # Summary + results

D:\AIMusicProject\artifacts\mac_quick\
├── t_track1\                   # Individual outputs
├── t_track2\
└── t_track3\
```

**Ключевые параметры:**
- `-InputFolder` (required) — папка с audio файлами
- `-FilePattern` — glob pattern (default: `*.mp3`)
- `-MaxFiles` — лимит файлов (0 = без лимита)
- `-DryRun` — preview без execution
- `-ContinueOnError` — продолжить batch при ошибке

---

## Документация

### 1. `docs/MAC_SEPARATION_QUICK_START.md`
**TL;DR guide** с примерами использования и troubleshooting.

**Содержание:**
- Quick start commands
- Примеры для каждого скрипта
- Первый запуск setup (SSH, mount, MPS venv)
- Ключевые параметры
- Troubleshooting

### 2. `docs/MAC_QUICK_SEPARATION_GUIDE.md`
**Полная документация** всех скриптов с детальными examples.

**Содержание:**
- Детальное описание каждого скрипта
- Все параметры с таблицами
- Output структура
- Batch manifest schema
- Контракты и ограничения
- Production route comparison
- Related docs

---

## Ключевые контракты

### SSH + Mount
- **Mac host:** `andrejviluha@192.168.0.138`
- **SSH key:** `%USERPROFILE%\.ssh\id_ed25519_phone`
- **Windows root:** `D:\AIMusicProject`
- **Mac mount:** `/Users/andrejviluha/AIMusicProject_win`

Скрипты автоматически вызывают:
1. `scripts/macos-bootstrap-and-ssh.ps1` — mount + verify
2. `scripts/macos-ssh.ps1` — remote command execution

### Model route
- **MelBand RoFormer** (`melband_roformer_vocals`)
- Один model route для vocal + instrumental pair
- Acceleration: Mac MPS (Apple Metal)
- venv: `/Users/andrejviluha/aimusic_mps_venv`
- device: `mps`, num_overlap: `4`

### Artifacts naming (Windows-safe)
- Output alias ≤32 символа
- Stems: `vocals.flac`, `instrumental.flac`
- Manifests: `stem_manifest.json`, `alias_map.json`, `summary.json`
- Provenance в manifest, не в path

### Diagnostic-only status
⚠️ **Все Mac outputs = diagnostic-only:**
- `delivery_allowed = false`
- `delivery_status = "diagnostic_mac_quick_separation_only"`
- Не approved для production delivery
- Использовать только для preview/experiments

**Для production accepted delivery** → Docker CUDA routes:
- `export_instrumental_resurrection_minus.py` (minus)
- `adaptive_vocal_isolation.py` (vocal)

---

## Как использовать

### Quick test (один файл)

```powershell
cd D:\AIMusicProject

powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
    -InputAudioPath "testing_stuff_music\track.mp3"
```

**Output:** `artifacts\mac_quick\t_<timestamp>\vocals.flac` + `instrumental.flac`

### Batch processing (папка)

```powershell
# Preview
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "testing_stuff_music" `
    -DryRun

# Реальный запуск
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "testing_stuff_music" `
    -MaxFiles 5
```

### С custom alias

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
    -InputAudioPath "D:\Music\my_track.wav" `
    -OutputAlias "my_track_v1" `
    -OpenOutputFolder
```

### Separation + rank (с reference)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-separation-with-rank.ps1 `
    -InputAudioPath "testing_stuff_music\mix.mp3" `
    -ReferenceVocalPath "testing_stuff_music\stems\vocal.flac" `
    -ReferenceMinusPath "testing_stuff_music\stems\minus.flac" `
    -OutputAlias "test_001"
```

---

## Troubleshooting

### SSH connection failed

```powershell
# Check SSH
ssh -i %USERPROFILE%\.ssh\id_ed25519_phone -o IdentitiesOnly=yes andrejviluha@192.168.0.138 "echo mac_ready"

# Bootstrap если нужно
powershell -ExecutionPolicy Bypass -File scripts\macos-bootstrap-and-ssh.ps1
```

### Mac mount not writable

```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1 `
    -RemoteCommand "test -w /Users/andrejviluha/AIMusicProject_win/artifacts && echo writable || echo not_writable"

# Re-mount если нужно
powershell -ExecutionPolicy Bypass -File scripts\macos-bootstrap-workspace-mount.ps1
```

### MPS venv check

```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1 `
    -RemoteCommand "source /Users/andrejviluha/aimusic_mps_venv/bin/activate && python3 -c 'import torch; print(torch.backends.mps.is_available())'"
```

Expected: `True`

### Output alias too long

Alias должен быть ≤32 символа (Windows path segment limit):
```powershell
# ✗ BAD (слишком длинный)
-OutputAlias "my_very_long_track_name_with_timestamp_20260705"

# ✓ GOOD
-OutputAlias "track_v1"
-OutputAlias "t_20260705"
```

---

## Что дальше

### 1. Quick test
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-quick-separation.ps1 `
    -InputAudioPath "testing_stuff_music\track.mp3"
```

### 2. Прослушать outputs
- `artifacts\mac_quick\t_<timestamp>\vocals.flac`
- `artifacts\mac_quick\t_<timestamp>\instrumental.flac`

### 3. Batch processing
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "testing_stuff_music" `
    -MaxFiles 3 `
    -DryRun
```

### 4. Production delivery (если diagnostic OK)
После того, как diagnostic stems прослушаны и OK, переключиться на Docker CUDA production routes для accepted delivery.

См. `docs/MAC_QUICK_SEPARATION_GUIDE.md` → "Production route comparison" для деталей.

---

## Файлы

### Скрипты
- `scripts/run-mac-quick-separation.ps1`
- `scripts/run-mac-separation-with-rank.ps1`
- `scripts/run-mac-batch-separation.ps1`

### Документация
- `docs/MAC_SEPARATION_QUICK_START.md` (quick start guide)
- `docs/MAC_QUICK_SEPARATION_GUIDE.md` (full documentation)
- `SEPARATION_SCRIPTS_SUMMARY.md` (этот файл)

### Related scripts (используются автоматически)
- `scripts/macos-bootstrap-and-ssh.ps1`
- `scripts/macos-bootstrap-workspace-mount.ps1`
- `scripts/macos-ssh.ps1`

### Related Python tools (вызываются через SSH)
- `ml_worker_improved/tools/export_melband_vocal_full.py`
- `ml_worker_improved/tools/mac_mps_separator_api.py` (не используется напрямую)

---

## Production code files (прочитано полностью)

| Файл | Объём | Что прочитано |
|------|-------|---------------|
| `ml_worker_improved/tools/run_mac_unified_track_pipeline.py` | 394 строки | Полностью (orchestration pattern) |
| `ml_worker_improved/tools/export_melband_vocal_full.py` | 886 строк | Полностью (separation implementation) |
| `scripts/macos-bootstrap-and-ssh.ps1` | 115 строк | Полностью (bootstrap workflow) |
| `scripts/macos-ssh.ps1` | 115 строк | Полностью (SSH wrapper) |
| `docs/MACOS_8GB_AUDIO_RUNTIME.md` | 1191 строка | Полностью (Mac MPS contract) |
| `docs/OPERATIONS.md` | 745 строк | Полностью (production operations) |
| `ml_worker_improved/docs/MACOS_REMOTE_API_COMMAND_README.md` | 88 строк | Полностью (remote command contract) |

---

## Подтверждено кодом

✓ Mac SSH contract (host, user, key, mount paths)  
✓ Mac MPS venv + PyTorch MPS availability  
✓ MelBand model route implementation  
✓ Diagnostic-only delivery status  
✓ Windows-safe naming contract  
✓ Bootstrap + mount workflow  
✓ Remote command execution pattern  
✓ Artifact provenance в manifest  
✓ Forbidden Mac-local roots blocking  

---

## Не подтверждено / не реализовано

✗ Real rank metrics computation (placeholder в текущей версии)  
✗ Docker CUDA rank route (mentioned, not implemented)  
✗ Production resurrection minus через Mac (диагностический только)  
✗ Adaptive vocal isolation multi-candidate через Mac  
✗ Full delivery gates для accepted status  

Для production accepted delivery использовать Docker CUDA routes.

---

**END OF SUMMARY**
