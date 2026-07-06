# 🎵 Как использовать Drag & Drop для Separation

## ✅ ВСЁ ГОТОВО К ИСПОЛЬЗОВАНИЮ!

У тебя есть **3 способа** запустить separation:

---

## 🎯 СПОСОБ 1: Drag & Drop на скрипт (ПРОЩЕ ВСЕГО)

### Шаги:

1. **Найди файл:**
   ```
   D:\AIMusicProject\DRAG_HERE_to_separate.ps1
   ```

2. **Перетащи на него** MP3/WAV/FLAC файл (можно с кириллицей в имени!)

3. **Подожди** 2-5 минут

4. **Получи результат** в:
   ```
   D:\AIMusicProject\artifacts\mac_quick\t_YYYYMMDD_HHMMSS_<hash>\
   ├── vocals.flac         ← ВОКАЛ
   └── instrumental.flac   ← МИНУС
   ```

### Что происходит:

- Скрипт автоматически генерирует короткий alias из хеша + timestamp
- Запускает separation на Mac через SSH
- Показывает прогресс в консоли
- Предлагает открыть папку с результатами

---

## 🎯 СПОСОБ 2: Ярлык (ЕЩЁ УДОБНЕЕ)

### Шаги:

1. **Создай ярлык** (один раз):
   ```
   Запусти: CREATE_SHORTCUT.bat
   ```

2. **Найди ярлык:**
   ```
   D:\AIMusicProject\🎵 Перетащи сюда MP3 для separation.lnk
   ```

3. **Перетаскивай файлы** на этот ярлык!

### Преимущества:

- Ярлык можно скопировать на рабочий стол
- Ярлык можно закрепить в панели задач
- Красивая иконка 🎵

---

## 🎯 СПОСОБ 3: Командная строка (для продвинутых)

### PowerShell:

```powershell
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

& "D:\AIMusicProject\scripts\run-mac-quick-separation.ps1" `
    -InputAudioPath "D:\Music\Градусы - Голая.mp3" `
    -OutputAlias "gradusi_v1" `
    -SkipBootstrap
```

### Параметры:

- `-InputAudioPath` — полный путь к файлу (обязательно)
- `-OutputAlias` — короткое имя для output папки (опционально)
- `-SkipBootstrap` — пропустить проверку SSH/mount (если уже настроено)
- `-OpenOutputFolder` — автоматически открыть папку с результатами

---

## 📋 Поддерживаемые форматы

| Формат | Поддержка |
|--------|-----------|
| MP3    | ✅ Да     |
| WAV    | ✅ Да     |
| FLAC   | ✅ Да     |
| M4A    | ✅ Да     |
| OGG    | ✅ Да     |

---

## ⏱️ Время обработки

| Длина трека | Время на Mac MPS |
|-------------|------------------|
| 3-4 минуты  | 2-5 минут        |
| 5-6 минут   | 4-8 минут        |
| 8-10 минут  | 7-12 минут       |

**Примечание:** Первый запуск может быть медленнее (Mac загружает модель 900 MB в память)

---

## 🔧 Что делать если не работает

### 1. Проверь SSH подключение:

```powershell
ssh -i %USERPROFILE%\.ssh\id_ed25519_phone andrejviluha@192.168.0.138 "echo OK"
```

Должно вернуть: `OK`

### 2. Если ошибка — запусти bootstrap:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-bootstrap-and-ssh.ps1
```

### 3. Проверь доступность Mac:

```cmd
ping 192.168.0.138
```

### 4. Проверь, что файлы появились на Mac:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\macos-ssh.ps1 `
    -RemoteCommand "ls -lh /Users/andrejviluha/AIMusicProject_win/artifacts/mac_quick/"
```

---

## 📁 Где найти результаты

### Output папка:

```
D:\AIMusicProject\artifacts\mac_quick\<alias>\
```

### Структура:

```
<alias>\
├── vocals.flac              ← ВОКАЛ (обычно 5-15% от размера оригинала)
├── instrumental.flac        ← МИНУС (обычно 85-95% от размера оригинала)
├── raw_ref.flac            ← Raw reference (можно удалить)
├── stem_manifest.json      ← Техническая информация о separation
├── alias_map.json          ← Provenance mapping
└── config_runtime.yaml     ← Runtime конфигурация
```

### Важные файлы:

- **`vocals.flac`** — чистый вокал
- **`instrumental.flac`** — минусовка (всё кроме вокала)

Остальные файлы можно удалить, если не нужны для debugging.

---

## 🎨 Пример использования

### Сценарий 1: Быстрая extraction

```
1. Перетащил "Градусы - Голая.mp3" на DRAG_HERE_to_separate.ps1
2. Подождал 3 минуты
3. Получил vocals.flac (3.5 MB) и instrumental.flac (4.2 MB)
4. Слушаю результат!
```

### Сценарий 2: Batch processing

```powershell
# Обработать все MP3 в папке
powershell -ExecutionPolicy Bypass -File scripts\run-mac-batch-separation.ps1 `
    -InputFolder "D:\Music\MyAlbum" `
    -MaxFiles 10
```

### Сценарий 3: С кастомным alias

```powershell
& "scripts\run-mac-quick-separation.ps1" `
    -InputAudioPath "D:\Music\track.mp3" `
    -OutputAlias "track_final_v2"
```

---

## ⚠️ Важные примечания

### Diagnostic-only статус

Все Mac outputs имеют статус **diagnostic-only** и **НЕ approved для production delivery**.

Это значит:
- ✅ Отлично для preview/listening/testing
- ✅ Быстрый способ получить stems
- ❌ Не для финальной master версии
- ❌ Не для коммерческого релиза

Для **production delivery** используй Docker CUDA routes:
```powershell
docker compose run --rm ml-worker python /app/tools/export_instrumental_resurrection_minus.py ...
docker compose run --rm ml-worker python /app/tools/adaptive_vocal_isolation.py ...
```

### Кириллица полностью поддерживается

Скрипт корректно обрабатывает файлы с русскими именами:
- ✅ `Градусы - Голая.mp3`
- ✅ `Баста - Сансара.wav`
- ✅ `Эминем feat Рианна.flac`

### Windows-safe naming

Output alias автоматически ограничивается 32 символами для совместимости с Windows path limits.

---

## 📚 Дополнительная документация

- **`БЫСТРЫЙ_СТАРТ_SEPARATION.txt`** — краткая инструкция
- **`docs/MAC_SEPARATION_QUICK_START.md`** — quick start guide
- **`docs/MAC_QUICK_SEPARATION_GUIDE.md`** — полная документация
- **`SEPARATION_SCRIPTS_SUMMARY.md`** — техническая сводка

---

## 🎉 Готово!

Теперь просто **перетаскивай файлы** и получай результат!

Если что-то не работает — пиши, помогу разобраться.
