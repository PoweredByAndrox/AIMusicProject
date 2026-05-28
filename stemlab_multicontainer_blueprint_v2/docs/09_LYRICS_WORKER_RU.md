# Lyrics Worker

## Зачем он нужен

Lyrics — это отдельный слой задачи.

Если держать его как “побочный текстовый лог” где-то внутри separation worker'а, потом становится больно:
- обновлять модель;
- дебажить таймкоды;
- подсовывать reference text;
- показывать результат во фронтенде.

Поэтому во второй версии lyrics вынесен в **отдельный контейнер**.

## Какая модель используется

С R5 `lyrics-worker` по умолчанию запускает WhisperX с моделью
`large-v3` (`LYRICS_WHISPERX_MODEL=large-v3`). Это даёт заметное
улучшение распознавания пения и многоязычных строк по сравнению с
`large-v2`. При нехватке VRAM можно вернуть `large-v2` через
`.env`/`docker-compose.yml`.

## Какие режимы есть

### 1. `transcribe_wordlevel`
- вход: audio stem или raw track;
- выход:
  - `lyrics_json`
  - `lyrics_txt`
  - `lyrics_lrc`
  - `lyrics_srt`
- смысл: лучший автоматический проход с пословными/посегментными таймкодами.

### 2. `align_reference_lyrics`
- вход: audio + reference text;
- выход:
  - `lyrics_json`
  - `lyrics_txt`
  - `lyrics_lrc`
  - `lyrics_srt`
- смысл: не угадывать текст заново, а **привязать данный текст к времени**.

## Какой режим выбирать

### Хочешь просто быстро получить текст с таймкодами
Используй `transcribe_wordlevel`.

### Хочешь максимально близко к “1-в-1”
Используй `align_reference_lyrics` и загружай `lyrics_reference_path`.

## Что ложится в UI

Во фронтенд уже протянуты:
- выбор audio артефакта;
- выбор lyrics артефакта;
- player;
- текущая позиция;
- подсветка активного сегмента;
- подсветка активного слова, если есть word-level данные.

## Какие файлы сохраняются

Каждый run может публиковать:
- `lyrics_json`
- `lyrics_txt`
- `lyrics_lrc`
- `lyrics_srt`

Плюс алиасы вида:
- `lyrics_wordlevel_json`
- `lyrics_aligned_json`

Это сделано, чтобы:
- downstream шаги знали, на какой конкретный артефакт ссылаться;
- UI мог брать универсальный ключ `lyrics_json`.

## Где менять поведение

- orchestration: `apps/api/app/services/orchestrator.py`
- uploads: `apps/api/app/routes/uploads.py`
- artifacts: `apps/api/app/services/artifacts.py`
- worker: `services/lyrics-worker/app/main.py`
- frontend parser: `apps/frontend/src/lyrics.ts`
- frontend UI: `apps/frontend/src/App.tsx`

## Ограничения

- автоматическая транскрипция не гарантирует идеальный текст на пении;
- сильный реверб, хор, даблы, крик, смешанные языки и lo-fi source ухудшают точность;
- exact-text режим требует reference text и корректный язык.
