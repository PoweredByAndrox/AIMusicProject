# Как я собирал эту вторую версию проекта

## Шаг 1. Сначала зафиксировал реальную проблему из твоего теста

Ты проверил первую версию правильно и нашёл не абстрактный “мне что-то не нравится”, а конкретные системные дыры:

- separation не шёл реально из-за `DRY_RUN=true`;
- UI мог показывать stale состояние;
- `bsroformer-worker` в живом контейнере использовал всю папку inputs;
- query/restoration были stub'ами;
- orchestration был полезен, но не production-like.

Поэтому во второй версии я не стал “косметически улучшать README”, а менял именно эти точки.

## Шаг 2. Исправил путь данных: input -> artifacts -> UI

Я построил более честную цепочку:

1. upload audio / text;
2. сохранить файл в `data/inputs`;
3. создать job;
4. worker публикует **stable named outputs**;
5. gateway сохраняет их в `job.artifacts`;
6. API отдаёт список артефактов;
7. UI показывает audio + lyrics через `/media/...`.

Это важно, потому что без стабильного слоя артефактов невозможно нормально соединить:
- stem'ы,
- lyrics,
- прослушивание,
- дальнейшие routing steps.

## Шаг 3. Исправил BS-RoFormer на isolated single-file input

Твой вывод был очень точным: `--input_folder /app/data/inputs` опасен, потому что в папке может лежать несколько файлов и мусор.

Поэтому я изменил worker так, чтобы он:
- создавал `_isolated_input/` внутри output шага;
- копировал туда **только один целевой файл**;
- уже эту папку отдавал `bs-roformer-infer`.

Это не косметика, а важный фикс для реального инференса.

## Шаг 4. Добавил lyrics-worker как отдельный контейнер

Я не стал засовывать lyrics в gateway или во фронтенд “как ещё один JSON helper”.

Вместо этого сделал отдельный контейнер, потому что lyrics pipeline — это отдельный ML-слой со своими зависимостями, VRAM и режимами:
- автоматическая транскрипция;
- word-level timestamps;
- alignment по reference text.

Так проще:
- дебажить;
- отключать;
- обновлять модель;
- масштабировать потом отдельно от separation.

## Шаг 5. Добавил два честных режима lyrics

### Режим 1. `transcribe_wordlevel`
Берёт вокал или raw track и делает автоматический текст + таймкоды.

### Режим 2. `align_reference_lyrics`
Берёт готовый reference text и аудио, а потом уже привязывает **этот конкретный текст** к времени.

Это сделано потому, что полностью автоматическое “1-в-1 для любой песни” обещать нечестно.

## Шаг 6. Протянул lyrics до фронтенда

Я добавил во фронтенд:
- upload audio;
- upload text;
- polling jobs;
- artifact browser;
- audio player;
- выбор текущего audio артефакта;
- выбор текущего lyrics артефакта;
- отображение сегментов/слов с подсветкой по текущему времени.

То есть это уже не просто “контейнеры где-то что-то сделали”, а реально usable test UI.

## Шаг 7. Оставил честную границу между real и stub

Я специально **не притворялся**, что query-based extraction и restoration magically уже готовы.

Во второй версии:
- `lyrics-worker` — реальный путь;
- `demucs-worker` и `bsroformer-worker` — каркас под реальный subprocess;
- `querysep-worker` и `restoration-worker` — честные stub/research зоны.

Это лучше, чем выдать фальшивую “полностью рабочую систему”, которую потом невозможно починить.

## Шаг 8. Что бы я делал дальше руками

1. добавить real output mapping для Demucs и BS-RoFormer;
2. подключить один настоящий query-based extractor;
3. сделать section router по плотности/типу микса;
4. добавить waveform preview;
5. сделать stem acceptance scoring;
6. вынести job queue и GPU scheduling отдельно.

## Шаг 9. P0–P3 reliability sweep (R1–R14)

После того как stem'ы реально пошли через pipeline, всплыл следующий
слой проблем: накопление ошибок в `other`, musical noise, hallucinations
у backbone'ов на single-pass, headroom round-trip между шагами, и
вообще **отсутствие способа проверить «лучше я сделал или хуже»**
после очередного рефакторинга. Это и был повод запустить
`stem-tandem-improvement-roadmap_86d167c0.plan.md`. По итогам
sweep'а:

- **P0** (R1, R2, R3, R4, R5, R7b, R-PCM) — заткнуты «дешёвые» дыры:
  honest 44100/`pcm_f32le`/`num_overlap=4`, `htdemucs_ft`,
  WhisperX `large-v3`, успокоенный `*_unmuffle`, исправленный
  Mel-Band instrumental subtraction.
- **P1** (R6, R-ACT, R7, R8, R9) — TTA, адаптивные activity-маски,
  прямой ML-`other` (`mode: direct`) вместо вычитания, MWF
  post-filter и FFT-ensemble нескольких кандидатов на стэм.
- **P2** (R10, R11, R12) — SCNet medium как опциональный третий
  backbone, Banquet полностью wired up в `querysep-worker` для
  long-tail, и честное переименование `audio-separator-worker →
  bus-mixer-worker` через `honest_name`, чтобы doc и UI больше не
  врали про PyPI-пакет.
- **P3** (R13, R14, R15) — `baseline_snapshot/compare` (отлов
  unintended drift), `sisdr_compare` (настоящий SI-SDR против
  ground-truth) и сама синхронизация документации с кодом.

Главный вывод этого шага — **сначала надо было построить
регрессионную сетку**, а уже потом крутить веса/параметры. Иначе
любая «оптимизация» становится верой, а не инженерией. R13+R14
закрывают этот пробел.
