# План реализации: Продвинутая система музыкального плеера

## Обзор

Реализация продвинутой системы музыкального плеера с 6 функциональными блоками. Задачи организованы по компонентам с инкрементальным подходом — каждая задача строится на предыдущих.

## Задачи

- [x] 1. Настройка базы данных и моделей
  - [x] 1.1 Создать миграции для новых таблиц (playlists, playlist_tracks, track_ratings, rating_history, user_track_preferences, listening_history, track_skips, radio_feedback, track_dj_analysis, user_settings)
    - Создать миграции в database/migrations/
    - _Requirements: 4.4, 5.1, 11.1, 14.1, 16.3, 18.2, 27.5_
  
  - [x] 1.2 Создать Eloquent модели для новых сущностей
    - Playlist, PlaylistTrack, TrackRating, RatingHistory, UserTrackPreference, ListeningHistory, TrackSkip, RadioFeedback, TrackDjAnalysis, UserSetting
    - Определить relationships и casts
    - _Requirements: 4.4, 5.1_
  
  - [x] 1.3 Обновить модель Fragment — добавить поле comment и изменить is_liked на nullable
    - _Requirements: 7.1, 8.1_

- [x] 2. Checkpoint — Проверить миграции и модели
  - Запустить миграции, убедиться что все таблицы созданы корректно

- [x] 3. Backend: Playlist Service
  - [x] 3.1 Создать PlaylistService с CRUD операциями
    - create(), update(), delete(), addTrack(), removeTrack(), reorderTracks()
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  
  - [x] 3.2 Создать PlaylistController с REST API endpoints
    - POST /api/playlists, GET /api/playlists, PUT /api/playlists/{id}, DELETE /api/playlists/{id}
    - POST /api/playlists/{id}/tracks, DELETE /api/playlists/{id}/tracks/{trackId}
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [x] 3.3 Написать property test для целостности плейлистов
    - **Property 6: Целостность плейлистов**
    - **Validates: Requirements 5.2, 5.3, 5.4, 5.6**

- [x] 4. Backend: Track Rating Service (5-star)
  - [x] 4.1 Создать TrackRatingService
    - setRating(), removeRating(), getRating(), getHistory(), getTracksWithRating()
    - Автоматическая запись в rating_history при изменении
    - _Requirements: 11.1, 11.2, 11.3, 14.1_
  
  - [x] 4.2 Создать TrackRatingController с REST API
    - POST /api/tracks/{id}/rating, DELETE /api/tracks/{id}/rating, GET /api/tracks/{id}/rating/history
    - _Requirements: 11.1, 11.3, 14.3_
  
  - [x] 4.3 Написать property test для жизненного цикла рейтинга
    - **Property 13: Жизненный цикл 5-звёздочного рейтинга**
    - **Validates: Requirements 11.1, 11.2, 11.3, 14.1**
  
  - [x] 4.4 Написать property test для формулы веса рейтинга
    - **Property 15: Формула веса рейтинга в рекомендациях**
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5**

- [x] 5. Backend: Fragment Rating Service
  - [x] 5.1 Расширить FragmentService для поддержки лайк/дизлайк и комментариев
    - rateFragment(), updateRating(), removeRating(), addComment(), updateComment(), removeComment()
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.1, 8.3, 8.4_
  
  - [x] 5.2 Создать FragmentController endpoints
    - POST /api/tracks/{id}/fragments/rate, PUT /api/fragments/{id}/rating
    - POST /api/fragments/{id}/comment, PUT /api/fragments/{id}/comment
    - _Requirements: 7.1, 8.1_
  
  - [x] 5.3 Написать property test для жизненного цикла рейтинга фрагментов
    - **Property 9: Жизненный цикл рейтинга фрагментов**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.7**
  
  - [x] 5.4 Написать property test для комментариев (round-trip)
    - **Property 10: Round-trip для комментариев к фрагментам**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4**

- [x] 6. Checkpoint — Проверить backend сервисы
  - Убедиться что все API endpoints работают, тесты проходят

- [x] 7. Backend: User Preferences Service
  - [x] 7.1 Создать UserTrackPreferenceService
    - saveVolume(), savePosition(), getPreferences()
    - _Requirements: 4.1, 4.2, 4.4_
  
  - [x] 7.2 Создать API endpoints для preferences
    - PUT /api/tracks/{id}/preferences
    - _Requirements: 4.1, 4.2_
  
  - [x] 7.3 Написать property test для round-trip настроек
    - **Property 5: Round-trip для пользовательских настроек трека**
    - **Validates: Requirements 4.1, 4.2, 4.5**

- [x] 8. Backend: Listening History & Skip Tracking
  - [x] 8.1 Создать ListeningHistoryService
    - recordPlay(), recordComplete(), getRecentHistory()
    - _Requirements: 16.3, 16.4_
  
  - [x] 8.2 Создать SkipTrackerService
    - recordSkip(), getSkipCount(), getSkipStats()
    - _Requirements: 18.1, 18.2_
  
  - [x] 8.3 Создать API endpoints
    - POST /api/tracks/{id}/play, POST /api/tracks/{id}/skip
    - _Requirements: 16.3, 18.1_
  
  - [x] 8.4 Написать property test для отслеживания пропусков
    - **Property 18: Отслеживание пропусков**
    - **Validates: Requirements 18.1, 18.3**

- [x] 9. Backend: Smart Radio Service
  - [x] 9.1 Создать SmartRadioService
    - enable(), disable(), getNextRecommendations(), recordFeedback()
    - Интеграция с FAISS для поиска похожих треков
    - _Requirements: 15.1, 15.2, 15.3, 17.1, 17.2_
  
  - [x] 9.2 Создать SmartRadioController
    - POST /api/radio/enable, POST /api/radio/disable, GET /api/radio/recommendations
    - POST /api/radio/feedback
    - _Requirements: 15.1, 17.1_
  
  - [x] 9.3 Написать property test для инварианта минимальной очереди
    - **Property 16: Инвариант минимальной очереди Smart Radio**
    - **Validates: Requirements 15.2, 16.1, 16.2**
  
  - [x] 9.4 Написать property test для обратной связи
    - **Property 17: Обратная связь Smart Radio**
    - **Validates: Requirements 17.1, 17.2, 17.4**

- [x] 10. Checkpoint — Проверить Smart Radio
  - Убедиться что рекомендации генерируются, история учитывается

- [x] 11. ML Worker: AI DJ Analyzer
  - [x] 11.1 Создать AIDJAnalyzer класс в Python
    - analyze() — определение Camelot key, BPM, energy, структурных точек
    - Использовать Essentia для анализа
    - _Requirements: 27.1, 27.2, 27.3, 27.4_
  
  - [x] 11.2 Добавить Camelot Wheel конвертер
    - _to_camelot() — преобразование музыкального ключа в Camelot нотацию
    - _Requirements: 27.1_
  
  - [x] 11.3 Интегрировать с очередью задач
    - Обработка задач dj_analyze из Redis
    - Сохранение результатов в track_dj_analysis
    - _Requirements: 27.5, 27.6_
  
  - [x] 11.4 Написать property test для валидности Camelot ключа
    - **Property 19: Валидность Camelot ключа**
    - **Validates: Requirements 27.1, 27.3, 27.4**

- [x] 12. ML Worker: AI DJ Matcher
  - [x] 12.1 Создать AIDJMatcher класс
    - find_compatible_tracks() — поиск совместимых треков по Camelot, BPM, energy
    - _calculate_key_score(), _calculate_bpm_score(), _calculate_energy_score()
    - _Requirements: 28.1, 28.2, 28.3, 28.4_
  
  - [x] 12.2 Реализовать Camelot Wheel совместимость
    - Маппинг совместимых ключей (±1 позиция, A↔B переход)
    - _Requirements: 28.1_
  
  - [x] 12.3 Написать property test для Camelot совместимости
    - **Property 20: Совместимость треков по Camelot Wheel**
    - **Validates: Requirements 28.1**
  
  - [x] 12.4 Написать property test для BPM-совместимости
    - **Property 21: BPM-совместимость для DJ микширования**
    - **Validates: Requirements 28.2, 29.4, 29.5**

- [x] 13. Backend: AI DJ API
  - [x] 13.1 Создать AIDJController
    - GET /api/dj/compatible/{trackId} — получить совместимые треки
    - POST /api/dj/analyze/{trackId} — запустить DJ-анализ
    - _Requirements: 28.5, 27.6_

- [x] 14. Checkpoint — Проверить AI DJ backend
  - Убедиться что анализ работает, совместимые треки находятся

- [x] 15. Frontend: Player Store (Pinia)
  - [x] 15.1 Создать playerStore с состоянием и действиями
    - State: currentTrack, queue, isPlaying, volume, shuffleMode, repeatMode, crossfadeEnabled
    - Actions: play, pause, next, previous, addToQueue, removeFromQueue, reorderQueue
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.3, 2.4_
  
  - [x] 15.2 Написать property test для инвариантов очереди
    - **Property 1: Инварианты очереди воспроизведения**
    - **Validates: Requirements 1.1, 1.4, 1.5**
  
  - [x] 15.3 Написать property test для shuffle round-trip
    - **Property 2: Round-trip для режима Shuffle**
    - **Validates: Requirements 2.1, 2.2**

- [x] 16. Frontend: Audio Engine с Crossfade
  - [x] 16.1 Создать AudioEngine класс с Web Audio API
    - Два AudioBufferSourceNode для crossfade
    - GainNode для управления громкостью
    - _Requirements: 3.1, 3.2_
  
  - [x] 16.2 Реализовать crossfade логику
    - prepareCrossfade(), executeCrossfade()
    - Линейное изменение громкости
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [x] 16.3 Написать property test для громкости при crossfade
    - **Property 3: Громкость при Crossfade**
    - **Validates: Requirements 3.2**
  
  - [x] 16.4 Написать property test для валидации длительности crossfade
    - **Property 4: Валидация длительности Crossfade**
    - **Validates: Requirements 3.3, 3.5**

- [x] 17. Frontend: Player Component
  - [x] 17.1 Создать PlayerComponent.vue
    - Интеграция с playerStore и AudioEngine
    - Элементы управления: play/pause, next/prev, volume, progress bar
    - _Requirements: 1.7, 2.6, 5.7_
  
  - [x] 17.2 Добавить Queue Panel
    - Отображение очереди с drag-and-drop
    - Кнопки добавления/удаления
    - _Requirements: 1.6, 1.7_
  
  - [x] 17.3 Добавить режимы Shuffle и Repeat
    - Toggle кнопки с визуальным состоянием
    - _Requirements: 2.1, 2.3, 2.4, 2.5, 2.6_

- [x] 18. Checkpoint — Проверить базовый плеер
  - Убедиться что воспроизведение, очередь, crossfade работают

- [x] 19. Frontend: Waveform с Fragment Overlay
  - [x] 19.1 Интегрировать Wavesurfer.js
    - Отображение формы волны
    - Навигация по клику
    - _Requirements: 10.1_
  
  - [x] 19.2 Добавить Fragment Overlay
    - Отображение лайкнутых фрагментов зелёным, дизлайкнутых красным
    - Иконки на границах фрагментов
    - _Requirements: 7.5, 7.6, 10.2, 10.3_
  
  - [x] 19.3 Добавить выделение области для рейтинга
    - Drag для выделения временного диапазона
    - Кнопки лайк/дизлайк для выделенной области
    - _Requirements: 7.1, 7.2_
  
  - [x] 19.4 Добавить tooltip с комментариями
    - Отображение комментария при наведении на фрагмент
    - _Requirements: 8.5_

- [x] 20. Frontend: Rating Widget (5-star)
  - [x] 20.1 Создать RatingWidget.vue
    - 5 звёзд с hover эффектом
    - Поддержка половинных звёзд для среднего рейтинга
    - _Requirements: 13.3, 13.4, 13.5_
  
  - [x] 20.2 Интегрировать в Player и Library
    - _Requirements: 13.1, 13.4_

- [x] 21. Frontend: Playlist Management
  - [x] 21.1 Создать PlaylistList.vue
    - Список плейлистов с возможностью создания/удаления
    - _Requirements: 5.1, 5.4_
  
  - [x] 21.2 Создать PlaylistDetail.vue
    - Список треков с drag-and-drop для reorder
    - Кнопка воспроизведения плейлиста
    - _Requirements: 5.6, 5.7_
  
  - [x] 21.3 Написать property test для загрузки плейлиста в очередь
    - **Property 7: Загрузка плейлиста в очередь**
    - **Validates: Requirements 5.7**

- [x] 22. Checkpoint — Проверить UI компоненты
  - Убедиться что waveform, рейтинги, плейлисты работают

- [x] 23. Frontend: Smart Radio UI
  - [x] 23.1 Добавить Smart Radio toggle в Player
    - Кнопка включения/выключения
    - Индикатор активного режима
    - _Requirements: 15.1, 15.6_
  
  - [x] 23.2 Добавить кнопки "Больше/Меньше такого"
    - Отображаются только в режиме Smart Radio
    - _Requirements: 17.5_
  
  - [x] 23.3 Добавить индикатор генерации рекомендаций
    - "Подбираю следующие треки..."
    - _Requirements: 19.5_

- [x] 24. Frontend: Theme System
  - [x] 24.1 Создать ThemeProvider с CSS variables
    - Тёмная тема по умолчанию (#121212, #1DB954)
    - Светлая тема как альтернатива
    - _Requirements: 21.1, 21.5_
  
  - [x] 24.2 Реализовать динамический градиент на основе обложки
    - Извлечение доминантного цвета из album art
    - _Requirements: 21.2_
  
  - [x] 24.3 Добавить переключатель темы в настройки
    - Сохранение выбора в user_settings
    - _Requirements: 21.5, 21.6_

- [x] 25. Frontend: Animation System
  - [x] 25.1 Создать базовые CSS transitions
    - Fade-in для появления элементов
    - Scale/glow для hover эффектов
    - _Requirements: 22.1, 22.2_
  
  - [x] 25.2 Добавить Vue transitions для списков
    - Анимация добавления/удаления в Queue
    - _Requirements: 22.4_
  
  - [x] 25.3 Добавить режим "Уменьшить движение"
    - prefers-reduced-motion media query
    - _Requirements: 22.6_

- [x] 26. Frontend: Audio Visualizer
  - [x] 26.1 Создать Visualizer компонент с Web Audio API
    - AnalyserNode для получения frequency data
    - Canvas для отрисовки
    - _Requirements: 23.1, 23.2_
  
  - [x] 26.2 Реализовать режимы визуализации
    - Bars, wave, circular
    - _Requirements: 23.3_
  
  - [x] 26.3 Добавить переключатель режимов и отключение
    - _Requirements: 23.4, 23.5_

- [x] 27. Checkpoint — Проверить UI/UX
  - Убедиться что тема, анимации, визуализация работают

- [x] 28. Frontend: Beatmatcher
  - [x] 28.1 Создать Beatmatcher класс
    - prepareBeatmatch() — подготовка перехода
    - Time-stretching через OfflineAudioContext
    - _Requirements: 29.1, 29.2, 29.3_
  
  - [x] 28.2 Интегрировать с AudioEngine
    - Автоматический beatmatch при AI DJ режиме
    - _Requirements: 29.6_
  
  - [x] 28.3 Написать property test для time-stretching
    - **Property 22: Time-stretching сохраняет длительность**
    - **Validates: Requirements 29.2, 29.3**

- [x] 29. Frontend: AI DJ Live Mode
  - [x] 29.1 Добавить AI DJ toggle в Player
    - Включение/выключение режима
    - _Requirements: 32.1_
  
  - [x] 29.2 Реализовать автоматический подбор и переход
    - Подготовка следующего трека за 30 секунд до конца
    - Автоматический beatmatch и переход
    - _Requirements: 32.2, 32.3_
  
  - [x] 29.3 Добавить визуализацию предстоящего перехода
    - Timeline с двумя waveforms
    - _Requirements: 32.6_

- [x] 30. Backend: AI Playlist Generator
  - [x] 30.1 Создать AIPlaylistGeneratorService
    - generate() — генерация плейлиста на основе параметров и Preference_Model
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [x] 30.2 Создать API endpoint
    - POST /api/playlists/generate
    - _Requirements: 6.1_
  
  - [x] 30.3 Написать property test для соответствия фильтрам
    - **Property 8: AI-плейлист соответствует фильтрам**
    - **Validates: Requirements 6.2, 6.4, 6.5**

- [x] 31. Frontend: AI Playlist Generator UI
  - [x] 31.1 Создать AIPlaylistDialog.vue
    - Форма с параметрами: длительность, жанр, настроение, BPM
    - _Requirements: 6.4_
  
  - [x] 31.2 Интегрировать в Playlist Management
    - Кнопка "Сгенерировать AI плейлист"
    - _Requirements: 6.1_

- [x] 32. Checkpoint — Проверить AI DJ и AI Playlists
  - Убедиться что AI DJ режим работает, плейлисты генерируются

- [x] 33. Frontend: Responsive Design
  - [x] 33.1 Адаптировать Player для мобильных устройств
    - Компактный режим для экранов < 768px
    - _Requirements: 25.1, 25.2_
  
  - [x] 33.2 Добавить полноэкранный режим
    - Увеличенная обложка и визуализация
    - _Requirements: 25.3_
  
  - [x] 33.3 Добавить мини-плеер
    - Picture-in-picture для фонового воспроизведения
    - _Requirements: 25.4_
  
  - [x] 33.4 Адаптировать Waveform для touch
    - Touch-жесты для навигации и выделения
    - _Requirements: 25.6_

- [x] 34. Frontend: Page Transitions
  - [x] 34.1 Настроить Vue Router transitions
    - Slide/fade переходы между страницами
    - _Requirements: 26.1_
  
  - [x] 34.2 Реализовать shared element transitions
    - Анимация обложки альбома при переходе к деталям трека
    - _Requirements: 26.5_

- [x] 35. Frontend: Microinteractions
  - [x] 35.1 Добавить анимацию рейтинга
    - Glow эффект при установке звёзд
    - _Requirements: 24.1_
  
  - [x] 35.2 Добавить toast уведомления
    - Анимированные уведомления о действиях
    - _Requirements: 24.2_
  
  - [x] 35.3 Добавить skeleton загрузчики
    - Placeholder контент во время загрузки
    - _Requirements: 24.4_
  
  - [x] 35.4 Добавить pulse эффект для лайков фрагментов
    - _Requirements: 24.3_

- [x] 36. Final Checkpoint — Полное тестирование
  - Убедиться что все компоненты интегрированы
  - Все property tests проходят
  - UI работает на desktop и mobile

## Примечания

- Все задачи являются обязательными для полного покрытия тестами
- Каждая задача ссылается на конкретные требования для трассируемости
- Checkpoints обеспечивают инкрементальную валидацию
- Property tests валидируют универсальные свойства корректности
