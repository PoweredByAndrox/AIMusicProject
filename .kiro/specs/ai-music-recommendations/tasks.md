# План реализации: AI Music Recommendations

## Обзор

Реализация персональной системы музыкальных рекомендаций с Docker-контейнеризацией. Проект включает Laravel-бэкенд, Vue.js-фронтенд, Python ML-воркер и инфраструктуру (PostgreSQL, Redis, FAISS).

## Задачи

- [x] 1. Настройка Docker-инфраструктуры и базовой структуры проекта
  - [x] 1.1 Создать docker-compose.yml с контейнерами: app, ml-worker, postgres, redis, node
    - Настроить сети и volumes для персистентности данных
    - Настроить переменные окружения
    - _Requirements: 13.1_
  
  - [x] 1.2 Создать Dockerfile для Laravel-приложения (PHP 8.2+, nginx)
    - Установить необходимые PHP-расширения (pdo_pgsql, redis)
    - Настроить nginx конфигурацию
    - _Requirements: 13.1_
  
  - [x] 1.3 Создать Dockerfile для ML-воркера (Python 3.11+)
    - Установить PyTorch, Essentia, FAISS, yt-dlp
    - Настроить поддержку GPU (опционально)
    - _Requirements: 3.1, 13.1_
  
  - [x] 1.4 Инициализировать Laravel 11 проект с базовой конфигурацией
    - Настроить подключение к PostgreSQL и Redis
    - Настроить файловое хранилище
    - _Requirements: 4.1, 13.1_

- [x] 2. Checkpoint - Проверить запуск всех контейнеров
  - Убедиться, что все контейнеры запускаются без ошибок
  - Проверить связность между сервисами

- [x] 3. Реализация моделей данных и миграций
  - [x] 3.1 Создать миграции PostgreSQL для таблиц tracks, fragments, preference_model_status, queue_tasks
    - Добавить все поля согласно схеме из дизайна
    - Создать индексы для оптимизации запросов
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [x] 3.2 Создать Eloquent модели Track, Fragment с отношениями
    - Настроить fillable, casts, relationships
    - Добавить scopes для фильтрации
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [x] 3.3 Написать property-тест для Track Round-Trip Consistency
    - **Property 1: Track Round-Trip Consistency**
    - **Validates: Requirements 4.1, 4.2**
  
  - [x] 3.4 Написать property-тест для Fragment Time Range Validity
    - **Property 2: Fragment Time Range Validity**
    - **Validates: Requirements 4.3, 6.1, 6.2**

- [x] 4. Реализация загрузки локальных аудиофайлов
  - [x] 4.1 Создать TrackController с методом upload()
    - Валидация формата файла (MP3, WAV)
    - Валидация размера файла (max 100MB)
    - Сохранение файла в storage
    - _Requirements: 1.1, 1.3, 1.5_
  
  - [x] 4.2 Создать MetadataExtractor сервис для извлечения ID3 тегов
    - Использовать getID3 библиотеку
    - Извлекать title, artist, album, year, duration
    - _Requirements: 1.2_
  
  - [x] 4.3 Создать TrackAnalysisJob для постановки в очередь Redis
    - Формировать payload для ML-воркера
    - Обновлять статус трека на 'processing'
    - _Requirements: 1.4, 13.2_
  
  - [x] 4.4 Написать property-тест для File Format Validation
    - **Property 15: File Format Validation**
    - **Validates: Requirements 1.1, 1.3**

- [x] 5. Реализация импорта с YouTube
  - [x] 5.1 Создать YouTubeImporter сервис
    - Валидация URL (видео, плейлист, канал)
    - Определение типа URL
    - _Requirements: 2.1, 2.2, 2.3, 2.5_
  
  - [x] 5.2 Создать YouTubeDownloadJob для асинхронной загрузки
    - Интеграция с yt-dlp через shell
    - Извлечение метаданных YouTube
    - Обработка плейлистов и каналов
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  
  - [x] 5.3 Добавить метод importYoutube() в TrackController
    - Обработка прогресса загрузки
    - Возврат статуса импорта
    - _Requirements: 2.6_
  
  - [x] 5.4 Написать property-тест для YouTube URL Validation
    - **Property 16: YouTube URL Validation**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.5**

- [x] 6. Checkpoint - Проверить загрузку файлов и YouTube импорт
  - Убедиться, что файлы загружаются и сохраняются
  - Проверить извлечение метаданных
  - Проверить постановку задач в очередь Redis

- [x] 7. Реализация ML-воркера: базовая инфраструктура
  - [x] 7.1 Создать QueueWorker класс для обработки задач из Redis
    - Подключение к Redis
    - Цикл обработки задач
    - Обновление статусов в PostgreSQL
    - _Requirements: 13.2, 13.3, 13.4_
  
  - [x] 7.2 Создать AudioAnalyzer класс с базовой структурой
    - Загрузка аудио через librosa/essentia
    - Интерфейс для всех методов анализа
    - _Requirements: 3.1_
  
  - [x] 7.3 Написать property-тест для Queue Task Processing Completeness
    - **Property 13: Queue Task Processing Completeness**
    - **Validates: Requirements 13.2, 13.3**

- [x] 8. Реализация ML-анализа: извлечение эмбеддингов
  - [x] 8.1 Реализовать extract_embeddings() с MusiCNN/EfficientAT
    - Загрузка предобученной модели
    - Извлечение векторного представления
    - Сохранение эмбеддинга в файл
    - _Requirements: 3.1_
  
  - [x] 8.2 Написать property-тест для Embedding Dimension Consistency
    - **Property 3: Embedding Dimension Consistency**
    - **Validates: Requirements 3.1, 3.8**

- [x] 9. Реализация ML-анализа: определение характеристик
  - [x] 9.1 Реализовать detect_genre() через Essentia
    - Использовать предобученный классификатор жанров
    - _Requirements: 3.2_
  
  - [x] 9.2 Реализовать detect_bpm() через TempoCNN
    - Определение темпа трека
    - _Requirements: 3.3_
  
  - [x] 9.3 Реализовать detect_key() для определения тональности
    - Определение ключа и режима (major/minor)
    - _Requirements: 3.4_
  
  - [x] 9.4 Реализовать detect_mood() для определения настроения
    - Классификация настроения (Happy, Sad, Aggressive и т.д.)
    - _Requirements: 3.5_
  
  - [x] 9.5 Реализовать detect_vocals() и detect_acoustic()
    - Определение наличия вокала
    - Определение акустический/электронный
    - _Requirements: 3.6_
  
  - [x] 9.6 Реализовать analyze_full() для полного анализа трека
    - Объединение всех методов анализа
    - Сохранение результатов в БД
    - _Requirements: 3.7_
  
  - [x] 9.7 Написать property-тест для ML Characteristics Valid Range
    - **Property 4: ML Characteristics Valid Range**
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6**

- [x] 10. Реализация FAISS индекса
  - [x] 10.1 Создать FAISSManager класс
    - Инициализация индекса с правильной размерностью
    - Методы add_embedding, remove_embedding
    - Методы save_index, load_index
    - _Requirements: 3.8, 4.5_
  
  - [x] 10.2 Реализовать search_similar() для поиска похожих
    - Поиск k ближайших соседей
    - Возврат ID и расстояний
    - _Requirements: 10.1, 10.2_
  
  - [x] 10.3 Написать property-тест для FAISS Index Synchronization
    - **Property 5: FAISS Index Synchronization**
    - **Validates: Requirements 4.5, 3.8**
  
  - [x] 10.4 Написать property-тест для Similar Search Returns K Results
    - **Property 10: Similar Search Returns K Results**
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4**

- [x] 11. Checkpoint - Проверить ML-анализ и FAISS
  - Убедиться, что треки анализируются корректно
  - Проверить добавление эмбеддингов в FAISS
  - Проверить поиск похожих треков

- [x] 12. Реализация системы лайков
  - [x] 12.1 Создать LikeController с методами likeTrack, unlikeTrack, likeFragment, unlikeFragment
    - Toggle логика для лайков
    - Обновление is_liked в БД
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [x] 12.2 Добавить проверку порога лайков для обучения модели
    - Уведомление при достижении 10 лайков
    - _Requirements: 7.5_
  
  - [x] 12.3 Написать property-тест для Like Toggle Idempotence
    - **Property 6: Like Toggle Idempotence**
    - **Validates: Requirements 7.1, 7.2, 7.3**

- [x] 13. Реализация модели предпочтений
  - [x] 13.1 Создать PreferenceModel класс в ML-воркере
    - Архитектура: MLP над эмбеддингами
    - Метод train() с BPR алгоритмом
    - Методы save(), load()
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  
  - [x] 13.2 Создать PreferenceModelController в Laravel
    - Метод train() для запуска обучения
    - Метод status() для получения статуса
    - _Requirements: 8.5, 8.6_
  
  - [x] 13.3 Реализовать predict() для ранжирования треков
    - Предсказание релевантности для всех треков
    - _Requirements: 9.1, 9.2_
  
  - [x] 13.4 Написать property-тест для Preference Model Round-Trip
    - **Property 8: Preference Model Round-Trip**
    - **Validates: Requirements 8.3**

- [x] 14. Реализация рекомендаций
  - [x] 14.1 Создать RecommendationController с методом getRecommendations()
    - Использование обученной модели
    - Фильтрация по жанру, настроению, BPM
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [x] 14.2 Реализовать findSimilarTracks() и findSimilarFragments()
    - Интеграция с FAISS через очередь
    - Возврат результатов с оценкой схожести
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  
  - [x] 14.3 Написать property-тест для Recommendations Sorted by Score
    - **Property 9: Recommendations Sorted by Score**
    - **Validates: Requirements 9.2**

- [x] 15. Checkpoint - Проверить лайки, модель и рекомендации
  - Убедиться, что лайки сохраняются корректно
  - Проверить обучение модели
  - Проверить генерацию рекомендаций

- [x] 16. Реализация фрагментов
  - [x] 16.1 Создать FragmentController с методами store, index, destroy
    - Создание фрагмента с start_time, end_time
    - Валидация временного диапазона
    - _Requirements: 6.1, 6.2, 6.5_
  
  - [x] 16.2 Создать FragmentAnalysisJob для анализа фрагментов
    - Извлечение аудио-сегмента
    - Извлечение эмбеддинга фрагмента
    - _Requirements: 6.6_
  
  - [x] 16.3 Добавить эмбеддинги фрагментов в отдельный FAISS индекс
    - Поиск похожих фрагментов
    - _Requirements: 10.2_

- [x] 17. Реализация ручной корректировки характеристик
  - [x] 17.1 Добавить метод update() в TrackController
    - Обновление жанра, BPM, тональности, настроения
    - Сохранение флага manually_edited
    - _Requirements: 14.1, 14.2, 14.3_
  
  - [x] 17.2 Модифицировать ML-анализ для учёта manually_edited
    - Пропуск полей с ручными правками при повторном анализе
    - _Requirements: 14.4_
  
  - [x] 17.3 Написать property-тест для Manual Edit Preservation
    - **Property 14: Manual Edit Preservation**
    - **Validates: Requirements 14.1, 14.2, 14.3, 14.4**

- [x] 18. Реализация удаления треков
  - [x] 18.1 Реализовать метод destroy() в TrackController
    - Удаление записи из БД
    - Удаление связанных фрагментов
    - Удаление файлов (аудио, эмбеддинги)
    - Удаление из FAISS индекса
    - _Requirements: 4.6_
  
  - [x] 18.2 Написать property-тест для Cascade Delete Completeness
    - **Property 7: Cascade Delete Completeness**
    - **Validates: Requirements 4.6**

- [x] 19. Checkpoint - Проверить фрагменты и удаление
  - Убедиться, что фрагменты создаются и анализируются
  - Проверить каскадное удаление

- [x] 20. Реализация библиотеки треков (Backend)
  - [x] 20.1 Реализовать метод index() в TrackController с фильтрацией
    - Фильтрация по жанру, настроению, BPM, лайкам
    - Сортировка по различным полям
    - Поиск по названию и исполнителю
    - Пагинация
    - _Requirements: 11.1, 11.2, 11.3, 11.4_
  
  - [x] 20.2 Написать property-тест для Library Filtering Correctness
    - **Property 11: Library Filtering Correctness**
    - **Validates: Requirements 11.2, 11.3, 11.4**

- [x] 21. Реализация экспорта в CSV
  - [x] 21.1 Создать ExportController с методом exportCsv()
    - Формирование CSV с метаданными и характеристиками
    - Streaming response для больших библиотек
    - _Requirements: 12.1, 12.2, 12.3, 12.4_
  
  - [x] 21.2 Написать property-тест для CSV Export Round-Trip
    - **Property 12: CSV Export Round-Trip**
    - **Validates: Requirements 12.1, 12.2**

- [x] 22. Checkpoint - Проверить библиотеку и экспорт
  - Убедиться, что фильтрация работает корректно
  - Проверить экспорт в CSV

- [x] 23. Реализация Frontend: базовая структура
  - [x] 23.1 Настроить Vue.js 3 с Vite в Laravel
    - Установить зависимости
    - Настроить TypeScript
    - Создать базовую структуру компонентов
    - _Requirements: 5.1, 11.1_
  
  - [x] 23.2 Создать API-клиент для взаимодействия с бэкендом
    - Axios instance с базовой конфигурацией
    - TypeScript типы для API responses
    - _Requirements: 11.1_

- [x] 24. Реализация Waveform Player
  - [x] 24.1 Создать WaveformPlayer компонент с Wavesurfer.js
    - Отображение формы волны
    - Навигация по клику
    - Zoom функциональность
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [x] 24.2 Добавить элементы управления воспроизведением
    - Play, pause, stop, громкость
    - Отображение текущей позиции
    - _Requirements: 5.5_
  
  - [x] 24.3 Реализовать разметку фрагментов на waveform
    - Выделение области для создания фрагмента
    - Отображение существующих фрагментов
    - Клик на фрагмент для воспроизведения
    - _Requirements: 6.1, 6.3, 6.4_

- [x] 25. Реализация Library View
  - [x] 25.1 Создать TrackLibrary компонент
    - Таблица/список треков с метаданными
    - Отображение статуса анализа
    - Отображение статуса лайка
    - _Requirements: 11.1, 11.6_
  
  - [x] 25.2 Добавить фильтрацию и сортировку
    - Фильтры по жанру, настроению, BPM, лайкам
    - Сортировка по различным полям
    - Поиск по названию и исполнителю
    - _Requirements: 11.2, 11.3, 11.4_
  
  - [x] 25.3 Реализовать открытие трека в Waveform Player
    - Переход к детальному просмотру
    - _Requirements: 11.5_

- [x] 26. Реализация Upload и Import UI
  - [x] 26.1 Создать TrackUpload компонент
    - Drag-and-drop загрузка файлов
    - Валидация формата и размера на клиенте
    - Отображение прогресса загрузки
    - _Requirements: 1.1, 1.3, 1.5_
  
  - [x] 26.2 Создать YouTubeImport компонент
    - Форма ввода URL
    - Отображение прогресса импорта
    - Обработка ошибок
    - _Requirements: 2.1, 2.5, 2.6_

- [x] 27. Реализация Recommendations View
  - [x] 27.1 Создать RecommendationsView компонент
    - Список рекомендованных треков
    - Фильтры для рекомендаций
    - Кнопки лайка и воспроизведения
    - _Requirements: 9.1, 9.3, 9.5_
  
  - [x] 27.2 Создать SimilarTracks компонент
    - Отображение похожих треков
    - Оценка схожести
    - _Requirements: 10.3, 10.4_

- [x] 28. Реализация UI для модели предпочтений
  - [x] 28.1 Создать PreferenceModelStatus компонент
    - Отображение статуса модели
    - Кнопка запуска обучения
    - Прогресс обучения
    - _Requirements: 8.5, 8.6_

- [x] 29. Checkpoint - Проверить весь Frontend
  - Убедиться, что все компоненты работают
  - Проверить интеграцию с бэкендом
  - Проверить waveform player

- [x] 30. Финальная интеграция и тестирование
  - [x] 30.1 Провести интеграционное тестирование всех компонентов
    - Полный цикл: загрузка → анализ → лайк → обучение → рекомендации
    - _Requirements: все_
  
  - [x] 30.2 Оптимизация производительности
    - Индексы БД
    - Кэширование
    - Lazy loading
    - _Requirements: все_
  
  - [x] 30.3 Документация API и развёртывания
    - README с инструкциями запуска
    - API документация
    - _Requirements: все_

- [x] 31. Final Checkpoint - Убедиться, что все тесты проходят
  - Запустить все unit и property тесты
  - Проверить работу системы end-to-end

## Примечания

- Все задачи являются обязательными для полного покрытия тестами
- Каждая задача ссылается на конкретные требования для трассируемости
- Checkpoints обеспечивают инкрементальную валидацию
- Property-тесты валидируют универсальные свойства корректности
- Unit-тесты фокусируются на конкретных примерах и edge cases
