# План реализации: AI Music Chat

## Обзор

Реализация чат-интерфейса для поиска похожих музыкальных моментов. Используем существующую инфраструктуру FAISS, AudioAnalyzer и Vue.js компоненты.

## Задачи

- [x] 1. Реализация SegmentIndexManager (Python)
  - [x] 1.1 Создать класс SegmentIndexManager в ml_worker/worker/segment_index.py
    - Инициализация FAISS индекса для сегментов (отдельно от треков)
    - Методы add_segment(), remove_segment(), contains()
    - Хранение метаданных (fragment_id, track_id, start_time, end_time)
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  
  - [x] 1.2 Написать property-тест для инварианта добавления/удаления
    - **Property 4: Инвариант добавления в индекс сегментов**
    - **Property 5: Инвариант удаления из индекса сегментов**
    - **Validates: Requirements 2.1, 2.2, 2.3**
  
  - [x] 1.3 Реализовать методы search_similar() и rebuild_index()
    - Поиск k ближайших соседей с исключением source track
    - Фильтрация по min_similarity
    - Перестроение индекса из БД
    - _Requirements: 2.4, 3.2, 3.4_
  
  - [x] 1.4 Реализовать персистентность индекса (save/load)
    - Сохранение FAISS индекса и метаданных на диск
    - Загрузка при инициализации
    - _Requirements: 2.5_
  
  - [x] 1.5 Написать property-тест для round-trip персистентности
    - **Property 6: Round-trip персистентности индекса**
    - **Validates: Requirements 2.5**

- [x] 2. Checkpoint - Проверить работу SegmentIndexManager
  - Убедиться, что все тесты проходят, задать вопросы при необходимости.

- [x] 3. Интеграция с ML Worker
  - [x] 3.1 Добавить обработчик segment_search в ml_worker/worker/main.py
    - Извлечение эмбеддинга для произвольного сегмента через AudioAnalyzer.extract_embeddings()
    - Поиск похожих сегментов через SegmentIndexManager
    - Возврат результатов через Redis
    - _Requirements: 3.1, 3.2_
  
  - [x] 3.2 Обновить handle_fragment_analysis для добавления в Segment_Index
    - После анализа фрагмента добавлять его в SegmentIndexManager
    - _Requirements: 2.2_

- [x] 4. Реализация QueryParser (TypeScript)
  - [x] 4.1 Создать модуль resources/js/utils/queryParser.ts
    - Функция parseTime() для парсинга "MM:SS" и "Ns" форматов
    - Функция formatTime() для обратного форматирования
    - Функция parseQuery() для извлечения временного диапазона из текста
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  
  - [x] 4.2 Написать property-тест для round-trip парсинга времени
    - **Property 1: Парсинг времени round-trip**
    - **Validates: Requirements 1.1, 1.2**
  
  - [x] 4.3 Реализовать валидацию временного диапазона
    - Проверка start < end
    - Проверка неотрицательности
    - Ограничение end_time длительностью трека
    - _Requirements: 1.5, 1.6_
  
  - [x] 4.4 Написать property-тест для валидации диапазонов
    - **Property 2: Отклонение некорректных временных диапазонов**
    - **Property 3: Ограничение временного диапазона длительностью трека**
    - **Validates: Requirements 1.5, 1.6**

- [x] 5. Checkpoint - Проверить QueryParser
  - Убедиться, что все тесты проходят, задать вопросы при необходимости.

- [x] 6. Реализация Laravel API
  - [x] 6.1 Создать ChatSearchService в app/Services/ChatSearchService.php
    - Метод searchSimilarSegments() для отправки задачи в ML Worker
    - Метод extractSegmentEmbedding() для извлечения эмбеддинга
    - Обработка результатов и формирование ответа
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [x] 6.2 Создать ChatController в app/Http/Controllers/ChatController.php
    - Endpoint POST /api/chat/search
    - Валидация входных параметров (track_id, start_time, end_time, limit, min_similarity)
    - Обработка ошибок 404, 422
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  
  - [x] 6.3 Добавить маршрут в routes/api.php
    - Route::post('chat/search', [ChatController::class, 'search'])
    - Route::get('chat/track/{id}', [ChatController::class, 'getTrack'])
    - _Requirements: 6.1_
  
  - [x] 6.4 Написать property-тест для валидации API
    - **Property 11: Валидация некорректных запросов API**
    - **Validates: Requirements 6.6**

- [x] 7. Checkpoint - Проверить API endpoint
  - Все 10 тестов ChatSearchValidationPropertyTest прошли успешно.

- [x] 8. Реализация MusicChatView (Vue.js)
  - [x] 8.1 Создать компонент resources/js/views/MusicChatView.vue
    - Поле ввода запроса
    - История сообщений (user/system/results)
    - Отображение текущего трека и прогресса
    - _Requirements: 4.1, 4.2, 4.3, 4.5_
  
  - [x] 8.2 Реализовать отображение результатов поиска
    - Список Match_Result с информацией о треке
    - Мини-плеер для каждого результата
    - Кнопка like для добавления в избранное
    - _Requirements: 4.4, 5.2, 5.4_
  
  - [x] 8.3 Интегрировать с playerStore для воспроизведения
    - Клик на результат запускает воспроизведение с start_time
    - Остановка при достижении end_time (опционально)
    - _Requirements: 4.6, 5.1, 5.3_
  
  - [x] 8.4 Добавить API клиент для chat/search
    - Метод searchSimilarSegments() в resources/js/api/client.ts
    - Типы ChatMessage, SearchResult в resources/js/types/index.ts
    - _Requirements: 6.2_

- [x] 9. Реализация истории чата
  - [x] 9.1 Создать модуль resources/js/utils/chatHistory.ts
    - Сохранение истории в localStorage
    - Загрузка истории при инициализации
    - Ограничение размера до 100 сообщений
    - Очистка истории
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [x] 9.2 Написать property-тесты для истории чата
    - **Property 12: Round-trip персистентности истории чата**
    - **Property 13: Инвариант размера истории чата**
    - **Validates: Requirements 7.1, 7.4**

- [x] 10. Интеграция и роутинг
  - [x] 10.1 Добавить маршрут в resources/js/router/index.ts
    - Route /chat для MusicChatView
    - _Requirements: 4.1_
  
  - [x] 10.2 Добавить навигацию в UI
    - Ссылка на чат в основном меню
    - _Requirements: 4.1_

- [x] 11. Финальный checkpoint
  - Все тесты прошли успешно:
    - TypeScript: 28 тестов (queryParser: 20, chatHistory: 8)
    - PHP: 10 тестов (ChatSearchValidationPropertyTest)
    - Python: 7 тестов (test_segment_index.py)
  - Убедиться, что все тесты проходят, задать вопросы при необходимости.

## Примечания

- Все задачи обязательны для полного покрытия тестами
- Каждая задача ссылается на конкретные требования для трассируемости
- Property-тесты валидируют универсальные свойства корректности
- Unit-тесты валидируют конкретные примеры и граничные случаи
