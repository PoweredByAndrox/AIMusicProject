# Pattern Search - Task List

## Поиск мелодических паттернов

### 1. Инфраструктура (Backend)
- [x] 1.1 Создать миграции для таблиц patterns, pattern_examples, pattern_matches
- [x] 1.2 Создать модели Pattern, PatternExample, PatternMatch
- [x] 1.3 Добавить chroma_path, pitch_contour_path в tracks
- [x] 1.4 Создать PatternController с CRUD операциями
- [x] 1.5 Добавить API роуты для паттернов

### 2. ML Worker - Извлечение фич
- [x] 2.1 Создать PatternAnalyzer класс
- [x] 2.2 Реализовать extract_chroma() с librosa
- [x] 2.3 Реализовать extract_pitch_contour() с Essentia
- [x] 2.4 Добавить обработчики в main.py

### 3. ML Worker - Поиск паттернов
- [x] 3.1 Реализовать DTW для сравнения chroma
- [x] 3.2 Добавить транспозиционную инвариантность
- [x] 3.3 Реализовать window-based индексирование
- [x] 3.4 Добавить handle_search_pattern()

### 4. Frontend - API клиент
- [x] 4.1 Добавить типы Pattern, PatternExample, PatternMatch
- [x] 4.2 Добавить API методы для паттернов

### 5. Frontend - UI компоненты
- [x] 5.1 Создать PatternList.vue - список паттернов
- [x] 5.2 Создать PatternDetail.vue - детали паттерна с примерами
- [x] 5.3 Добавить "Сохранить как паттерн" в WaveformPlayer
- [x] 5.4 Создать PatternMatches.vue - результаты поиска (в PatternDetail)

### 6. Интеграция с рекомендациями
- [x] 6.1 Добавить pattern_scores в фичи трека
- [x] 6.2 Использовать паттерны в ранжировании
- [x] 6.3 Показывать паттерн-матчи в объяснениях

---

## Статус: 100% выполнено

Всё готово:
- Миграции и модели созданы
- API endpoints работают
- ML worker извлекает chroma/pitch и ищет паттерны
- UI компоненты для работы с паттернами
- Интеграция паттернов в систему рекомендаций
