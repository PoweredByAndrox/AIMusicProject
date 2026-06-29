<?php

namespace App\Services;

use App\Models\Track;
use App\Models\Fragment;
use App\Models\ListeningHistory;
use App\Models\TrackRating;
use App\Models\UserTrackLike;
use App\Models\UserTrackScore;
use App\Services\MLQueueService;
use App\Services\TrackRatingService;
use App\Services\FragmentService;
use App\Services\PreferenceTrainingService;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Redis;

/**
 * LLMChatService - AI-powered music chat using a local LLM backend.
 * 
 * Understands natural language queries about music library and
 * translates them into search/filter operations.
 */
class LLMChatService
{
    private string $ollamaUrl;
    private string $provider;
    private string $openAiCompatibleUrl;
    private string $model;
    private int $timeout;
    private TrackRatingService $ratingService;
    private FragmentService $fragmentService;

    public function __construct()
    {
        $this->provider = strtolower((string) env('LLM_PROVIDER', 'ollama'));
        $this->ollamaUrl = env('OLLAMA_URL', 'http://ollama:11434');
        $this->openAiCompatibleUrl = rtrim((string) env('VLLM_URL', 'http://vllm:8000/v1'), '/');
        $defaultModel = in_array($this->provider, ['vllm', 'openai_compatible', 'openai-compatible'], true)
            ? env('VLLM_MODEL', '')
            : env('OLLAMA_MODEL', 'llama3.1:8b');
        $this->model = (string) env('LLM_MODEL', $defaultModel);
        $this->timeout = (int) env('LLM_TIMEOUT', env('OLLAMA_TIMEOUT', 120));
        $this->ratingService = app(TrackRatingService::class);
        $this->fragmentService = app(FragmentService::class);
    }

    /**
     * Process a natural language chat message about music.
     */
    public function chat(string $message, ?array $context = null): array
    {
        // Build system prompt with library context
        $systemPrompt = $this->buildSystemPrompt($context);
        
        // Call configured local LLM backend.
        $response = $this->callLLM($systemPrompt, $message);
        
        if (isset($response['error'])) {
            return $response;
        }
        
        // Parse LLM response and execute actions
        return $this->processLLMResponse($response['content'], $message);
    }

    /**
     * Build system prompt with music library context.
     */
    private function buildSystemPrompt(?array $context): string
    {
        // Get library stats
        $stats = $this->getLibraryStats();
        
        // Get available genres and moods
        $genres = Track::whereNotNull('genre')
            ->distinct()
            ->pluck('genre')
            ->take(30)
            ->toArray();
        
        $moods = Track::whereNotNull('mood')
            ->distinct()
            ->pluck('mood')
            ->take(10)
            ->toArray();

        $prompt = <<<PROMPT
Ты - AI-ассистент для музыкальной библиотеки. Помогаешь пользователю находить треки, управлять коллекцией и обучаешь систему его вкусам.

СТАТИСТИКА БИБЛИОТЕКИ:
- Всего треков: {$stats['total']}
- Проанализировано: {$stats['analyzed']}
- С лайками: {$stats['liked']}

ДОСТУПНЫЕ ЖАНРЫ: {$this->formatList($genres)}

ДОСТУПНЫЕ НАСТРОЕНИЯ: {$this->formatList($moods)}

{$this->getFragmentFeedbackContext()}

ТВОИ ВОЗМОЖНОСТИ:
1. SEARCH - поиск треков по критериям (жанр, настроение, BPM, исполнитель, название)
2. RECOMMEND - рекомендации на основе предпочтений пользователя
3. INFO - информация о конкретном треке
4. SIMILAR - поиск похожих треков
5. STATS - статистика библиотеки
6. LIKE - поставить лайк треку (улучшает рекомендации)
7. DISLIKE - убрать лайк с трека
8. RATE - оценить трек от 1 до 5 звёзд
9. RATE_FRAGMENT - оценить фрагмент трека (лайк/дизлайк с комментарием)
10. LEARN - обучить систему вкусам пользователя (анализирует описание и предлагает оценки)
11. FRAGMENTS - показать оценённые фрагменты трека
12. IMPORT - импортировать YouTube ссылки (одну или список)

ФОРМАТ ОТВЕТА:
Всегда отвечай в JSON формате:
{
  "action": "SEARCH|RECOMMEND|INFO|SIMILAR|STATS|LIKE|DISLIKE|RATE|RATE_FRAGMENT|LEARN|FRAGMENTS|IMPORT|CHAT",
  "params": { ... параметры для действия ... },
  "message": "Текст ответа пользователю на русском"
}

Примеры params для SEARCH:
- {"genre": "Drum & Bass", "bpm_min": 170, "bpm_max": 180}
- {"mood": "Energetic", "has_vocals": false}
- {"artist": "Noisia"}

Для SIMILAR:
- {"track_id": 123} или {"track_title": "название трека"}

Для RECOMMEND:
- {"count": 5, "genre": "House"} или просто {"count": 5}

Для LIKE/DISLIKE:
- {"track_id": 123} или {"track_title": "название трека"}

Для RATE:
- {"track_id": 123, "rating": 5} или {"track_title": "название", "rating": 4}

Для RATE_FRAGMENT (оценка фрагмента):
- {"track_title": "название", "start_time": 30, "end_time": 60, "is_liked": true, "comment": "крутой дроп"}
- {"track_id": 123, "start_time": 90, "end_time": 120, "is_liked": false, "comment": "скучный момент"}
- Время в секундах. Комментарий объясняет почему нравится/не нравится.

Для FRAGMENTS (показать фрагменты):
- {"track_title": "название"} или {"track_id": 123}

Для LEARN (обучение вкусам):
- {"preferences": "люблю энергичную музыку 170+ BPM", "auto_apply": false}
- Если auto_apply=true, автоматически применит оценки. Если false - только покажет что сделает.

Для IMPORT (импорт YouTube):
- {"urls": ["https://youtube.com/watch?v=xxx", "https://youtube.com/watch?v=yyy"]}
- {"url": "https://youtube.com/watch?v=xxx"} - одна ссылка
- Поддерживает видео, плейлисты и каналы. Автоматически ставит лайк импортированным трекам если указано like_all=true.

Если запрос не связан с музыкой, используй action: "CHAT" и просто ответь.

ВАЖНО: 
- Отвечай ТОЛЬКО валидным JSON. Никакого текста до или после JSON.
- При LEARN с auto_apply=false сначала покажи план, потом спроси подтверждение.
- Лайки, оценки и комментарии к фрагментам улучшают персональные рекомендации.
- Комментарии к фрагментам помогают системе понять ПОЧЕМУ пользователю нравится/не нравится музыка.
PROMPT;

        // Add current track context if available
        if ($context && isset($context['current_track'])) {
            $track = $context['current_track'];
            $prompt .= "\n\nСЕЙЧАС ИГРАЕТ: {$track['title']} - {$track['artist']} ({$track['genre']}, {$track['bpm']} BPM)";
        }

        return $prompt;
    }

    /**
     * Call configured local LLM backend.
     */
    private function callLLM(string $systemPrompt, string $userMessage): array
    {
        return match ($this->provider) {
            'vllm', 'openai_compatible', 'openai-compatible' => $this->callOpenAICompatible($systemPrompt, $userMessage),
            default => $this->callOllama($systemPrompt, $userMessage),
        };
    }

    /**
     * Call Ollama API.
     */
    private function callOllama(string $systemPrompt, string $userMessage): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->ollamaUrl}/api/chat", [
                    'model' => $this->model,
                    'messages' => [
                        ['role' => 'system', 'content' => $systemPrompt],
                        ['role' => 'user', 'content' => $userMessage],
                    ],
                    'stream' => false,
                    'options' => [
                        'temperature' => 0.3,
                        'num_predict' => 500,
                    ],
                ]);

            if (!$response->successful()) {
                Log::error('Ollama API error', ['status' => $response->status(), 'body' => $response->body()]);
                return ['error' => 'LLM сервис недоступен. Попробуйте позже.'];
            }

            $data = $response->json();
            return ['content' => $data['message']['content'] ?? ''];

        } catch (\Exception $e) {
            Log::error('Ollama connection error', ['error' => $e->getMessage()]);
            return ['error' => 'Не удалось подключиться к LLM. Убедитесь что Ollama запущен.'];
        }
    }

    /**
     * Process LLM response and execute the requested action.
     */
    private function processLLMResponse(string $llmResponse, string $originalMessage): array
    {
        // Try to parse JSON from response
        $parsed = $this->parseJsonResponse($llmResponse);
        
        if (!$parsed) {
            // Fallback: treat as simple chat response
            return [
                'type' => 'chat',
                'message' => $llmResponse,
                'tracks' => [],
            ];
        }

        $action = strtoupper($parsed['action'] ?? 'CHAT');
        $params = $parsed['params'] ?? [];
        $message = $parsed['message'] ?? '';

        switch ($action) {
            case 'SEARCH':
                $tracks = $this->executeSearch($params);
                return [
                    'type' => 'search',
                    'message' => $message ?: "Найдено " . count($tracks) . " треков",
                    'tracks' => $tracks,
                    'query' => $params,
                ];

            case 'RECOMMEND':
                $tracks = $this->executeRecommend($params);
                $explanations = $this->formatTrackExplanations($tracks);
                return [
                    'type' => 'recommendations',
                    'message' => $message ?: ($explanations ?: "Рекомендую " . count($tracks) . " треков"),
                    'tracks' => $tracks,
                ];

            case 'SIMILAR':
                $tracks = $this->executeSimilar($params);
                return [
                    'type' => 'similar',
                    'message' => $message ?: "Похожие треки (" . count($tracks) . ")",
                    'tracks' => $tracks,
                ];

            case 'STATS':
                $stats = $this->getLibraryStats();
                return [
                    'type' => 'stats',
                    'message' => $message,
                    'stats' => $stats,
                    'tracks' => [],
                ];

            case 'INFO':
                $track = $this->getTrackInfo($params);
                return [
                    'type' => 'info',
                    'message' => $message,
                    'tracks' => $track ? [$track] : [],
                ];

            case 'LIKE':
                $result = $this->executeLike($params);
                return [
                    'type' => 'like',
                    'message' => $result['message'],
                    'tracks' => $result['track'] ? [$result['track']] : [],
                    'success' => $result['success'],
                ];

            case 'DISLIKE':
                $result = $this->executeDislike($params);
                return [
                    'type' => 'dislike',
                    'message' => $result['message'],
                    'tracks' => $result['track'] ? [$result['track']] : [],
                    'success' => $result['success'],
                ];

            case 'RATE':
                $result = $this->executeRate($params);
                return [
                    'type' => 'rate',
                    'message' => $result['message'],
                    'tracks' => $result['track'] ? [$result['track']] : [],
                    'success' => $result['success'],
                ];

            case 'LEARN':
                $result = $this->executeLearn($params);
                return [
                    'type' => 'learn',
                    'message' => $result['message'],
                    'tracks' => $result['tracks'] ?? [],
                    'plan' => $result['plan'] ?? null,
                    'applied' => $result['applied'] ?? false,
                ];

            case 'RATE_FRAGMENT':
                $result = $this->executeRateFragment($params);
                return [
                    'type' => 'rate_fragment',
                    'message' => $result['message'],
                    'fragment' => $result['fragment'] ?? null,
                    'tracks' => $result['track'] ? [$result['track']] : [],
                    'success' => $result['success'],
                ];

            case 'FRAGMENTS':
                $result = $this->executeGetFragments($params);
                return [
                    'type' => 'fragments',
                    'message' => $result['message'],
                    'fragments' => $result['fragments'] ?? [],
                    'tracks' => $result['track'] ? [$result['track']] : [],
                ];

            case 'IMPORT':
                $result = $this->executeImport($params);
                return [
                    'type' => 'import',
                    'message' => $result['message'],
                    'queued' => $result['queued'] ?? 0,
                    'invalid' => $result['invalid'] ?? 0,
                    'tracks' => [],
                ];

            default:
                return [
                    'type' => 'chat',
                    'message' => $message ?: $llmResponse,
                    'tracks' => [],
                ];
        }
    }

    /**
     * Parse JSON from LLM response (handles markdown code blocks).
     */
    private function parseJsonResponse(string $response): ?array
    {
        // Try direct JSON parse
        $decoded = json_decode($response, true);
        if ($decoded !== null) {
            return $decoded;
        }

        // Try to extract JSON from markdown code block
        if (preg_match('/```(?:json)?\s*(\{.*?\})\s*```/s', $response, $matches)) {
            $decoded = json_decode($matches[1], true);
            if ($decoded !== null) {
                return $decoded;
            }
        }

        // Try to find JSON object in response
        if (preg_match('/\{[^{}]*"action"[^{}]*\}/s', $response, $matches)) {
            $decoded = json_decode($matches[0], true);
            if ($decoded !== null) {
                return $decoded;
            }
        }

        return null;
    }

    /**
     * Execute search query.
     */
    private function executeSearch(array $params): array
    {
        $query = Track::where('analysis_status', 'completed');

        if (!empty($params['genre'])) {
            $query->where('genre', 'ILIKE', '%' . $params['genre'] . '%');
        }

        if (!empty($params['mood'])) {
            $query->where('mood', 'ILIKE', '%' . $params['mood'] . '%');
        }

        if (!empty($params['artist'])) {
            $query->where('artist', 'ILIKE', '%' . $params['artist'] . '%');
        }

        if (!empty($params['title'])) {
            $query->where('title', 'ILIKE', '%' . $params['title'] . '%');
        }

        if (isset($params['bpm_min'])) {
            $query->where('bpm', '>=', (int) $params['bpm_min']);
        }

        if (isset($params['bpm_max'])) {
            $query->where('bpm', '<=', (int) $params['bpm_max']);
        }

        if (isset($params['has_vocals'])) {
            $query->where('has_vocals', (bool) $params['has_vocals']);
        }

        if (array_key_exists('is_liked', $params)) {
            $isLiked = filter_var($params['is_liked'], FILTER_VALIDATE_BOOLEAN, FILTER_NULL_ON_FAILURE);
            if ($isLiked !== null) {
                $this->applyTrackLikeFilter($query, $isLiked, $this->currentUserId());
            }
        }

        return $query->limit(20)->get()->toArray();
    }

    /**
     * Execute recommendations using embedding similarity + preference model.
     */
    private function executeRecommend(array $params): array
    {
        $count = min($params['count'] ?? 5, 20);
        $genre = $params['genre'] ?? null;
        $mood = $params['mood'] ?? null;
        
        // Try embedding-based recommendations first
        $embeddingResults = $this->getEmbeddingRecommendations($count, $genre, $mood);
        
        if (!empty($embeddingResults)) {
            return $embeddingResults;
        }
        
        // Fallback to scored recommendations
        return $this->getScoredRecommendations($count, $genre, $mood);
    }
    
    /**
     * Get recommendations using ML worker embedding similarity.
     */
    private function getEmbeddingRecommendations(int $count, ?string $genre, ?string $mood): array
    {
        $taskId = uniqid('llm_recommend_', true);
        $userId = $this->currentUserId();
        
        try {
            MLQueueService::pushPriority([
                'type' => 'embedding_recommendations',
                'task_id' => $taskId,
                'user_id' => $userId,
                'limit' => $count,
                'genre_filter' => $genre,
                'mood_filter' => $mood,
                'created_at' => now()->toIso8601String(),
            ]);
            
            // Wait for result with shorter timeout for chat context
            $resultKey = "chat_recommend_result:{$taskId}";
            $startWait = time();
            $timeout = 15;
            
            while (time() - $startWait < $timeout) {
                $result = Redis::connection()->get($resultKey);
                
                if ($result) {
                    Redis::connection()->del($resultKey);
                    $data = json_decode($result, true);
                    
                    // Extract tracks with explanations
                    if (!empty($data['recommendations'])) {
                        return array_map(function ($rec) {
                            $track = $rec['track'] ?? $rec;
                            $track['_explanation'] = $rec['explanation'] ?? null;
                            $track['_similarity'] = $rec['similarity'] ?? null;
                            return $track;
                        }, $data['recommendations']);
                    }
                    
                    return [];
                }
                
                usleep(100000); // 100ms
            }
        } catch (\Exception $e) {
            Log::warning('Embedding recommendations failed', ['error' => $e->getMessage()]);
        }
        
        return [];
    }
    
    /**
     * Fallback: scored recommendations based on user preferences.
     */
    private function getScoredRecommendations(int $count, ?string $genre, ?string $mood): array
    {
        $userId = $this->currentUserId();

        // Get user's liked tracks
        $likedTracks = Track::likedByUser($userId)
            ->where('analysis_status', 'completed')
            ->get();
        
        // Get highly rated tracks (4-5 stars)
        $highRatedTrackIds = TrackRating::forUser($userId)
            ->where('rating', '>=', 4)
            ->pluck('track_id')
            ->toArray();
        
        $highRatedTracks = Track::whereIn('id', $highRatedTrackIds)
            ->where('analysis_status', 'completed')
            ->get();
        
        // Combine for preference profile
        $preferredTracks = $likedTracks->merge($highRatedTracks)->unique('id');
        
        if ($preferredTracks->isEmpty()) {
            $query = Track::where('analysis_status', 'completed');
            $this->applyTrackLikeFilter($query, false, $userId);
            
            if ($genre) {
                $query->where('genre', 'ILIKE', '%' . $genre . '%');
            }
            if ($mood) {
                $query->where('mood', 'ILIKE', '%' . $mood . '%');
            }

            $candidates = $query->get();
            if ($candidates->isEmpty()) {
                return [];
            }

            $preferenceScores = $this->getUserTrackScoreMap($userId, $candidates->pluck('id')->all());
            if (empty($preferenceScores)) {
                return $candidates
                    ->shuffle()
                    ->take($count)
                    ->values()
                    ->toArray();
            }

            return $this->diversifyScoredRecommendations(
                $candidates
                    ->map(function ($track) use ($preferenceScores) {
                        $trackArray = $track->toArray();
                        $trackArray['_score'] = (float) (($preferenceScores[$track->id] ?? 0.0) * 100.0);
                        $trackArray['_explanation'] = ($preferenceScores[$track->id] ?? null) !== null
                            ? 'Похож по звуку на ваши лайки и сегменты'
                            : null;

                        return $trackArray;
                    })
                    ->filter(fn($track) => $track['_score'] > 0),
                $count,
                $genre !== null
            )->values()->toArray();
        }
        
        // Build preference profile
        $preferredGenres = $preferredTracks->pluck('genre')->filter()->countBy()->sortDesc();
        $preferredMoods = $preferredTracks->pluck('mood')->filter()->countBy()->sortDesc();
        $avgBpm = $preferredTracks->avg('bpm') ?: 140;
        
        // Get listening history to exclude
        $listenedTrackIds = ListeningHistory::forUser($userId)->pluck('track_id')->unique()->toArray();
        
        // Get candidates
        $query = Track::where('analysis_status', 'completed')
            ->whereNotIn('id', $listenedTrackIds);
        $this->applyTrackLikeFilter($query, false, $userId);
        
        if ($genre) {
            $query->where('genre', 'ILIKE', '%' . $genre . '%');
        }
        if ($mood) {
            $query->where('mood', 'ILIKE', '%' . $mood . '%');
        }
        
        $candidates = $query->get();
        
        if ($candidates->isEmpty()) {
            return [];
        }

        $preferenceScores = $this->getUserTrackScoreMap($userId, $candidates->pluck('id')->all());
        
        // Score and rank candidates
        $ranked = $candidates->map(function ($track) use ($preferredGenres, $preferredMoods, $avgBpm, $preferenceScores) {
            $preferenceScore = $preferenceScores[$track->id] ?? null;
            $score = $preferenceScore !== null
                ? (float) $preferenceScore * 100.0
                : 35.0;
            $reasons = [];

            if ($preferenceScore !== null) {
                $reasons[] = 'похож по звуку на ваши лайки и сегменты';
            }
            
            // Genre matching
            if ($track->genre && $preferredGenres->isNotEmpty()) {
                $topGenre = $preferredGenres->keys()->first();
                if ($track->genre === $topGenre) {
                    $score += 4;
                    $reasons[] = "ваш любимый жанр";
                } elseif ($preferredGenres->has($track->genre)) {
                    $score += 4;
                    $reasons[] = "жанр вам нравится";
                }
            }
            
            // BPM matching
            if ($track->bpm) {
                $bpmDiff = abs($track->bpm - $avgBpm);
                if ($bpmDiff < 10) {
                    $score += 6;
                    $reasons[] = "подходящий темп";
                } elseif ($bpmDiff < 20) {
                    $score += 3;
                }
            }
            
            // Mood matching
            if ($track->mood && $preferredMoods->isNotEmpty() && $preferredMoods->has($track->mood)) {
                $score += 5;
                $reasons[] = "подходящее настроение";
            }
            
            $trackArray = $track->toArray();
            $trackArray['_score'] = $score;
            $trackArray['_explanation'] = !empty($reasons) 
                ? ucfirst(implode(', ', $reasons))
                : null;
            
            return $trackArray;
        })
        ->filter(fn($t) => $t['_score'] >= 35);

        return $this->diversifyScoredRecommendations($ranked, $count, $genre !== null)
            ->values()
            ->toArray();
    }

    /**
     * Call an OpenAI-compatible local LLM API, including vLLM.
     */
    private function callOpenAICompatible(string $systemPrompt, string $userMessage): array
    {
        if (trim($this->model) === '') {
            return ['error' => 'LLM модель для OpenAI-compatible backend не настроена. Укажите LLM_MODEL или VLLM_MODEL.'];
        }

        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->openAiCompatibleUrl}/chat/completions", [
                    'model' => $this->model,
                    'messages' => [
                        ['role' => 'system', 'content' => $systemPrompt],
                        ['role' => 'user', 'content' => $userMessage],
                    ],
                    'stream' => false,
                    'temperature' => 0.3,
                    'max_tokens' => 500,
                ]);

            if (!$response->successful()) {
                Log::error('OpenAI-compatible LLM API error', ['status' => $response->status(), 'body' => $response->body()]);
                return ['error' => 'LLM сервис недоступен. Попробуйте позже.'];
            }

            $data = $response->json();
            return ['content' => $data['choices'][0]['message']['content'] ?? ''];

        } catch (\Exception $e) {
            Log::error('OpenAI-compatible LLM connection error', ['error' => $e->getMessage()]);
            return ['error' => 'Не удалось подключиться к LLM. Убедитесь что локальный LLM backend запущен.'];
        }
    }

    private function diversifyScoredRecommendations($tracks, int $count, bool $preserveGenreOrder = false)
    {
        $pool = $tracks->sortByDesc('_score')->take(max($count * 5, 20))->values();

        if ($preserveGenreOrder) {
            return $pool->take($count)->values();
        }

        $selected = collect();
        $genreCounts = [];

        while ($selected->count() < $count && $pool->isNotEmpty()) {
            $bestIndex = 0;
            $bestAdjustedScore = null;

            foreach ($pool as $index => $track) {
                $genreKey = strtolower(trim((string) ($track['genre'] ?? '')));
                $penalty = $genreKey !== ''
                    ? min(6.0, (($genreCounts[$genreKey] ?? 0) * 2.5))
                    : 0.0;
                $adjustedScore = (float) $track['_score'] - $penalty;

                if ($bestAdjustedScore === null || $adjustedScore > $bestAdjustedScore) {
                    $bestAdjustedScore = $adjustedScore;
                    $bestIndex = $index;
                }
            }

            $chosen = $pool->pull($bestIndex);
            if ($chosen === null) {
                break;
            }

            $selected->push($chosen);

            $genreKey = strtolower(trim((string) ($chosen['genre'] ?? '')));
            if ($genreKey !== '') {
                $genreCounts[$genreKey] = ($genreCounts[$genreKey] ?? 0) + 1;
            }
        }

        return $selected;
    }

    /**
     * Find similar tracks using embedding similarity.
     */
    private function executeSimilar(array $params): array
    {
        $sourceTrack = null;

        if (!empty($params['track_id'])) {
            $sourceTrack = Track::find($params['track_id']);
        } elseif (!empty($params['track_title'])) {
            $sourceTrack = Track::where('title', 'ILIKE', '%' . $params['track_title'] . '%')->first();
        }

        if (!$sourceTrack) {
            return [];
        }

        // Try embedding-based similarity first
        $embeddingResults = $this->getEmbeddingSimilar($sourceTrack->id, 10);
        
        if (!empty($embeddingResults)) {
            return $embeddingResults;
        }

        // Fallback: Find tracks with similar characteristics
        return Track::where('id', '!=', $sourceTrack->id)
            ->where('analysis_status', 'completed')
            ->where(function ($q) use ($sourceTrack) {
                $q->where('genre', $sourceTrack->genre)
                  ->orWhere('mood', $sourceTrack->mood)
                  ->orWhereBetween('bpm', [$sourceTrack->bpm - 10, $sourceTrack->bpm + 10]);
            })
            ->limit(10)
            ->get()
            ->toArray();
    }
    
    /**
     * Get similar tracks using ML worker embedding similarity.
     */
    private function getEmbeddingSimilar(int $trackId, int $limit): array
    {
        $taskId = uniqid('llm_similar_', true);
        
        try {
            MLQueueService::push([
                'type' => 'find_similar',
                'task_id' => $taskId,
                'track_id' => $trackId,
                'limit' => $limit,
                'created_at' => now()->toIso8601String(),
            ]);
            
            $resultKey = "similar_result:{$taskId}";
            $startWait = time();
            $timeout = 15;
            
            while (time() - $startWait < $timeout) {
                $result = Redis::connection()->get($resultKey);
                
                if ($result) {
                    Redis::connection()->del($resultKey);
                    $data = json_decode($result, true);
                    
                    if (!empty($data['similar_tracks'])) {
                        return array_map(function ($item) {
                            $track = $item['track'] ?? $item;
                            $track['_similarity'] = $item['similarity'] ?? null;
                            return $track;
                        }, $data['similar_tracks']);
                    }
                    
                    return [];
                }
                
                usleep(100000);
            }
        } catch (\Exception $e) {
            Log::warning('Embedding similar search failed', ['error' => $e->getMessage()]);
        }
        
        return [];
    }

    /**
     * Get track info.
     */
    private function getTrackInfo(array $params): ?array
    {
        if (!empty($params['track_id'])) {
            $track = Track::find($params['track_id']);
        } elseif (!empty($params['track_title'])) {
            $track = Track::where('title', 'ILIKE', '%' . $params['track_title'] . '%')->first();
        } else {
            return null;
        }

        return $track?->toArray();
    }

    /**
     * Execute like action.
     */
    private function executeLike(array $params): array
    {
        $track = $this->findTrackByParams($params);
        
        if (!$track) {
            return [
                'success' => false,
                'message' => 'Трек не найден',
                'track' => null,
            ];
        }
        
        $this->setTrackLiked($track, true);
        
        // NOTE: Auto-training disabled — user triggers training manually via UI button
        
        return [
            'success' => true,
            'message' => "Лайк поставлен: {$track->title}" . ($track->artist ? " — {$track->artist}" : ""),
            'track' => $track->fresh()->toArray(),
        ];
    }

    /**
     * Execute dislike (remove like) action.
     */
    private function executeDislike(array $params): array
    {
        $track = $this->findTrackByParams($params);
        
        if (!$track) {
            return [
                'success' => false,
                'message' => 'Трек не найден',
                'track' => null,
            ];
        }
        
        $this->setTrackLiked($track, false);
        
        return [
            'success' => true,
            'message' => "Лайк убран: {$track->title}",
            'track' => $track->fresh()->toArray(),
        ];
    }

    /**
     * Execute rate action.
     */
    private function executeRate(array $params): array
    {
        $track = $this->findTrackByParams($params);
        
        if (!$track) {
            return [
                'success' => false,
                'message' => 'Трек не найден',
                'track' => null,
            ];
        }
        
        $rating = (int) ($params['rating'] ?? 0);
        
        if ($rating < 1 || $rating > 5) {
            return [
                'success' => false,
                'message' => 'Рейтинг должен быть от 1 до 5',
                'track' => $track->toArray(),
            ];
        }
        
        try {
            $this->ratingService->setRating($track->id, $rating);
            
            $stars = str_repeat('⭐', $rating);
            return [
                'success' => true,
                'message' => "Оценка {$stars} поставлена: {$track->title}",
                'track' => $track->fresh()->toArray(),
            ];
        } catch (\Exception $e) {
            return [
                'success' => false,
                'message' => 'Ошибка при установке рейтинга: ' . $e->getMessage(),
                'track' => $track->toArray(),
            ];
        }
    }

    /**
     * Execute learn preferences action.
     * Analyzes user's taste description and suggests/applies ratings.
     */
    private function executeLearn(array $params): array
    {
        $userId = $this->currentUserId();
        $preferences = $params['preferences'] ?? '';
        $autoApply = (bool) ($params['auto_apply'] ?? false);
        
        if (empty($preferences)) {
            return [
                'message' => 'Опишите ваши музыкальные предпочтения, например: "люблю энергичную музыку 170+ BPM" или "предпочитаю спокойные треки без вокала"',
                'tracks' => [],
                'plan' => null,
                'applied' => false,
            ];
        }
        
        // Parse preferences into search criteria
        $criteria = $this->parsePreferences($preferences);
        
        // Find matching tracks
        $query = Track::where('analysis_status', 'completed');
        $this->applyTrackLikeFilter($query, false, $userId);
        
        if (!empty($criteria['genre'])) {
            $query->where('genre', 'ILIKE', '%' . $criteria['genre'] . '%');
        }
        if (!empty($criteria['mood'])) {
            $query->where('mood', 'ILIKE', '%' . $criteria['mood'] . '%');
        }
        if (isset($criteria['bpm_min'])) {
            $query->where('bpm', '>=', $criteria['bpm_min']);
        }
        if (isset($criteria['bpm_max'])) {
            $query->where('bpm', '<=', $criteria['bpm_max']);
        }
        if (isset($criteria['has_vocals'])) {
            $query->where('has_vocals', $criteria['has_vocals']);
        }
        
        $matchingTracks = $query->limit(20)->get();
        
        if ($matchingTracks->isEmpty()) {
            return [
                'message' => 'Не нашёл треков, соответствующих описанию. Попробуйте другие критерии.',
                'tracks' => [],
                'plan' => null,
                'applied' => false,
            ];
        }
        
        // Build plan
        $plan = [
            'criteria' => $criteria,
            'tracks_count' => $matchingTracks->count(),
            'action' => 'like_all',
            'tracks' => $matchingTracks->map(fn($t) => [
                'id' => $t->id,
                'title' => $t->title,
                'artist' => $t->artist,
                'genre' => $t->genre,
                'bpm' => $t->bpm,
            ])->toArray(),
        ];
        
        if (!$autoApply) {
            // Just show plan, don't apply
            $trackList = $matchingTracks->take(5)->map(fn($t) => 
                "• {$t->title}" . ($t->artist ? " — {$t->artist}" : "") . " ({$t->genre}, {$t->bpm} BPM)"
            )->join("\n");
            
            $more = $matchingTracks->count() > 5 ? "\n...и ещё " . ($matchingTracks->count() - 5) . " треков" : "";
            
            return [
                'message' => "Нашёл {$matchingTracks->count()} треков по вашим критериям:\n{$trackList}{$more}\n\nСказать \"да\" или \"применить\" чтобы поставить лайки всем этим трекам?",
                'tracks' => $matchingTracks->toArray(),
                'plan' => $plan,
                'applied' => false,
            ];
        }
        
        // Auto apply - like all matching tracks
        $likedCount = 0;
        foreach ($matchingTracks as $track) {
            $this->setTrackLiked($track, true);
            $likedCount++;
        }
        
        // NOTE: Auto-training disabled — user triggers training manually via UI button
        // PreferenceTrainingService::triggerTraining();
        
        return [
            'message' => "Готово! Поставил лайки {$likedCount} трекам. Система рекомендаций теперь учитывает ваши предпочтения.",
            'tracks' => $matchingTracks->toArray(),
            'plan' => $plan,
            'applied' => true,
        ];
    }

    /**
     * Parse natural language preferences into search criteria.
     */
    private function parsePreferences(string $preferences): array
    {
        $criteria = [];
        $text = mb_strtolower($preferences);
        
        // BPM patterns
        if (preg_match('/(\d{2,3})\s*\+\s*bpm/i', $text, $m)) {
            $criteria['bpm_min'] = (int) $m[1];
        }
        if (preg_match('/(\d{2,3})\s*-\s*bpm/i', $text, $m)) {
            $criteria['bpm_max'] = (int) $m[1];
        }
        if (preg_match('/bpm\s*(\d{2,3})\s*-\s*(\d{2,3})/i', $text, $m)) {
            $criteria['bpm_min'] = (int) $m[1];
            $criteria['bpm_max'] = (int) $m[2];
        }
        
        // Mood keywords
        $moodMap = [
            'энергичн' => 'Energetic',
            'спокойн' => 'Calm',
            'агрессивн' => 'Aggressive',
            'меланхолич' => 'Melancholic',
            'весел' => 'Happy',
            'грустн' => 'Sad',
            'тёмн' => 'Dark',
            'светл' => 'Bright',
        ];
        
        foreach ($moodMap as $keyword => $mood) {
            if (mb_strpos($text, $keyword) !== false) {
                $criteria['mood'] = $mood;
                break;
            }
        }
        
        // Genre keywords
        $genres = Track::whereNotNull('genre')
            ->distinct()
            ->pluck('genre')
            ->toArray();
        
        foreach ($genres as $genre) {
            if (mb_stripos($text, mb_strtolower($genre)) !== false) {
                $criteria['genre'] = $genre;
                break;
            }
        }
        
        // Vocals
        if (preg_match('/без\s*вокал/i', $text) || preg_match('/инструментал/i', $text)) {
            $criteria['has_vocals'] = false;
        }
        if (preg_match('/с\s*вокал/i', $text)) {
            $criteria['has_vocals'] = true;
        }
        
        return $criteria;
    }

    /**
     * Find track by ID or title from params.
     */
    private function findTrackByParams(array $params): ?Track
    {
        if (!empty($params['track_id'])) {
            return Track::find($params['track_id']);
        }
        
        if (!empty($params['track_title'])) {
            return Track::where('title', 'ILIKE', '%' . $params['track_title'] . '%')->first();
        }
        
        return null;
    }

    /**
     * Execute rate fragment action.
     */
    private function executeRateFragment(array $params): array
    {
        $track = $this->findTrackByParams($params);
        
        if (!$track) {
            return [
                'success' => false,
                'message' => 'Трек не найден',
                'track' => null,
                'fragment' => null,
            ];
        }
        
        $startTime = $params['start_time'] ?? null;
        $endTime = $params['end_time'] ?? null;
        $isLiked = $params['is_liked'] ?? null;
        $comment = $params['comment'] ?? null;
        
        if ($startTime === null || $endTime === null) {
            return [
                'success' => false,
                'message' => 'Укажите start_time и end_time фрагмента в секундах',
                'track' => $track->toArray(),
                'fragment' => null,
            ];
        }
        
        if ($isLiked === null) {
            return [
                'success' => false,
                'message' => 'Укажите is_liked: true (нравится) или false (не нравится)',
                'track' => $track->toArray(),
                'fragment' => null,
            ];
        }
        
        try {
            $fragment = $this->fragmentService->rateFragment(
                $track->id,
                (float) $startTime,
                (float) $endTime,
                (bool) $isLiked,
                $comment
            );
            
            $emoji = $isLiked ? '👍' : '👎';
            $timeRange = $this->formatTime($startTime) . ' - ' . $this->formatTime($endTime);
            $msg = "{$emoji} Фрагмент {$timeRange} в \"{$track->title}\" оценён";
            if ($comment) {
                $msg .= "\nКомментарий: \"{$comment}\"";
            }
            
            return [
                'success' => true,
                'message' => $msg,
                'track' => $track->toArray(),
                'fragment' => $fragment->toArray(),
            ];
        } catch (\Exception $e) {
            return [
                'success' => false,
                'message' => 'Ошибка: ' . $e->getMessage(),
                'track' => $track->toArray(),
                'fragment' => null,
            ];
        }
    }

    /**
     * Get fragments for a track.
     */
    private function executeGetFragments(array $params): array
    {
        $track = $this->findTrackByParams($params);
        
        if (!$track) {
            return [
                'message' => 'Трек не найден',
                'track' => null,
                'fragments' => [],
            ];
        }
        
        $fragments = Fragment::forUser($this->currentUserId())
            ->where('track_id', $track->id)
            ->whereNotNull('is_liked')
            ->orderBy('start_time')
            ->get();
        
        if ($fragments->isEmpty()) {
            return [
                'message' => "У трека \"{$track->title}\" нет оценённых фрагментов",
                'track' => $track->toArray(),
                'fragments' => [],
            ];
        }
        
        $lines = ["Фрагменты трека \"{$track->title}\":"];
        foreach ($fragments as $f) {
            $emoji = $f->is_liked ? '👍' : '👎';
            $time = $this->formatTime($f->start_time) . ' - ' . $this->formatTime($f->end_time);
            $line = "{$emoji} {$time}";
            if ($f->comment) {
                $line .= " — \"{$f->comment}\"";
            }
            $lines[] = $line;
        }
        
        return [
            'message' => implode("\n", $lines),
            'track' => $track->toArray(),
            'fragments' => $fragments->toArray(),
        ];
    }

    /**
     * Format seconds to MM:SS.
     */
    private function formatTime(float $seconds): string
    {
        $mins = floor($seconds / 60);
        $secs = floor($seconds % 60);
        return sprintf('%d:%02d', $mins, $secs);
    }

    /**
     * Get fragment feedback context for system prompt.
     * Shows recent comments to help LLM understand user preferences.
     */
    private function getFragmentFeedbackContext(): string
    {
        // Get recent fragments with comments
        $likedWithComments = Fragment::with('track')
            ->forUser($this->currentUserId())
            ->where('is_liked', true)
            ->whereNotNull('comment')
            ->orderByDesc('rated_at')
            ->limit(10)
            ->get();
        
        $dislikedWithComments = Fragment::with('track')
            ->forUser($this->currentUserId())
            ->where('is_liked', false)
            ->whereNotNull('comment')
            ->orderByDesc('rated_at')
            ->limit(5)
            ->get();
        
        if ($likedWithComments->isEmpty() && $dislikedWithComments->isEmpty()) {
            return '';
        }
        
        $lines = ["ОБРАТНАЯ СВЯЗЬ ПОЛЬЗОВАТЕЛЯ (комментарии к фрагментам):"];
        
        if ($likedWithComments->isNotEmpty()) {
            $lines[] = "Нравится:";
            foreach ($likedWithComments as $f) {
                $trackTitle = $f->track?->title ?? 'Unknown';
                $lines[] = "- \"{$f->comment}\" ({$trackTitle}, {$this->formatTime($f->start_time)}-{$this->formatTime($f->end_time)})";
            }
        }
        
        if ($dislikedWithComments->isNotEmpty()) {
            $lines[] = "Не нравится:";
            foreach ($dislikedWithComments as $f) {
                $trackTitle = $f->track?->title ?? 'Unknown';
                $lines[] = "- \"{$f->comment}\" ({$trackTitle}, {$this->formatTime($f->start_time)}-{$this->formatTime($f->end_time)})";
            }
        }
        
        $lines[] = "";
        $lines[] = "Используй эту информацию для понимания вкусов пользователя при рекомендациях.";
        
        return implode("\n", $lines);
    }

    /**
     * Execute import YouTube URLs.
     */
    private function executeImport(array $params): array
    {
        $urls = $params['urls'] ?? [];
        $singleUrl = $params['url'] ?? null;
        $likeAll = (bool) ($params['like_all'] ?? true); // Default: like imported tracks
        
        // Support single URL
        if ($singleUrl && empty($urls)) {
            $urls = [$singleUrl];
        }
        
        // Support newline-separated string
        if (is_string($urls)) {
            $urls = array_filter(array_map('trim', explode("\n", $urls)));
        }
        
        if (empty($urls)) {
            return [
                'message' => 'Укажите YouTube ссылки для импорта',
                'queued' => 0,
                'invalid' => 0,
            ];
        }
        
        $youtubeImporter = app(\App\Services\YouTubeImporter::class);
        
        $queued = 0;
        $invalid = 0;
        $errors = [];
        $delay = 0;
        $userId = $this->currentUserId();
        
        foreach ($urls as $url) {
            $url = trim($url);
            if (empty($url)) continue;
            
            $validation = $youtubeImporter->validateUrl($url);
            
            if (!$validation['valid']) {
                $invalid++;
                continue;
            }
            
            // Dispatch with staggered delay
            \App\Jobs\YouTubeDownloadJob::dispatch($url, $validation['type'], $validation['id'], 200, $userId, $likeAll)
                ->delay(now()->addSeconds($delay));
            
            $queued++;
            $delay += 3;
        }
        
        $message = "🎵 Импорт запущен!\n";
        $message .= "• В очереди: {$queued} ссылок\n";
        if ($invalid > 0) {
            $message .= "• Некорректных: {$invalid}\n";
        }
        $message .= "• Треки будут автоматически проанализированы\n";
        if ($likeAll) {
            $message .= "• После анализа треки получат лайки для обучения модели";
        }
        
        return [
            'message' => $message,
            'queued' => $queued,
            'invalid' => $invalid,
        ];
    }

    /**
     * Get library statistics.
     */
    private function getLibraryStats(): array
    {
        $userId = $this->currentUserId();

        return [
            'total' => Track::count(),
            'analyzed' => Track::where('analysis_status', 'completed')->count(),
            'liked' => $userId === null
                ? Track::where('is_liked', true)->count()
                : Track::likedByUser($userId)->count(),
            'pending' => Track::where('analysis_status', 'pending')->count(),
        ];
    }

    private function currentUserId(): ?int
    {
        $userId = Auth::id();

        return $userId === null ? null : (int) $userId;
    }

    private function applyTrackLikeFilter(Builder $query, bool $isLiked, ?int $userId): void
    {
        if ($userId === null) {
            $query->where('is_liked', $isLiked);
            return;
        }

        if ($isLiked) {
            $query->likedByUser($userId);
            return;
        }

        $query->whereDoesntHave('trackLikes', function (Builder $likeQuery) use ($userId) {
            $likeQuery->where('user_id', $userId);
        });
    }

    private function setTrackLiked(Track $track, bool $isLiked): void
    {
        $userId = $this->currentUserId();

        if ($userId === null) {
            $track->is_liked = $isLiked;
            $track->save();
            return;
        }

        if ($isLiked) {
            UserTrackLike::updateOrCreate([
                'user_id' => $userId,
                'track_id' => $track->id,
            ], []);
            return;
        }

        UserTrackLike::query()
            ->where('user_id', $userId)
            ->where('track_id', $track->id)
            ->delete();
    }

    private function getUserTrackScoreMap(?int $userId, array $trackIds): array
    {
        if ($userId === null || empty($trackIds)) {
            return [];
        }

        return UserTrackScore::forUser($userId)
            ->whereIn('track_id', $trackIds)
            ->pluck('preference_score', 'track_id')
            ->toArray();
    }

    /**
     * Format list for prompt.
     */
    private function formatList(array $items): string
    {
        return implode(', ', array_filter($items));
    }
    
    /**
     * Format track explanations for response message.
     */
    private function formatTrackExplanations(array $tracks): ?string
    {
        if (empty($tracks)) {
            return 'Не нашёл подходящих треков по вашим предпочтениям.';
        }
        
        $withExplanations = array_filter($tracks, fn($t) => !empty($t['_explanation']));
        
        if (empty($withExplanations)) {
            return null;
        }
        
        $lines = ["Рекомендую " . count($tracks) . " треков на основе ваших предпочтений:"];
        
        foreach (array_slice($withExplanations, 0, 3) as $track) {
            $title = $track['title'] ?? 'Unknown';
            $artist = $track['artist'] ?? '';
            $explanation = $track['_explanation'];
            
            $line = "• {$title}";
            if ($artist) {
                $line .= " — {$artist}";
            }
            $line .= " ({$explanation})";
            $lines[] = $line;
        }
        
        if (count($tracks) > 3) {
            $lines[] = "...и ещё " . (count($tracks) - 3);
        }
        
        return implode("\n", $lines);
    }

    /**
     * Check if the configured local LLM backend is available and model is loaded.
     */
    public function checkHealth(): array
    {
        if (in_array($this->provider, ['vllm', 'openai_compatible', 'openai-compatible'], true)) {
            return $this->checkOpenAICompatibleHealth();
        }

        try {
            $response = Http::timeout(5)->get("{$this->ollamaUrl}/api/tags");
            
            if (!$response->successful()) {
                return ['status' => 'error', 'message' => 'Ollama недоступен'];
            }

            $models = $response->json()['models'] ?? [];
            $modelNames = array_column($models, 'name');
            
            $hasModel = in_array($this->model, $modelNames) || 
                        in_array(explode(':', $this->model)[0], array_map(fn($m) => explode(':', $m)[0], $modelNames));

            return [
                'status' => 'ok',
                'provider' => 'ollama',
                'ollama_url' => $this->ollamaUrl,
                'model' => $this->model,
                'model_loaded' => $hasModel,
                'available_models' => $modelNames,
            ];

        } catch (\Exception $e) {
            return ['status' => 'error', 'message' => $e->getMessage()];
        }
    }

    private function checkOpenAICompatibleHealth(): array
    {
        if (trim($this->model) === '') {
            return ['status' => 'error', 'message' => 'LLM_MODEL или VLLM_MODEL не задан для OpenAI-compatible backend'];
        }

        try {
            $response = Http::timeout(5)->get("{$this->openAiCompatibleUrl}/models");

            if (!$response->successful()) {
                return ['status' => 'error', 'message' => 'OpenAI-compatible LLM недоступен'];
            }

            $models = $response->json()['data'] ?? [];
            $modelNames = array_values(array_filter(array_map(fn($model) => $model['id'] ?? null, $models)));
            $hasModel = in_array($this->model, $modelNames, true);

            return [
                'status' => 'ok',
                'provider' => $this->provider,
                'openai_compatible_url' => $this->openAiCompatibleUrl,
                'model' => $this->model,
                'model_loaded' => $hasModel,
                'available_models' => $modelNames,
            ];

        } catch (\Exception $e) {
            return ['status' => 'error', 'message' => $e->getMessage()];
        }
    }
}
