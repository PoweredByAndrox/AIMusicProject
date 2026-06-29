<?php

namespace Tests\Unit;

use App\Services\LLMChatService;
use Illuminate\Support\Facades\Http;
use Tests\TestCase;

class LLMChatServiceProviderTest extends TestCase
{
    protected function tearDown(): void
    {
        foreach ([
            'LLM_PROVIDER',
            'LLM_MODEL',
            'LLM_TIMEOUT',
            'OLLAMA_URL',
            'OLLAMA_MODEL',
            'VLLM_URL',
            'VLLM_MODEL',
        ] as $key) {
            putenv($key);
            unset($_ENV[$key], $_SERVER[$key]);
        }

        parent::tearDown();
    }

    public function test_ollama_provider_remains_default_for_health_check(): void
    {
        $this->setEnv('OLLAMA_URL', 'http://ollama:11434');
        $this->setEnv('OLLAMA_MODEL', 'llama3.1:8b');

        Http::fake([
            'http://ollama:11434/api/tags' => Http::response([
                'models' => [
                    ['name' => 'llama3.1:8b'],
                ],
            ]),
        ]);

        $health = (new LLMChatService())->checkHealth();

        $this->assertSame('ok', $health['status']);
        $this->assertSame('ollama', $health['provider']);
        $this->assertSame('llama3.1:8b', $health['model']);
        $this->assertTrue($health['model_loaded']);
    }

    public function test_vllm_health_uses_openai_compatible_models_endpoint(): void
    {
        $this->setEnv('LLM_PROVIDER', 'vllm');
        $this->setEnv('VLLM_URL', 'http://vllm:8000/v1');
        $this->setEnv('VLLM_MODEL', 'local/test-llm');

        Http::fake([
            'http://vllm:8000/v1/models' => Http::response([
                'data' => [
                    ['id' => 'local/test-llm'],
                ],
            ]),
        ]);

        $health = (new LLMChatService())->checkHealth();

        $this->assertSame('ok', $health['status']);
        $this->assertSame('vllm', $health['provider']);
        $this->assertSame('http://vllm:8000/v1', $health['openai_compatible_url']);
        $this->assertSame('local/test-llm', $health['model']);
        $this->assertTrue($health['model_loaded']);
    }

    public function test_vllm_chat_uses_openai_compatible_chat_payload(): void
    {
        $this->setEnv('LLM_PROVIDER', 'vllm');
        $this->setEnv('VLLM_URL', 'http://vllm:8000/v1');
        $this->setEnv('VLLM_MODEL', 'local/test-llm');

        Http::fake([
            'http://vllm:8000/v1/chat/completions' => Http::response([
                'choices' => [
                    [
                        'message' => [
                            'content' => '{"action":"CHAT","params":{},"message":"ok"}',
                        ],
                    ],
                ],
            ]),
        ]);

        $service = new LLMChatService();
        $method = new \ReflectionMethod($service, 'callOpenAICompatible');
        $method->setAccessible(true);
        $result = $method->invoke($service, 'system prompt', 'user message');

        $this->assertSame('{"action":"CHAT","params":{},"message":"ok"}', $result['content']);
        Http::assertSent(function ($request) {
            $payload = $request->data();

            return $request->url() === 'http://vllm:8000/v1/chat/completions'
                && $payload['model'] === 'local/test-llm'
                && $payload['stream'] === false
                && $payload['temperature'] === 0.3
                && $payload['max_tokens'] === 500
                && $payload['messages'][0]['role'] === 'system'
                && $payload['messages'][1]['content'] === 'user message';
        });
    }

    private function setEnv(string $key, string $value): void
    {
        putenv("{$key}={$value}");
        $_ENV[$key] = $value;
        $_SERVER[$key] = $value;
    }
}
