<?php

namespace App\Services;

use App\Models\ApiToken;
use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Str;

class RadioAccessTokenService
{
    private const TTL_SECONDS = 600;
    private const CACHE_PREFIX = 'public_radio_access_token:';

    /**
     * @return array{access_token:string,expires_at:string,expires_in:int}
     */
    public function issue(User $user): array
    {
        $accessToken = Str::random(64);
        $payload = [
            'user_id' => (int) $user->id,
            'issued_at' => now()->toIso8601String(),
            'expires_at' => now()->addSeconds(self::TTL_SECONDS)->toIso8601String(),
        ];

        Cache::put($this->cacheKey($accessToken), $payload, now()->addSeconds(self::TTL_SECONDS));

        return [
            'access_token' => $accessToken,
            'expires_at' => $payload['expires_at'],
            'expires_in' => self::TTL_SECONDS,
        ];
    }

    public function resolve(string $accessToken): ?User
    {
        $token = trim($accessToken);
        if ($token === '') {
            return null;
        }

        $payload = Cache::get($this->cacheKey($token));
        if (!is_array($payload)) {
            return null;
        }

        $userId = (int) ($payload['user_id'] ?? 0);
        if ($userId <= 0) {
            return null;
        }

        return User::find($userId);
    }

    public function resolveFromRequest(Request $request): ?User
    {
        $accessToken = $request->query('access_token');

        if (!is_string($accessToken) || trim($accessToken) === '') {
            $headerToken = $request->header('X-Radio-Access-Token');
            $accessToken = is_string($headerToken) ? $headerToken : '';
        }

        $resolved = $this->resolve((string) $accessToken);
        if ($resolved) {
            return $resolved;
        }

        $plainToken = $request->bearerToken();
        if (!is_string($plainToken) || trim($plainToken) === '') {
            $queryToken = $request->query('token');
            $plainToken = is_string($queryToken) ? $queryToken : '';
        }

        if (!is_string($plainToken) || trim($plainToken) === '') {
            $plainToken = is_string($accessToken) ? $accessToken : '';
        }

        return $this->resolveApiToken($plainToken);
    }

    private function cacheKey(string $accessToken): string
    {
        return self::CACHE_PREFIX . $accessToken;
    }

    private function resolveApiToken(string $plainToken): ?User
    {
        $token = trim($plainToken);
        if ($token === '') {
            return null;
        }

        $apiToken = ApiToken::with('user')
            ->where('token_hash', hash('sha256', $token))
            ->first();

        if (!$apiToken || !$apiToken->user) {
            return null;
        }

        if ($apiToken->expires_at && $apiToken->expires_at->isPast()) {
            return null;
        }

        $apiToken->forceFill(['last_used_at' => now()])->save();

        return $apiToken->user;
    }
}
