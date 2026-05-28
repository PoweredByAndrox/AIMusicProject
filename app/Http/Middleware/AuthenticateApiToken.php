<?php

namespace App\Http\Middleware;

use App\Models\ApiToken;
use App\Models\User;
use Closure;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Cache;
use Symfony\Component\HttpFoundation\Response;

class AuthenticateApiToken
{
    private const AUTH_CACHE_TTL_SECONDS = 30;
    private const LAST_USED_TOUCH_INTERVAL_SECONDS = 300;

    /**
     * Authenticate the request using a bearer token or ?token= fallback.
     *
     * @param  Closure(Request): Response  $next
     */
    public function handle(Request $request, Closure $next): Response
    {
        $plainToken = $request->bearerToken() ?: $request->query('token');

        if (!$plainToken) {
            return $this->unauthorized('Authentication token is required');
        }

        $tokenHash = hash('sha256', $plainToken);
        $token = $this->resolveToken($tokenHash);

        if (!$token || !$token->user) {
            return $this->unauthorized('Authentication token is invalid');
        }

        if ($token->expires_at && $token->expires_at->isPast()) {
            Cache::forget($this->authCacheKey($tokenHash));
            return $this->unauthorized('Authentication token has expired');
        }

        $this->touchLastUsedAt($token);

        Auth::setUser($token->user);
        $request->setUserResolver(fn () => $token->user);
        $request->attributes->set('api_token_model', $token);

        return $next($request);
    }

    private function resolveToken(string $tokenHash): ?ApiToken
    {
        $cached = Cache::remember(
            $this->authCacheKey($tokenHash),
            self::AUTH_CACHE_TTL_SECONDS,
            static function () use ($tokenHash): ?array {
                $token = ApiToken::with('user')
                    ->where('token_hash', $tokenHash)
                    ->first();

                if (!$token || !$token->user) {
                    return null;
                }

                return [
                    'token' => $token->getAttributes(),
                    'user' => $token->user->getAttributes(),
                ];
            },
        );

        if (!is_array($cached) || !is_array($cached['token'] ?? null) || !is_array($cached['user'] ?? null)) {
            return null;
        }

        /** @var ApiToken $token */
        $token = (new ApiToken())->newFromBuilder($cached['token']);
        /** @var User $user */
        $user = (new User())->newFromBuilder($cached['user']);
        $token->setRelation('user', $user);

        return $token;
    }

    private function touchLastUsedAt(ApiToken $token): void
    {
        $tokenId = (int) $token->getKey();
        if ($tokenId <= 0) {
            return;
        }

        $touchCacheKey = 'api_token:last_used_touch:' . $tokenId;
        if (Cache::has($touchCacheKey)) {
            return;
        }

        Cache::put($touchCacheKey, true, self::LAST_USED_TOUCH_INTERVAL_SECONDS);
        ApiToken::query()->whereKey($tokenId)->update(['last_used_at' => now()]);
    }

    private function authCacheKey(string $tokenHash): string
    {
        return 'api_token:auth:' . $tokenHash;
    }

    private function unauthorized(string $message): JsonResponse
    {
        return response()->json([
            'error' => 'Unauthorized',
            'message' => $message,
        ], 401);
    }
}
