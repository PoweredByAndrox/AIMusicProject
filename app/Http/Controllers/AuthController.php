<?php

namespace App\Http\Controllers;

use App\Models\ApiToken;
use App\Models\User;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Facades\Validator;
use Illuminate\Support\Str;

class AuthController extends Controller
{
    public function register(Request $request): JsonResponse
    {
        $validator = Validator::make($request->all(), [
            'name' => ['required', 'string', 'max:255'],
            'email' => ['required', 'string', 'email', 'max:255', 'unique:users,email'],
            'password' => ['required', 'string', 'min:8', 'confirmed'],
        ]);

        if ($validator->fails()) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => $validator->errors()->first(),
                'errors' => $validator->errors(),
            ], 422);
        }

        $role = User::query()->exists() ? User::ROLE_USER : User::ROLE_ADMIN;

        $user = User::create([
            'name' => trim((string) $request->input('name')),
            'email' => strtolower(trim((string) $request->input('email'))),
            'password' => Hash::make((string) $request->input('password')),
            'role' => $role,
        ]);

        [$plainToken, $token] = $this->issueToken($user, 'register');

        return response()->json([
            'message' => 'Registration successful',
            'token' => $plainToken,
            'user' => $this->serializeUser($user),
            'token_id' => $token->id,
        ], 201);
    }

    public function login(Request $request): JsonResponse
    {
        $validator = Validator::make($request->all(), [
            'email' => ['required', 'string', 'email'],
            'password' => ['required', 'string'],
        ]);

        if ($validator->fails()) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => $validator->errors()->first(),
                'errors' => $validator->errors(),
            ], 422);
        }

        $user = User::where('email', strtolower(trim((string) $request->input('email'))))->first();

        if (!$user || !Hash::check((string) $request->input('password'), $user->password)) {
            return response()->json([
                'error' => 'Authentication failed',
                'message' => 'Invalid email or password',
            ], 401);
        }

        [$plainToken, $token] = $this->issueToken($user, 'login');

        return response()->json([
            'message' => 'Login successful',
            'token' => $plainToken,
            'user' => $this->serializeUser($user),
            'token_id' => $token->id,
        ]);
    }

    public function me(Request $request): JsonResponse
    {
        return response()->json([
            'user' => $this->serializeUser($request->user()),
        ]);
    }

    public function logout(Request $request): JsonResponse
    {
        /** @var ApiToken|null $token */
        $token = $request->attributes->get('api_token_model');
        if ($token) {
            $token->delete();
        }

        return response()->json([
            'message' => 'Logged out successfully',
        ]);
    }

    /**
     * @return array{0:string,1:ApiToken}
     */
    private function issueToken(User $user, string $name): array
    {
        $plainToken = Str::random(80);

        $token = ApiToken::create([
            'user_id' => $user->id,
            'name' => $name,
            'token_hash' => hash('sha256', $plainToken),
            'created_at' => now(),
        ]);

        return [$plainToken, $token];
    }

    private function serializeUser(User $user): array
    {
        return [
            'id' => $user->id,
            'name' => $user->name,
            'email' => $user->email,
            'role' => $user->role,
            'created_at' => $user->created_at?->toIso8601String(),
            'updated_at' => $user->updated_at?->toIso8601String(),
        ];
    }
}
