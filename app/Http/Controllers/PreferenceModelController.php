<?php

namespace App\Http\Controllers;

use App\Models\PreferenceModelStatus;
use App\Services\PreferenceTrainingService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;

/**
 * PreferenceModelController
 *
 * Handles preference model training and status.
 */
class PreferenceModelController extends Controller
{
    public function train(Request $request): JsonResponse
    {
        $userId = (int) $request->user()->id;
        $stats = PreferenceTrainingService::getStats($userId);

        if ($stats['total_positive'] < 1) {
            return response()->json([
                'error' => 'Insufficient data',
                'message' => 'Нужен хотя бы один рейтинг 4+ для обучения вашей модели',
                'stats' => $stats,
            ], 400);
        }

        $status = PreferenceModelStatus::forUser($userId)->first();
        if ($status && $status->status === 'training') {
            $isStuck = $status->updated_at && $status->updated_at->lt(now()->subMinutes(10));
            if (!$isStuck) {
                return response()->json([
                    'error' => 'Already training',
                    'message' => 'Ваша модель уже обучается',
                    'stats' => $stats,
                ], 400);
            }
        }

        PreferenceTrainingService::triggerTraining($userId);

        return response()->json([
            'message' => 'Training started',
            'stats' => $stats,
            'status' => 'training',
        ], 202);
    }

    public function status(Request $request): JsonResponse
    {
        $userId = (int) $request->user()->id;
        $status = PreferenceModelStatus::forUser($userId)->first();
        $stats = PreferenceTrainingService::getStats($userId);

        if (!$status) {
            return response()->json([
                'status' => 'not_trained',
                'trained_at' => null,
                'can_train' => $stats['total_positive'] >= 1,
                'message' => 'Ваша модель ещё не обучена',
                'stats' => $stats,
            ]);
        }

        return response()->json([
            'status' => $status->status,
            'trained_at' => $status->trained_at?->toIso8601String(),
            'model_path' => $status->model_path,
            'can_train' => $stats['total_positive'] >= 1 && $status->status !== 'training',
            'stats' => $stats,
        ]);
    }
}
