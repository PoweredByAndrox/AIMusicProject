<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('tracks', function (Blueprint $table) {
            // Top-N genre tags with probabilities: [{"tag": "Dubstep", "prob": 0.85}, ...]
            $table->jsonb('genre_tags')->nullable()->after('genre');
            // Top-N mood tags with probabilities: [{"tag": "Energetic", "prob": 0.72}, ...]
            $table->jsonb('mood_tags')->nullable()->after('mood');
        });
    }

    public function down(): void
    {
        Schema::table('tracks', function (Blueprint $table) {
            $table->dropColumn(['genre_tags', 'mood_tags']);
        });
    }
};
