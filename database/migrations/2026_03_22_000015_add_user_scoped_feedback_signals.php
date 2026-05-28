<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('user_track_likes', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->cascadeOnDelete();
            $table->foreignId('track_id')->constrained('tracks')->cascadeOnDelete();
            $table->timestamps();

            $table->unique(['user_id', 'track_id'], 'uq_user_track_likes_user_track');
            $table->index('track_id', 'idx_user_track_likes_track_id');
        });

        Schema::table('fragments', function (Blueprint $table) {
            $table->foreignId('user_id')->nullable()->after('track_id')->constrained()->cascadeOnDelete();
            $table->index(['user_id', 'track_id'], 'idx_fragments_user_track');
        });

        Schema::table('listening_history', function (Blueprint $table) {
            $table->foreignId('user_id')->nullable()->after('track_id')->constrained()->cascadeOnDelete();
            $table->index(['user_id', 'started_at'], 'idx_listening_history_user_started');
        });

        Schema::create('user_pattern_example_ratings', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->cascadeOnDelete();
            $table->foreignId('pattern_example_id')->constrained('pattern_examples')->cascadeOnDelete();
            $table->float('rating');
            $table->boolean('is_positive');
            $table->timestamps();

            $table->unique(['user_id', 'pattern_example_id'], 'uq_user_pattern_example_ratings_user_segment');
            $table->index(['user_id', 'rating'], 'idx_user_pattern_example_ratings_user_rating');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('user_pattern_example_ratings');

        Schema::table('listening_history', function (Blueprint $table) {
            $table->dropIndex('idx_listening_history_user_started');
            $table->dropConstrainedForeignId('user_id');
        });

        Schema::table('fragments', function (Blueprint $table) {
            $table->dropIndex('idx_fragments_user_track');
            $table->dropConstrainedForeignId('user_id');
        });

        Schema::dropIfExists('user_track_likes');
    }
};
