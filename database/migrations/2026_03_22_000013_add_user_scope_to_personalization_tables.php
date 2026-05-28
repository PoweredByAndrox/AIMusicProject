<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('playlists', function (Blueprint $table) {
            $table->foreignId('user_id')->nullable()->after('id')->constrained('users')->nullOnDelete();
            $table->index('user_id', 'idx_playlists_user');
        });

        Schema::table('track_ratings', function (Blueprint $table) {
            $table->foreignId('user_id')->nullable()->after('id')->constrained('users')->nullOnDelete();
        });

        Schema::table('track_ratings', function (Blueprint $table) {
            $table->dropUnique('track_ratings_track_id_unique');
            $table->unique(['user_id', 'track_id'], 'track_ratings_user_track_unique');
            $table->index('user_id', 'idx_track_ratings_user');
        });

        Schema::table('rating_history', function (Blueprint $table) {
            $table->foreignId('user_id')->nullable()->after('id')->constrained('users')->nullOnDelete();
            $table->index('user_id', 'idx_rating_history_user');
        });

        Schema::table('user_track_preferences', function (Blueprint $table) {
            $table->foreignId('user_id')->nullable()->after('id')->constrained('users')->nullOnDelete();
        });

        Schema::table('user_track_preferences', function (Blueprint $table) {
            $table->dropUnique('user_track_preferences_track_id_unique');
            $table->unique(['user_id', 'track_id'], 'user_track_preferences_user_track_unique');
            $table->index('user_id', 'idx_user_track_preferences_user');
        });
    }

    public function down(): void
    {
        Schema::table('user_track_preferences', function (Blueprint $table) {
            $table->dropUnique('user_track_preferences_user_track_unique');
            $table->dropIndex('idx_user_track_preferences_user');
            $table->dropConstrainedForeignId('user_id');
            $table->unique('track_id', 'user_track_preferences_track_id_unique');
        });

        Schema::table('rating_history', function (Blueprint $table) {
            $table->dropIndex('idx_rating_history_user');
            $table->dropConstrainedForeignId('user_id');
        });

        Schema::table('track_ratings', function (Blueprint $table) {
            $table->dropUnique('track_ratings_user_track_unique');
            $table->dropIndex('idx_track_ratings_user');
            $table->dropConstrainedForeignId('user_id');
            $table->unique('track_id', 'track_ratings_track_id_unique');
        });

        Schema::table('playlists', function (Blueprint $table) {
            $table->dropIndex('idx_playlists_user');
            $table->dropConstrainedForeignId('user_id');
        });
    }
};
