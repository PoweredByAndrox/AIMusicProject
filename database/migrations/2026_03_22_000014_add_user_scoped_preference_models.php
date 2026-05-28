<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('preference_model_status', function (Blueprint $table) {
            $table->foreignId('user_id')
                ->nullable()
                ->after('id')
                ->constrained()
                ->cascadeOnDelete();

            $table->unique(['user_id'], 'preference_model_status_user_unique');
        });

        Schema::table('preference_model_training_runs', function (Blueprint $table) {
            $table->foreignId('user_id')
                ->nullable()
                ->after('queue_task_id')
                ->constrained()
                ->cascadeOnDelete();

            $table->index(['user_id'], 'idx_pmt_runs_user_id');
        });

        Schema::create('user_track_scores', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->cascadeOnDelete();
            $table->foreignId('track_id')->constrained()->cascadeOnDelete();
            $table->double('preference_score');
            $table->timestamp('score_calculated_at')->nullable();
            $table->timestamps();

            $table->unique(['user_id', 'track_id'], 'user_track_scores_user_track_unique');
            $table->index(['user_id', 'preference_score'], 'idx_user_track_scores_user_score');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('user_track_scores');

        Schema::table('preference_model_training_runs', function (Blueprint $table) {
            $table->dropIndex('idx_pmt_runs_user_id');
            $table->dropConstrainedForeignId('user_id');
        });

        Schema::table('preference_model_status', function (Blueprint $table) {
            $table->dropUnique('preference_model_status_user_unique');
            $table->dropConstrainedForeignId('user_id');
        });
    }
};
