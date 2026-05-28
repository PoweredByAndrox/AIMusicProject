<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Stores per-run training summary metrics for debugging/regression.
     */
    public function up(): void
    {
        Schema::create('preference_model_training_runs', function (Blueprint $table) {
            $table->id();

            $table->unsignedBigInteger('queue_task_id')->nullable();
            $table->string('model_path', 500)->nullable();

            $table->timestamp('started_at')->nullable();
            $table->timestamp('finished_at')->nullable();

            $table->float('final_loss')->nullable();

            $table->integer('positives_total')->default(0);
            $table->integer('positives_tracks')->default(0);
            $table->integer('positives_fragments')->default(0);
            $table->integer('positives_segments')->default(0);

            $table->integer('explicit_negatives_total')->default(0);
            $table->integer('explicit_negative_fragments')->default(0);
            $table->integer('explicit_negative_segments')->default(0);

            $table->integer('all_tracks_pool')->default(0);
            $table->integer('tracks_scored')->default(0);
            $table->integer('tracks_with_segments')->default(0);
            $table->integer('segment_improved')->default(0);

            $table->integer('segments_rated_total')->default(0);
            $table->integer('segments_rated_with_embedding')->default(0);

            $table->jsonb('meta')->nullable();

            $table->timestamps();

            $table->index('queue_task_id', 'idx_pmt_runs_queue_task_id');
            $table->index('created_at', 'idx_pmt_runs_created_at');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('preference_model_training_runs');
    }
};

