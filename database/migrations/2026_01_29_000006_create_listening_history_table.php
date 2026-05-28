<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the listening_history table for tracking playback history.
     * Requirements: 16.3 - Record track_id, timestamp, duration_played, completed
     */
    public function up(): void
    {
        Schema::create('listening_history', function (Blueprint $table) {
            $table->id();
            $table->foreignId('track_id')->constrained('tracks')->onDelete('cascade');
            $table->timestamp('started_at')->useCurrent();
            $table->float('duration_played')->nullable(); // seconds
            $table->boolean('completed')->default(false);
            $table->string('source', 50)->nullable(); // 'queue', 'playlist', 'radio', 'ai_dj'
            
            // Indexes for query optimization
            $table->index('track_id', 'idx_listening_history_track');
            $table->index('started_at', 'idx_listening_history_started_at');
            $table->index('source', 'idx_listening_history_source');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('listening_history');
    }
};
