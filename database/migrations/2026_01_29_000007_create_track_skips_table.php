<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the track_skips table for tracking skipped tracks.
     * Requirements: 18.2 - Store track_id, timestamp, position_at_skip, source
     */
    public function up(): void
    {
        Schema::create('track_skips', function (Blueprint $table) {
            $table->id();
            $table->foreignId('track_id')->constrained('tracks')->onDelete('cascade');
            $table->float('position_at_skip')->nullable();
            $table->string('source', 50)->nullable(); // 'queue', 'playlist', 'radio', 'ai_dj'
            $table->timestamp('skipped_at')->useCurrent();
            
            // Indexes for query optimization
            $table->index('track_id', 'idx_track_skips_track');
            $table->index('skipped_at', 'idx_track_skips_skipped_at');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('track_skips');
    }
};
