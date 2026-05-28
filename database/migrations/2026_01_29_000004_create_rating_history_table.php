<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the rating_history table for tracking rating changes over time.
     * Requirements: 14.1 - Record rating changes with old_rating, new_rating, timestamp
     */
    public function up(): void
    {
        Schema::create('rating_history', function (Blueprint $table) {
            $table->id();
            $table->foreignId('track_id')->nullable()->constrained('tracks')->onDelete('set null');
            $table->smallInteger('old_rating')->nullable();
            $table->smallInteger('new_rating')->nullable();
            $table->timestamp('changed_at')->useCurrent();
            
            // Indexes for query optimization
            $table->index('track_id', 'idx_rating_history_track');
            $table->index('changed_at', 'idx_rating_history_changed_at');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('rating_history');
    }
};
