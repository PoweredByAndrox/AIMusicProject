<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the track_ratings table for 5-star track ratings.
     * Requirements: 11.1 - 5-star rating system (1-5 stars)
     */
    public function up(): void
    {
        Schema::create('track_ratings', function (Blueprint $table) {
            $table->id();
            $table->foreignId('track_id')->unique()->constrained('tracks')->onDelete('cascade');
            $table->smallInteger('rating')->nullable(); // 1-5, nullable for unrated
            $table->timestamp('created_at')->useCurrent();
            $table->timestamp('updated_at')->useCurrent();
            
            // Indexes for query optimization
            $table->index('rating', 'idx_track_ratings_rating');
        });
        
        // Add check constraint for rating range (PostgreSQL)
        DB::statement('ALTER TABLE track_ratings ADD CONSTRAINT chk_rating_range CHECK (rating IS NULL OR (rating >= 1 AND rating <= 5))');
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('track_ratings');
    }
};
