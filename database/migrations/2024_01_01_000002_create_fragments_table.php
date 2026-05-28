<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\DB;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the fragments table for storing marked audio segments within tracks.
     * Requirements: 4.3
     */
    public function up(): void
    {
        Schema::create('fragments', function (Blueprint $table) {
            // Primary key
            $table->id();
            
            // Foreign key to tracks
            $table->foreignId('track_id')
                ->constrained('tracks')
                ->onDelete('cascade');
            
            // Time range
            $table->float('start_time');
            $table->float('end_time');
            
            // Flags
            $table->boolean('is_liked')->default(false);
            $table->string('embedding_path', 500)->nullable();
            $table->string('analysis_status', 20)->default('pending'); // 'pending' | 'processing' | 'completed' | 'failed'
            
            // Timestamps
            $table->timestamp('created_at')->useCurrent();
            
            // Indexes
            $table->index('track_id', 'idx_fragments_track_id');
            $table->index('is_liked', 'idx_fragments_liked');
        });
        
        // Add CHECK constraint for valid time range (end_time > start_time)
        DB::statement('ALTER TABLE fragments ADD CONSTRAINT valid_time_range CHECK (end_time > start_time)');
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('fragments');
    }
};
