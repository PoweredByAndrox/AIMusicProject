<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the tracks table for storing audio track metadata and ML characteristics.
     * Requirements: 4.1, 4.2
     */
    public function up(): void
    {
        Schema::create('tracks', function (Blueprint $table) {
            // Primary key
            $table->id();
            
            // Basic metadata
            $table->string('title', 255);
            $table->string('artist', 255)->nullable();
            $table->string('album', 255)->nullable();
            $table->integer('year')->nullable();
            $table->float('duration');
            $table->string('file_path', 500);
            $table->string('source', 50); // 'upload' | 'youtube'
            $table->string('youtube_url', 500)->nullable();
            
            // ML characteristics
            $table->string('genre', 100)->nullable();
            $table->float('bpm')->nullable();
            $table->string('key', 10)->nullable();
            $table->string('mode', 10)->nullable(); // 'major' | 'minor'
            $table->string('mood', 50)->nullable();
            $table->boolean('has_vocals')->nullable();
            $table->boolean('is_acoustic')->nullable();
            
            // Flags
            $table->string('analysis_status', 20)->default('pending'); // 'pending' | 'processing' | 'completed' | 'failed'
            $table->boolean('is_liked')->default(false);
            $table->jsonb('manually_edited')->default('{}'); // fields edited manually
            
            // Embedding
            $table->string('embedding_path', 500)->nullable();
            
            // Timestamps
            $table->timestamp('created_at')->useCurrent();
            $table->timestamp('updated_at')->useCurrent();
            
            // Indexes for query optimization
            $table->index('genre', 'idx_tracks_genre');
            $table->index('mood', 'idx_tracks_mood');
            $table->index('bpm', 'idx_tracks_bpm');
            $table->index('is_liked', 'idx_tracks_liked');
            $table->index('analysis_status', 'idx_tracks_analysis_status');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('tracks');
    }
};
