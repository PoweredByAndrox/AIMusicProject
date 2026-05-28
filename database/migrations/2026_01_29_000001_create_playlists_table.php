<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the playlists table for storing user and AI-generated playlists.
     * Requirements: 5.1 - Playlist creation with name and description
     */
    public function up(): void
    {
        Schema::create('playlists', function (Blueprint $table) {
            $table->id();
            $table->string('name', 255);
            $table->text('description')->nullable();
            $table->boolean('is_ai_generated')->default(false);
            $table->integer('track_count')->default(0);
            $table->integer('total_duration')->default(0); // seconds
            $table->timestamp('created_at')->useCurrent();
            $table->timestamp('updated_at')->useCurrent();
            
            // Indexes for query optimization
            $table->index('is_ai_generated', 'idx_playlists_ai_generated');
            $table->index('created_at', 'idx_playlists_created_at');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('playlists');
    }
};
