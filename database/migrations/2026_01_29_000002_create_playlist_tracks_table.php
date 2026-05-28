<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the playlist_tracks pivot table for playlist-track relationships.
     * Requirements: 5.2 - Adding tracks to playlists with position ordering
     */
    public function up(): void
    {
        Schema::create('playlist_tracks', function (Blueprint $table) {
            $table->id();
            $table->foreignId('playlist_id')->constrained('playlists')->onDelete('cascade');
            $table->foreignId('track_id')->constrained('tracks')->onDelete('cascade');
            $table->integer('position');
            $table->timestamp('added_at')->useCurrent();
            
            // Unique constraint: each track can only appear once per playlist
            $table->unique(['playlist_id', 'track_id'], 'uq_playlist_tracks');
            
            // Indexes for query optimization
            $table->index('playlist_id', 'idx_playlist_tracks_playlist');
            $table->index('track_id', 'idx_playlist_tracks_track');
            $table->index('position', 'idx_playlist_tracks_position');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('playlist_tracks');
    }
};
