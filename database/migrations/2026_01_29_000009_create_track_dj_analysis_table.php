<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the track_dj_analysis table for AI DJ analysis results.
     * Requirements: 27.5 - Save analysis results to database for quick access
     */
    public function up(): void
    {
        Schema::create('track_dj_analysis', function (Blueprint $table) {
            $table->id();
            $table->foreignId('track_id')->unique()->constrained('tracks')->onDelete('cascade');
            $table->string('camelot_key', 3)->nullable(); // e.g., '8A', '11B'
            $table->float('bpm')->nullable();
            $table->smallInteger('energy_level')->nullable(); // 1-10
            $table->float('intro_point')->nullable(); // seconds
            $table->float('outro_point')->nullable(); // seconds
            $table->jsonb('drop_points')->default('[]');
            $table->jsonb('breakdown_points')->default('[]');
            $table->timestamp('analyzed_at')->useCurrent();
            
            // Indexes for query optimization
            $table->index('camelot_key', 'idx_track_dj_analysis_key');
            $table->index('bpm', 'idx_track_dj_analysis_bpm');
            $table->index('energy_level', 'idx_track_dj_analysis_energy');
        });
        
        // Add check constraint for energy_level range (PostgreSQL)
        DB::statement('ALTER TABLE track_dj_analysis ADD CONSTRAINT chk_energy_level_range CHECK (energy_level IS NULL OR (energy_level >= 1 AND energy_level <= 10))');
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('track_dj_analysis');
    }
};
