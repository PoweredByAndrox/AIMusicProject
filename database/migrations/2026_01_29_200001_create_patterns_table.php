<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/**
 * Create patterns table for melodic pattern storage.
 * 
 * Patterns are user-defined melodic concepts that can be searched
 * across the music library.
 */
return new class extends Migration
{
    public function up(): void
    {
        Schema::create('patterns', function (Blueprint $table) {
            $table->id();
            $table->string('name');
            $table->text('description')->nullable();
            
            // Type: pitch_contour, chroma, midi
            $table->string('representation_type')->default('chroma');
            
            // Settings
            $table->boolean('transpose_invariant')->default(true);
            $table->boolean('tempo_invariant')->default(true);
            $table->float('match_threshold')->default(0.7);
            $table->integer('importance')->default(1); // 0-3
            
            // Prototype vector (averaged from positive examples)
            $table->string('prototype_path')->nullable();
            
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('patterns');
    }
};
