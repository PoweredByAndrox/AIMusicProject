<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/**
 * Create pattern_matches table for caching search results.
 */
return new class extends Migration
{
    public function up(): void
    {
        Schema::create('pattern_matches', function (Blueprint $table) {
            $table->id();
            $table->foreignId('pattern_id')->constrained()->onDelete('cascade');
            $table->foreignId('track_id')->constrained()->onDelete('cascade');
            
            $table->float('start_time');
            $table->float('end_time');
            $table->float('score'); // DTW distance or similarity
            
            // User feedback: null = not reviewed, true = confirmed, false = rejected
            $table->boolean('confirmed')->nullable();
            
            $table->timestamps();
            
            $table->index(['pattern_id', 'score']);
            $table->index(['track_id', 'pattern_id']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('pattern_matches');
    }
};
