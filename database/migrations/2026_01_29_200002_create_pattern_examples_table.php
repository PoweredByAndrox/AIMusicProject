<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/**
 * Create pattern_examples table for storing positive/negative examples.
 */
return new class extends Migration
{
    public function up(): void
    {
        Schema::create('pattern_examples', function (Blueprint $table) {
            $table->id();
            $table->foreignId('pattern_id')->constrained()->onDelete('cascade');
            $table->foreignId('track_id')->constrained()->onDelete('cascade');
            
            $table->float('start_time');
            $table->float('end_time');
            
            // positive = true, negative = false
            $table->boolean('is_positive')->default(true);
            
            $table->text('notes')->nullable();
            
            // Extracted representation path
            $table->string('representation_path')->nullable();
            
            $table->timestamps();
            
            $table->index(['pattern_id', 'is_positive']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('pattern_examples');
    }
};
