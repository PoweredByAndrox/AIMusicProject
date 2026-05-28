<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the preference_model_status table for tracking the personal preference model state.
     * Requirements: 4.1
     */
    public function up(): void
    {
        Schema::create('preference_model_status', function (Blueprint $table) {
            // Primary key
            $table->id();
            
            // Model status
            $table->string('status', 20)->default('not_trained'); // 'not_trained' | 'training' | 'trained'
            $table->string('model_path', 500)->nullable();
            $table->timestamp('trained_at')->nullable();
            $table->integer('likes_count')->default(0);
            
            // Timestamps
            $table->timestamp('created_at')->useCurrent();
            $table->timestamp('updated_at')->useCurrent();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('preference_model_status');
    }
};
