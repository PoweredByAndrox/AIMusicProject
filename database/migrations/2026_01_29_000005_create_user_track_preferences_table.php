<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the user_track_preferences table for per-track user settings.
     * Requirements: 4.4 - Store preferences in database with track_id binding
     */
    public function up(): void
    {
        Schema::create('user_track_preferences', function (Blueprint $table) {
            $table->id();
            $table->foreignId('track_id')->unique()->constrained('tracks')->onDelete('cascade');
            $table->float('volume')->default(1.0);
            $table->float('last_position')->default(0);
            $table->timestamp('updated_at')->useCurrent();
            
            // Indexes for query optimization
            $table->index('track_id', 'idx_user_track_preferences_track');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('user_track_preferences');
    }
};
