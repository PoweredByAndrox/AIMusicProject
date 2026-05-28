<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the radio_feedback table for Smart Radio "more/less" feedback.
     * Requirements: 17.4 - Save feedback for Preference_Model training
     */
    public function up(): void
    {
        Schema::create('radio_feedback', function (Blueprint $table) {
            $table->id();
            $table->foreignId('track_id')->constrained('tracks')->onDelete('cascade');
            $table->string('feedback_type', 10); // 'more' or 'less'
            $table->timestamp('created_at')->useCurrent();
            
            // Indexes for query optimization
            $table->index('track_id', 'idx_radio_feedback_track');
            $table->index('feedback_type', 'idx_radio_feedback_type');
            $table->index('created_at', 'idx_radio_feedback_created_at');
        });
        
        // Add check constraint for feedback_type (PostgreSQL)
        DB::statement("ALTER TABLE radio_feedback ADD CONSTRAINT chk_feedback_type CHECK (feedback_type IN ('more', 'less'))");
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('radio_feedback');
    }
};
