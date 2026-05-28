<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the queue_tasks table for monitoring ML worker task processing.
     * Requirements: 4.1
     */
    public function up(): void
    {
        Schema::create('queue_tasks', function (Blueprint $table) {
            // Primary key
            $table->id();
            
            // Task details
            $table->string('task_type', 50); // 'analysis' | 'fragment_analysis' | 'training' | 'recommendation'
            $table->jsonb('payload');
            $table->string('status', 20)->default('pending'); // 'pending' | 'processing' | 'completed' | 'failed'
            $table->text('error_message')->nullable();
            
            // Timestamps
            $table->timestamp('created_at')->useCurrent();
            $table->timestamp('completed_at')->nullable();
            
            // Indexes for common queries
            $table->index('task_type', 'idx_queue_tasks_type');
            $table->index('status', 'idx_queue_tasks_status');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('queue_tasks');
    }
};
