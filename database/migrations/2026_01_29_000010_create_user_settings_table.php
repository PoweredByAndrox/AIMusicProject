<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Creates the user_settings table for storing user preferences (theme, animations, etc.).
     * Requirements: 21.6 - Save theme choice in User_Preference
     */
    public function up(): void
    {
        Schema::create('user_settings', function (Blueprint $table) {
            $table->id();
            $table->string('key', 100)->unique();
            $table->jsonb('value');
            $table->timestamp('updated_at')->useCurrent();
            
            // Index on key for fast lookups
            $table->index('key', 'idx_user_settings_key');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('user_settings');
    }
};
