<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\DB;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Adds comment and rated_at fields to fragments table.
     * Changes is_liked to nullable (null = not rated, true = liked, false = disliked).
     * Requirements: 7.1, 8.1
     */
    public function up(): void
    {
        Schema::table('fragments', function (Blueprint $table) {
            // Add comment field for fragment comments (up to 500 characters)
            // Requirements: 8.1, 8.2
            $table->text('comment')->nullable()->after('is_liked');
            
            // Add rated_at timestamp to track when fragment was rated
            // Requirements: 7.1
            $table->timestamp('rated_at')->nullable()->after('comment');
        });

        // Change is_liked to nullable using raw SQL for PostgreSQL compatibility
        // null = not rated, true = liked, false = disliked
        // Requirements: 7.1
        DB::statement('ALTER TABLE fragments ALTER COLUMN is_liked DROP DEFAULT');
        DB::statement('ALTER TABLE fragments ALTER COLUMN is_liked DROP NOT NULL');
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        // Restore is_liked to NOT NULL with default false
        DB::statement('UPDATE fragments SET is_liked = false WHERE is_liked IS NULL');
        DB::statement('ALTER TABLE fragments ALTER COLUMN is_liked SET NOT NULL');
        DB::statement('ALTER TABLE fragments ALTER COLUMN is_liked SET DEFAULT false');

        Schema::table('fragments', function (Blueprint $table) {
            $table->dropColumn(['comment', 'rated_at']);
        });
    }
};
