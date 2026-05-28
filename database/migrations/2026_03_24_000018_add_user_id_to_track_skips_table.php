<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('track_skips', function (Blueprint $table) {
            $table->foreignId('user_id')->nullable()->constrained()->nullOnDelete();
            $table->index(['user_id', 'track_id'], 'idx_track_skips_user_track');
        });
    }

    public function down(): void
    {
        Schema::table('track_skips', function (Blueprint $table) {
            $table->dropIndex('idx_track_skips_user_track');
            $table->dropConstrainedForeignId('user_id');
        });
    }
};
