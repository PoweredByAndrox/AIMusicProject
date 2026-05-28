<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('tracks', function (Blueprint $table) {
            $table->jsonb('artist_credits')->nullable()->after('artist');
            $table->jsonb('artist_slugs')->nullable()->after('artist_credits');
        });
    }

    public function down(): void
    {
        Schema::table('tracks', function (Blueprint $table) {
            $table->dropColumn(['artist_credits', 'artist_slugs']);
        });
    }
};
