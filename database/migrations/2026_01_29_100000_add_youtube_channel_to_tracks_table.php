<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/**
 * Add youtube_channel field to tracks table.
 * 
 * Stores the YouTube channel name separately from artist,
 * allowing proper artist parsing from video titles.
 */
return new class extends Migration
{
    public function up(): void
    {
        Schema::table('tracks', function (Blueprint $table) {
            $table->string('youtube_channel')->nullable()->after('youtube_video_id');
        });
    }

    public function down(): void
    {
        Schema::table('tracks', function (Blueprint $table) {
            $table->dropColumn('youtube_channel');
        });
    }
};
