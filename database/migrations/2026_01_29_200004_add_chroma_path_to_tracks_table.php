<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/**
 * Add chroma and pitch contour paths to tracks table.
 */
return new class extends Migration
{
    public function up(): void
    {
        Schema::table('tracks', function (Blueprint $table) {
            $table->string('chroma_path')->nullable()->after('embedding_path');
            $table->string('pitch_contour_path')->nullable()->after('chroma_path');
        });
    }

    public function down(): void
    {
        Schema::table('tracks', function (Blueprint $table) {
            $table->dropColumn(['chroma_path', 'pitch_contour_path']);
        });
    }
};
