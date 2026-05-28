<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('tracks', function (Blueprint $table) {
            $table->float('loudness_integrated_lufs')->nullable()->after('is_acoustic');
            $table->float('loudness_true_peak_db')->nullable()->after('loudness_integrated_lufs');
            $table->float('loudness_gain_db')->nullable()->after('loudness_true_peak_db');
            $table->float('loudness_target_lufs')->nullable()->after('loudness_gain_db');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('tracks', function (Blueprint $table) {
            $table->dropColumn([
                'loudness_integrated_lufs',
                'loudness_true_peak_db',
                'loudness_gain_db',
                'loudness_target_lufs',
            ]);
        });
    }
};
