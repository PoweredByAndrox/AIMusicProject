<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Bring pattern_examples schema in sync with runtime code:
     * - NULL is_positive means "unrated" (auto-generated segments)
     * - rating stores 0..5 stars
     * - embedding_path stores audio embedding for the segment
     */
    public function up(): void
    {
        Schema::table('pattern_examples', function (Blueprint $table) {
            if (!Schema::hasColumn('pattern_examples', 'rating')) {
                $table->float('rating')->nullable();
            }

            if (!Schema::hasColumn('pattern_examples', 'embedding_path')) {
                $table->string('embedding_path', 500)->nullable();
            }
        });

        // Allow NULL for "unrated" segments (worker inserts NULL).
        // Use raw SQL to avoid doctrine/dbal dependency for ->change().
        try {
            DB::statement('ALTER TABLE pattern_examples ALTER COLUMN is_positive DROP NOT NULL');
        } catch (\Throwable $e) {
            // If it's already nullable or DB differs, keep migration non-fatal.
        }

        // Add index for track_id if missing.
        // Naming is explicit to avoid collisions.
        try {
            DB::statement('CREATE INDEX IF NOT EXISTS idx_pattern_examples_track_id ON pattern_examples (track_id)');
        } catch (\Throwable $e) {
            // Best-effort: IF NOT EXISTS may be unsupported on some PG variants.
        }
    }

    public function down(): void
    {
        Schema::table('pattern_examples', function (Blueprint $table) {
            if (Schema::hasColumn('pattern_examples', 'embedding_path')) {
                $table->dropColumn('embedding_path');
            }
            if (Schema::hasColumn('pattern_examples', 'rating')) {
                $table->dropColumn('rating');
            }
        });

        // Restore NOT NULL for is_positive (best-effort).
        try {
            DB::statement('UPDATE pattern_examples SET is_positive = true WHERE is_positive IS NULL');
            DB::statement('ALTER TABLE pattern_examples ALTER COLUMN is_positive SET NOT NULL');
        } catch (\Throwable $e) {
            // Best-effort rollback.
        }

        try {
            DB::statement('DROP INDEX IF EXISTS idx_pattern_examples_track_id');
        } catch (\Throwable $e) {
            // Best-effort rollback.
        }
    }
};
