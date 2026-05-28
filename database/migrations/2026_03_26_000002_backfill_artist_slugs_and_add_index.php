<?php

use App\Services\ArtistCreditParser;
use Illuminate\Database\Migrations\Migration;
use Illuminate\Support\Facades\DB;

return new class extends Migration
{
    public function up(): void
    {
        $parser = app(ArtistCreditParser::class);

        DB::table('tracks')
            ->select(['id', 'title', 'artist', 'artist_credits', 'artist_slugs', 'source', 'youtube_channel', 'file_path'])
            ->orderBy('id')
            ->chunkById(200, function ($tracks) use ($parser): void {
                foreach ($tracks as $track) {
                    $parsed = $parser->resolveFromAttributes((array) $track);

                    DB::table('tracks')
                        ->where('id', $track->id)
                        ->update([
                            'title' => $parsed['title'],
                            'artist' => $parsed['artist'],
                            'artist_credits' => json_encode($parsed['artist_credits'], JSON_UNESCAPED_UNICODE),
                            'artist_slugs' => json_encode($parsed['artist_slugs'], JSON_UNESCAPED_UNICODE),
                            'updated_at' => now(),
                        ]);
                }
            }, 'id');

        if (DB::getDriverName() === 'pgsql') {
            DB::statement('CREATE INDEX IF NOT EXISTS idx_tracks_artist_slugs_gin ON tracks USING gin (artist_slugs)');
        }
    }

    public function down(): void
    {
        if (DB::getDriverName() === 'pgsql') {
            DB::statement('DROP INDEX IF EXISTS idx_tracks_artist_slugs_gin');
        }
    }
};
