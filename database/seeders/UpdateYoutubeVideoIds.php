<?php

namespace Database\Seeders;

use App\Models\Track;
use Illuminate\Database\Seeder;

class UpdateYoutubeVideoIds extends Seeder
{
    public function run(): void
    {
        $tracks = Track::whereNotNull('youtube_url')
            ->whereNull('youtube_video_id')
            ->get();

        $updated = 0;
        foreach ($tracks as $track) {
            if (preg_match('/(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([a-zA-Z0-9_-]{11})/', $track->youtube_url, $matches)) {
                $track->youtube_video_id = $matches[1];
                $track->save();
                $this->command->info("Updated track {$track->id}: {$matches[1]}");
                $updated++;
            }
        }

        $this->command->info("Total updated: {$updated}");
    }
}
