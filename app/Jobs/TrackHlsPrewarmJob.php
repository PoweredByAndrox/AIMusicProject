<?php

namespace App\Jobs;

use App\Models\Track;
use App\Services\PublicRadioHlsService;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldBeUnique;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;

class TrackHlsPrewarmJob implements ShouldQueue, ShouldBeUnique
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public int $tries = 2;
    public int $backoff = 60;
    public int $uniqueFor = 1800;

    public function __construct(
        public int $trackId
    ) {
        $this->onQueue('radio-hls-prewarm');
    }

    public function uniqueId(): string
    {
        return 'track-hls-prewarm:' . $this->trackId;
    }

    public function handle(PublicRadioHlsService $hlsService): void
    {
        $track = Track::find($this->trackId);
        if (!$track) {
            return;
        }

        $hlsService->ensureHlsPackage($track);
    }
}
