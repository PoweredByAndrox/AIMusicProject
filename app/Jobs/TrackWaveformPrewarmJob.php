<?php

namespace App\Jobs;

use App\Services\PublicRadioWaveformService;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldBeUnique;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use App\Models\Track;

class TrackWaveformPrewarmJob implements ShouldQueue, ShouldBeUnique
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public int $tries = 2;
    public int $backoff = 60;
    public int $uniqueFor = 1800;

    public function __construct(
        public int $trackId
    ) {
        $this->onQueue('radio-prewarm');
    }

    public function uniqueId(): string
    {
        return 'track-waveform-prewarm:' . $this->trackId;
    }

    public function handle(PublicRadioWaveformService $waveformService): void
    {
        $track = Track::find($this->trackId);
        if (!$track) {
            return;
        }

        $waveformService->ensureWaveformPackage($track);
    }
}
