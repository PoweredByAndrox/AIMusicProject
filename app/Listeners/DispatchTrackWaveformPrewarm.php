<?php

namespace App\Listeners;

use App\Events\TrackWithFilePathCreated;
use App\Jobs\TrackHlsPrewarmJob;
use App\Jobs\TrackWaveformPrewarmJob;

class DispatchTrackWaveformPrewarm
{
    public function handle(TrackWithFilePathCreated $event): void
    {
        TrackWaveformPrewarmJob::dispatch((int) $event->track->id);
        TrackHlsPrewarmJob::dispatch((int) $event->track->id);
    }
}
