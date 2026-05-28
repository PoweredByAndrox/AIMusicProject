<?php

namespace App\Events;

use App\Models\Track;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

class TrackWithFilePathCreated
{
    use Dispatchable, SerializesModels;

    public function __construct(
        public Track $track
    ) {}
}
