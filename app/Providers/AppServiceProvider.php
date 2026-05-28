<?php

namespace App\Providers;

use App\Events\TrackWithFilePathCreated;
use App\Listeners\DispatchTrackWaveformPrewarm;
use Illuminate\Support\ServiceProvider;
use Illuminate\Support\Facades\Event;
use Illuminate\Support\Facades\URL;

class AppServiceProvider extends ServiceProvider
{
    /**
     * Register any application services.
     */
    public function register(): void
    {
        //
    }

    /**
     * Bootstrap any application services.
     */
    public function boot(): void
    {
        Event::listen(TrackWithFilePathCreated::class, DispatchTrackWaveformPrewarm::class);

        // Force HTTPS in production
        if ($this->app->environment('production')) {
            URL::forceScheme('https');
        }
    }
}
