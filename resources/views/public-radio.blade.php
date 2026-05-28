<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="csrf-token" content="{{ csrf_token() }}">

    <title>Public Radio</title>
    @php
        $manifestPath = public_path('build/.vite/manifest.json');
        $manifest = file_exists($manifestPath)
            ? json_decode(file_get_contents($manifestPath), true)
            : [];
        $radioEntry = $manifest['resources/js/public-radio.ts'] ?? null;
        $radioCss = $radioEntry['css'] ?? [];
    @endphp

    @foreach ($radioCss as $cssFile)
        <link rel="stylesheet" href="{{ '/build/' . ltrim($cssFile, '/') }}">
    @endforeach
</head>
<body
    class="{{ $embed ? 'bg-transparent' : 'min-h-screen bg-[radial-gradient(circle_at_top,_rgba(125,211,252,0.28),_transparent_42%),linear-gradient(180deg,#eff6ff_0%,#f8fafc_48%,#e2e8f0_100%)]' }}"
    data-radio-slug="{{ $slug }}"
    data-radio-embed="{{ $embed ? '1' : '0' }}"
>
    <div class="{{ $embed ? 'mx-auto max-w-3xl p-4' : 'w-full p-4 md:p-8' }}">
        <section class="{{ $embed ? 'surface-card !rounded-[1.8rem] !p-5' : 'surface-card overflow-hidden !rounded-[2rem] !p-0' }}">
            <div class="{{ $embed ? 'flex flex-col gap-5' : 'border-b border-slate-200/80 bg-slate-50/80 px-6 py-6 md:px-8' }}">
                <div class="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                    <div>
                        <p class="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Public Radio</p>
                        <h1 id="radio-station-name" class="mt-2 text-3xl font-semibold text-slate-950">Loading station</h1>
                        <p id="radio-station-description" class="mt-3 max-w-2xl text-sm leading-6 text-slate-600">
                            Shared station link backed by server-side schedule state.
                        </p>
                    </div>

                    @if (!$embed)
                        <div class="grid gap-3 text-sm text-slate-600">
                            <div class="rounded-[1.2rem] border border-slate-200 bg-white/80 px-4 py-3">
                                <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Public URL</div>
                                <div id="radio-public-url" class="mt-2 break-all font-medium text-slate-950">Loading...</div>
                            </div>
                            <div class="rounded-[1.2rem] border border-slate-200 bg-white/80 px-4 py-3">
                                <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Embed URL</div>
                                <div id="radio-embed-url" class="mt-2 break-all font-medium text-slate-950">Loading...</div>
                            </div>
                        </div>
                    @endif
                </div>

                @if (!$embed)
                    <div class="mt-5">
                        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Stations</div>
                        <div id="radio-stations" class="mt-3 flex flex-wrap gap-3">
                            <span class="badge-soft">Loading stations...</span>
                        </div>

                        <section id="radio-wave-genres-shell" class="mt-4 hidden rounded-[1.3rem] border border-slate-200 bg-white/80 px-4 py-4">
                            <div class="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                                <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Your Wave Genres</div>
                                <div id="radio-wave-genres-status" class="text-xs font-medium text-slate-400">
                                    Choose genres for your recommendation-driven wave.
                                </div>
                            </div>

                            <div id="radio-wave-genres-panel" class="mt-3">
                                <p class="text-sm text-slate-500">Genre controls will appear for your personal station.</p>
                            </div>
                        </section>
                    </div>
                @endif
            </div>

            <div class="{{ $embed ? 'grid gap-5' : 'grid gap-6 px-6 py-6 md:grid-cols-[minmax(0,1.4fr)_minmax(22rem,0.9fr)] md:px-8 md:py-8' }}">
                <div class="grid gap-5">
                    <div class="rounded-[1.5rem] border border-slate-200 bg-white/85 p-5 shadow-[0_20px_60px_rgba(15,23,42,0.08)]">
                        <div class="flex items-center justify-between gap-4">
                            <div>
                                <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Now Playing</div>
                                <div id="radio-now-title" class="mt-2 text-2xl font-semibold text-slate-950">Loading station state...</div>
                                <div id="radio-now-artist" class="mt-2 text-sm text-slate-600">Connecting to radio snapshot...</div>
                            </div>

                            <div class="flex flex-wrap items-start justify-end gap-3">
                                <div class="rounded-[1.2rem] bg-emerald-50 px-4 py-3 text-right">
                                    <div class="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">Listening Now</div>
                                    <div id="radio-listener-count" class="mt-2 text-lg font-semibold text-emerald-900">0</div>
                                </div>
                            </div>
                        </div>

                        <div id="radio-waveform-shell" class="mt-5 rounded-[1.35rem] border border-slate-200 bg-slate-950 px-4 py-4 shadow-[0_18px_45px_rgba(15,23,42,0.18)]">
                            <div class="flex items-center justify-between gap-3">
                                <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-300">Waveform</div>
                                <div id="radio-waveform-status" class="text-xs font-medium text-slate-400">Loading waveform...</div>
                            </div>
                            <div id="radio-waveform" class="mt-4 overflow-hidden rounded-[1.1rem] border border-white/10 bg-slate-900/90 px-2 py-2"></div>
                        </div>

                        <audio id="radio-audio" class="hidden" preload="auto" autoplay playsinline aria-hidden="true"></audio>

                        <div class="mt-5 rounded-[1.25rem] border border-slate-200 bg-slate-50/90 px-4 py-4">
                            <div class="flex flex-wrap items-center gap-3">
                                <button id="radio-play-toggle" type="button" class="action-primary !min-w-[8.75rem] !justify-center !rounded-xl !px-4 !py-2">
                                    Start radio
                                </button>
                                <button id="radio-mute-toggle" type="button" class="action-secondary !min-w-[8.75rem] !justify-center !rounded-xl !px-3 !py-2">
                                    Mute
                                </button>

                                <label for="radio-volume" class="w-16 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                                    Volume
                                </label>
                                <input
                                    id="radio-volume"
                                    type="range"
                                    min="0"
                                    max="1"
                                    step="0.01"
                                    value="1"
                                    class="h-2 w-40 cursor-pointer accent-sky-600"
                                >

                                <label for="radio-bitrate" class="w-16 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                                    Bitrate
                                </label>
                                <select
                                    id="radio-bitrate"
                                    class="w-36 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm outline-none transition focus:border-sky-300 focus:ring-2 focus:ring-sky-100"
                                >
                                    <option value="auto">Auto</option>
                                    <option value="32">32 kbps</option>
                                    <option value="48">48 kbps</option>
                                    <option value="64">64 kbps</option>
                                    <option value="96">96 kbps</option>
                                    <option value="128">128 kbps</option>
                                    <option value="160">160 kbps</option>
                                    <option value="192">192 kbps</option>
                                    <option value="256">256 kbps</option>
                                    <option value="320">320 kbps</option>
                                </select>

                                <label for="radio-transport" class="w-20 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                                    Transport
                                </label>
                                <select
                                    id="radio-transport"
                                    class="w-32 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm outline-none transition focus:border-sky-300 focus:ring-2 focus:ring-sky-100"
                                >
                                    <option value="direct">Direct</option>
                                    <option value="hls">HLS</option>
                                </select>

                                <div class="ml-auto text-sm font-medium text-slate-500">
                                    <span id="radio-time">0:00 / 0:00</span>
                                </div>
                            </div>
                        </div>

                        <div class="mt-4 flex flex-wrap items-center gap-3 text-sm">
                            <span id="radio-remaining" class="badge-soft">Loading station state...</span>
                            <button
                                id="radio-direct-share-track"
                                type="button"
                                class="hidden rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-600 transition hover:border-sky-200 hover:text-sky-700"
                            >
                                Send track to friend
                            </button>
                        </div>
                    </div>

                    <div id="radio-error" class="hidden rounded-[1.5rem] border border-rose-200 bg-rose-50/80 px-5 py-4 text-sm font-medium text-rose-700"></div>
                    <div id="radio-loading" class="rounded-[1.5rem] border border-slate-200 bg-white/80 px-5 py-4 text-sm text-slate-600">
                        Loading shared station state...
                    </div>
                </div>

                @if (!$embed)
                    <div class="grid gap-5">
                        <section class="rounded-[1.5rem] border border-slate-200 bg-white/85 p-5">
                            <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Your Feedback</div>
                            <div id="radio-feedback-auth" class="mt-4 text-sm text-slate-500">
                                Sign in to rate the current track and its segments.
                            </div>

                            <div id="radio-feedback-panel" class="mt-4 hidden">
                                <div class="rounded-[1.15rem] border border-sky-200 bg-sky-50/80 px-4 py-4">
                                    <div class="text-xs font-semibold uppercase tracking-[0.16em] text-sky-700">Your Match</div>
                                    <div id="radio-track-fit-value" class="mt-2 text-base font-semibold text-sky-950">No personal fit score yet</div>
                                </div>

                                <div class="rounded-[1.15rem] border border-slate-200 bg-slate-50/85 px-4 py-4">
                                    <div class="flex items-center justify-between gap-3">
                                        <div>
                                            <div class="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Track Rating</div>
                                            <div id="radio-track-rating-value" class="mt-2 text-sm font-medium text-slate-700">Not rated yet</div>
                                        </div>
                                        <div id="radio-track-stars" class="flex items-center gap-1"></div>
                                    </div>
                                </div>

                                <div class="mt-4">
                                    <div class="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Track Segments</div>
                                    <div class="mt-2 text-xs text-slate-400">
                                        Radio mode keeps segment ratings, but seeking and jumping are disabled.
                                    </div>
                                    <div id="radio-segments-status" class="mt-3 text-sm text-slate-500">Loading segments...</div>
                                    <div id="radio-segments" class="mt-3 grid gap-3"></div>
                                </div>
                            </div>
                        </section>

                        <section class="rounded-[1.5rem] border border-slate-200 bg-white/85 p-5">
                            <div class="flex items-center justify-between gap-3">
                                <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Station Chat</div>
                                <div id="radio-chat-status" class="text-xs font-medium text-slate-400">Sign in to join the chat.</div>
                            </div>

                            <div id="radio-chat-auth" class="mt-4 text-sm text-slate-500">
                                Sign in to chat and share segments with other listeners.
                            </div>

                            <div id="radio-chat-panel" class="mt-4 hidden">
                                <div id="radio-chat-list" class="max-h-80 space-y-3 overflow-y-auto rounded-[1.15rem] border border-slate-200 bg-slate-50/80 px-3 py-3">
                                    <p class="text-sm text-slate-500">No chat messages yet.</p>
                                </div>

                                <form id="radio-chat-form" class="mt-4 flex gap-3">
                                    <input
                                        id="radio-chat-input"
                                        type="text"
                                        maxlength="400"
                                        autocomplete="off"
                                        placeholder="Write a message about what is playing..."
                                        class="flex-1 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700 shadow-sm outline-none transition focus:border-sky-300 focus:ring-2 focus:ring-sky-100"
                                    >
                                    <button id="radio-chat-submit" type="submit" class="action-primary !rounded-xl !px-4 !py-3 !text-sm">
                                        Send
                                    </button>
                                </form>
                            </div>
                        </section>

                        <section class="rounded-[1.5rem] border border-slate-200 bg-white/85 p-5">
                            <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Up Next</div>
                            <div id="radio-up-next" class="mt-4 grid gap-3">
                                <p class="text-sm text-slate-500">No tracks queued yet.</p>
                            </div>
                        </section>

                        <section class="rounded-[1.5rem] border border-slate-200 bg-white/85 p-5">
                            <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Recent History</div>
                            <div id="radio-history" class="mt-4 grid gap-3">
                                <p class="text-sm text-slate-500">History will appear after the station advances.</p>
                            </div>
                        </section>
                    </div>
                @endif
            </div>
        </section>
    </div>
    @if ($radioEntry)
        <script type="module" src="{{ '/build/' . ltrim($radioEntry['file'], '/') }}"></script>
    @endif
</body>
</html>
