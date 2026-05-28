<?php

namespace App\Http\Controllers;

use App\Models\Track;
use Illuminate\Http\Request;
use Illuminate\Http\Response;
use Symfony\Component\HttpFoundation\StreamedResponse;

/**
 * ExportController handles data export functionality.
 * 
 * Requirements: 12.1, 12.2, 12.3, 12.4
 */
class ExportController extends Controller
{
    /**
     * CSV column headers for track export.
     */
    private const CSV_HEADERS = [
        'id',
        'title',
        'artist',
        'album',
        'year',
        'duration',
        'genre',
        'bpm',
        'key',
        'mood',
        'has_vocals',
        'is_acoustic',
        'is_liked',
        'source',
        'youtube_url',
        'created_at',
        'updated_at',
    ];

    /**
     * Export library to CSV format.
     * 
     * Uses streaming response for large libraries to prevent memory issues.
     * 
     * Requirements: 12.1, 12.2, 12.3, 12.4
     *
     * @param Request $request
     * @return StreamedResponse
     */
    public function exportCsv(Request $request): StreamedResponse
    {
        $filename = 'music_library_' . date('Y-m-d_His') . '.csv';

        return new StreamedResponse(function () use ($request) {
            $handle = fopen('php://output', 'w');

            // Write UTF-8 BOM for Excel compatibility
            fwrite($handle, "\xEF\xBB\xBF");

            // Write headers
            fputcsv($handle, self::CSV_HEADERS);

            // Build query with optional filters
            $query = Track::query();

            if ($request->has('genre')) {
                $query->byGenre($request->input('genre'));
            }

            if ($request->has('mood')) {
                $query->byMood($request->input('mood'));
            }

            if ($request->has('liked') && $request->input('liked') === 'true') {
                $query->liked();
            }

            if ($request->has('bpm_min') && $request->has('bpm_max')) {
                $query->bpmRange(
                    (float) $request->input('bpm_min'),
                    (float) $request->input('bpm_max')
                );
            }

            // Stream tracks in chunks to handle large libraries
            $query->orderBy('id')->chunk(100, function ($tracks) use ($handle) {
                foreach ($tracks as $track) {
                    fputcsv($handle, $this->trackToRow($track));
                }
            });

            fclose($handle);
        }, 200, [
            'Content-Type' => 'text/csv; charset=UTF-8',
            'Content-Disposition' => "attachment; filename=\"{$filename}\"",
            'Cache-Control' => 'no-cache, no-store, must-revalidate',
            'Pragma' => 'no-cache',
            'Expires' => '0',
        ]);
    }

    /**
     * Convert a Track model to a CSV row array.
     *
     * @param Track $track
     * @return array
     */
    private function trackToRow(Track $track): array
    {
        return [
            $track->id,
            $track->title,
            $track->artist,
            $track->album,
            $track->year,
            $track->duration,
            $track->genre,
            $track->bpm,
            $track->key,
            $track->mood,
            $track->has_vocals ? 'true' : 'false',
            $track->is_acoustic ? 'true' : 'false',
            $track->is_liked ? 'true' : 'false',
            $track->source,
            $track->youtube_url,
            $track->created_at?->toIso8601String(),
            $track->updated_at?->toIso8601String(),
        ];
    }
}
