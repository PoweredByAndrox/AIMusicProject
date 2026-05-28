<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use Illuminate\Support\Facades\DB;

class SegmentsCoverage extends Command
{
    protected $signature = 'segments:coverage {--json}';
    protected $description = 'Report coverage of pattern_examples.embedding_path (segment embedding fill)';

    public function handle(): int
    {
        $asJson = (bool) $this->option('json');

        $totalExamples = (int) DB::table('pattern_examples')->count();
        $withEmbedding = (int) DB::table('pattern_examples')->whereNotNull('embedding_path')->count();

        $ratedTotal = (int) DB::table('pattern_examples')
            ->where(function ($q) {
                $q->whereNotNull('rating')->orWhereNotNull('is_positive');
            })
            ->count();

        $ratedWithEmbedding = (int) DB::table('pattern_examples')
            ->whereNotNull('embedding_path')
            ->where(function ($q) {
                $q->whereNotNull('rating')->orWhereNotNull('is_positive');
            })
            ->count();

        $tracksWithSegments = (int) DB::table('pattern_examples')->distinct('track_id')->count('track_id');

        $analyzedTracksWithEmbeddings = (int) DB::table('tracks')
            ->where('analysis_status', 'completed')
            ->whereNotNull('embedding_path')
            ->count();

        $pct = $totalExamples > 0 ? ($withEmbedding / $totalExamples) * 100.0 : 0.0;
        $ratedPct = $ratedTotal > 0 ? ($ratedWithEmbedding / $ratedTotal) * 100.0 : 0.0;

        $result = [
            'pattern_examples_total' => $totalExamples,
            'pattern_examples_with_embedding' => $withEmbedding,
            'pattern_examples_embedding_pct' => round($pct, 2),
            'rated_examples_total' => $ratedTotal,
            'rated_examples_with_embedding' => $ratedWithEmbedding,
            'rated_examples_embedding_pct' => round($ratedPct, 2),
            'tracks_with_segments' => $tracksWithSegments,
            'analyzed_tracks_with_embeddings' => $analyzedTracksWithEmbeddings,
        ];

        if ($asJson) {
            $this->line(json_encode($result, JSON_UNESCAPED_SLASHES));
            return 0;
        }

        $this->info('Segment embedding coverage:');
        $this->line("pattern_examples: {$withEmbedding}/{$totalExamples} (" . $result['pattern_examples_embedding_pct'] . '%)');
        $this->line("rated_examples: {$ratedWithEmbedding}/{$ratedTotal} (" . $result['rated_examples_embedding_pct'] . '%)');
        $this->line("tracks_with_segments: {$tracksWithSegments}");
        $this->line("analyzed_tracks_with_embeddings: {$analyzedTracksWithEmbeddings}");

        return 0;
    }
}

