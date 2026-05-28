<?php

namespace App\Services;

use App\Models\Track;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Storage;
use RuntimeException;

class PublicRadioWaveformService
{
    private const WAVEFORM_PACKAGE_VERSION = 'v2';
    private const WAVEFORM_SAMPLE_RATE = 8000;
    private const WAVEFORM_MIN_PEAKS = 240;
    private const WAVEFORM_MAX_PEAKS = 1600;

    /**
     * @return array{duration:float,peaks:array<int,float>}
     */
    public function getWaveformPayload(Track $track): array
    {
        $waveformPath = $this->ensureWaveformPackage($track);
        $payload = @file_get_contents($waveformPath);
        if ($payload === false) {
            throw new RuntimeException('Failed to load waveform package');
        }

        $decoded = json_decode($payload, true);
        if (!is_array($decoded)) {
            throw new RuntimeException('Waveform package is invalid');
        }

        return [
            'duration' => (float) ($decoded['duration'] ?? 0),
            'peaks' => array_values(array_map(
                static fn ($peak): float => (float) $peak,
                is_array($decoded['peaks'] ?? null) ? $decoded['peaks'] : [],
            )),
        ];
    }

    public function ensureWaveformPackage(Track $track): string
    {
        if (!function_exists('proc_open')) {
            throw new RuntimeException('Waveform packaging is not available in this runtime');
        }

        $sourcePath = $this->resolveSourcePath($track);
        $duration = $this->resolveTrackDurationSeconds($track, $sourcePath);
        $peakCount = $this->resolveWaveformPeakCount($duration);
        $signature = sha1(implode('|', [
            self::WAVEFORM_PACKAGE_VERSION,
            (string) $track->id,
            (string) (filesize($sourcePath) ?: 0),
            (string) (filemtime($sourcePath) ?: 0),
            (string) self::WAVEFORM_SAMPLE_RATE,
            (string) $peakCount,
        ]));

        $directory = storage_path('app/public_radio_waveforms/' . (int) $track->id);
        $waveformPath = $directory . DIRECTORY_SEPARATOR . $signature . '.json';
        if (is_file($waveformPath)) {
            $this->publishCurrentWaveformPackage($directory, $waveformPath);
            return $waveformPath;
        }

        $lockKey = 'public_radio_waveform_package:' . $track->id . ':' . $signature;

        return Cache::lock($lockKey, 300)->block(15, function () use ($directory, $waveformPath, $sourcePath, $duration, $peakCount): string {
            if (is_file($waveformPath)) {
                $this->publishCurrentWaveformPackage($directory, $waveformPath);
                return $waveformPath;
            }

            if (!is_dir($directory) && !mkdir($directory, 0775, true) && !is_dir($directory)) {
                throw new RuntimeException('Failed to create waveform package directory');
            }

            $peaks = $this->generateWaveformPeaks($sourcePath, $duration, $peakCount);
            $payload = json_encode([
                'duration' => round($duration, 3),
                'peaks' => $peaks,
            ], JSON_UNESCAPED_SLASHES);

            if ($payload === false) {
                throw new RuntimeException('Failed to encode waveform package');
            }

            $tempPath = $waveformPath . '.tmp' . bin2hex(random_bytes(4));
            if (@file_put_contents($tempPath, $payload) === false) {
                @unlink($tempPath);
                throw new RuntimeException('Failed to write waveform package');
            }

            if (!@rename($tempPath, $waveformPath)) {
                @unlink($tempPath);
                throw new RuntimeException('Failed to finalize waveform package');
            }

            $this->publishCurrentWaveformPackage($directory, $waveformPath);

            return $waveformPath;
        });
    }

    public function currentPublicUrl(int $trackId): ?string
    {
        return is_file($this->currentWaveformPath($trackId))
            ? '/storage/public_radio_waveforms/' . $trackId . '/current.json'
            : null;
    }

    public function currentWaveformPath(int $trackId): string
    {
        return storage_path('app/public_radio_waveforms/' . $trackId . '/current.json');
    }

    private function resolveSourcePath(Track $track): string
    {
        if (!$track->file_path || !Storage::disk('local')->exists($track->file_path)) {
            throw new RuntimeException('Audio file not found');
        }

        return Storage::disk('local')->path($track->file_path);
    }

    private function resolveTrackDurationSeconds(Track $track, string $sourcePath): float
    {
        $storedDuration = (float) $track->duration;
        if ($storedDuration > 0) {
            return $storedDuration;
        }

        $descriptors = [
            0 => ['pipe', 'r'],
            1 => ['pipe', 'w'],
            2 => ['pipe', 'w'],
        ];

        $process = proc_open([
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            $sourcePath,
        ], $descriptors, $pipes);

        if (!is_resource($process)) {
            throw new RuntimeException('Failed to start ffprobe for waveform duration');
        }

        fclose($pipes[0]);
        $stdout = stream_get_contents($pipes[1]);
        $stderr = stream_get_contents($pipes[2]);
        fclose($pipes[1]);
        fclose($pipes[2]);
        $exitCode = proc_close($process);

        $duration = is_string($stdout) ? (float) trim($stdout) : 0.0;
        if ($exitCode !== 0 || $duration <= 0) {
            throw new RuntimeException(trim((string) $stderr) !== '' ? trim((string) $stderr) : 'Failed to resolve waveform duration');
        }

        return $duration;
    }

    private function resolveWaveformPeakCount(float $duration): int
    {
        $dynamicCount = (int) ceil(max(1.0, $duration) * 4);
        return max(self::WAVEFORM_MIN_PEAKS, min(self::WAVEFORM_MAX_PEAKS, $dynamicCount));
    }

    /**
     * @return array<int, float>
     */
    private function generateWaveformPeaks(string $sourcePath, float $duration, int $peakCount): array
    {
        $descriptors = [
            0 => ['pipe', 'r'],
            1 => ['pipe', 'w'],
            2 => ['pipe', 'w'],
        ];

        $process = proc_open([
            'ffmpeg',
            '-v', 'error',
            '-nostdin',
            '-i', $sourcePath,
            '-vn',
            '-map', 'a:0',
            '-ac', '1',
            '-ar', (string) self::WAVEFORM_SAMPLE_RATE,
            '-f', 'f32le',
            'pipe:1',
        ], $descriptors, $pipes);

        if (!is_resource($process)) {
            throw new RuntimeException('Failed to start waveform peak generator');
        }

        fclose($pipes[0]);

        $peakMaxima = array_fill(0, $peakCount, 0.0);
        $bucketEnergy = array_fill(0, $peakCount, 0.0);
        $bucketSamples = array_fill(0, $peakCount, 0);
        $estimatedSamples = max($peakCount, (int) ceil(max(1.0, $duration) * self::WAVEFORM_SAMPLE_RATE));
        $samplesPerBucket = max(1.0, $estimatedSamples / $peakCount);
        $sampleIndex = 0;
        $buffer = '';

        while (!feof($pipes[1])) {
            $chunk = fread($pipes[1], 65536);
            if ($chunk === false || $chunk === '') {
                continue;
            }

            $buffer .= $chunk;
            $usableLength = strlen($buffer) - (strlen($buffer) % 4);
            if ($usableLength <= 0) {
                continue;
            }

            $payload = substr($buffer, 0, $usableLength);
            $buffer = (string) substr($buffer, $usableLength);
            $samples = unpack('g*', $payload);
            if ($samples === false) {
                continue;
            }

            foreach ($samples as $value) {
                $bucketIndex = min($peakCount - 1, (int) floor($sampleIndex / $samplesPerBucket));
                $amplitude = min(1.0, abs((float) $value));
                if ($amplitude > $peakMaxima[$bucketIndex]) {
                    $peakMaxima[$bucketIndex] = $amplitude;
                }
                $bucketEnergy[$bucketIndex] += $amplitude * $amplitude;
                $bucketSamples[$bucketIndex]++;
                $sampleIndex++;
            }
        }

        $stderr = stream_get_contents($pipes[2]);
        fclose($pipes[1]);
        fclose($pipes[2]);
        $exitCode = proc_close($process);

        if ($exitCode !== 0) {
            throw new RuntimeException(trim((string) $stderr) !== '' ? trim((string) $stderr) : 'Waveform peak generation failed');
        }

        $peaks = [];
        $maxPeak = 0.0;

        for ($index = 0; $index < $peakCount; $index++) {
            $sampleCount = max(1, $bucketSamples[$index]);
            $rms = sqrt($bucketEnergy[$index] / $sampleCount);
            $smoothedPeak = ($rms * 0.9) + ($peakMaxima[$index] * 0.1);
            $peaks[$index] = $smoothedPeak;
            if ($smoothedPeak > $maxPeak) {
                $maxPeak = $smoothedPeak;
            }
        }

        if ($maxPeak <= 0.0) {
            return array_fill(0, $peakCount, 0.0);
        }

        return array_map(
            static fn (float $peak): float => round(min(1.0, $peak / $maxPeak), 4),
            $peaks,
        );
    }

    private function publishCurrentWaveformPackage(string $directory, string $waveformPath): void
    {
        $currentPath = $directory . DIRECTORY_SEPARATOR . 'current.json';

        if (realpath($currentPath) === realpath($waveformPath)) {
            return;
        }

        if (is_link($currentPath) || is_file($currentPath)) {
            @unlink($currentPath);
        }

        if (function_exists('symlink') && @symlink(basename($waveformPath), $currentPath)) {
            return;
        }

        if (@copy($waveformPath, $currentPath) === false) {
            throw new RuntimeException('Failed to publish current waveform package');
        }
    }
}
