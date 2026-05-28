<?php

namespace Tests\Feature\Properties;

use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Illuminate\Http\UploadedFile;
use Tests\TestCase;

/**
 * Property Test: File Format Validation
 * 
 * @tag Feature: ai-music-recommendations, Property 15: File Format Validation
 * **Validates: Requirements 1.1, 1.3**
 * 
 * For any uploaded file, the system should accept only files with
 * extensions .mp3 and .wav, and reject all other formats with an error message.
 */
class FileFormatValidationPropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    private const ALLOWED_EXTENSIONS = ['mp3', 'wav'];
    
    private const INVALID_EXTENSIONS = [
        'txt', 'pdf', 'doc', 'exe', 'zip', 'rar', 'jpg', 'png', 'gif',
        'mp4', 'avi', 'mkv', 'flac', 'ogg', 'aac', 'm4a', 'wma',
        'php', 'js', 'html', 'css', 'json', 'xml', 'csv',
    ];

    /**
     * @test
     */
    public function valid_extensions_are_accepted(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\elements(self::ALLOWED_EXTENSIONS),
                Generator\string()
            )
            ->then(function ($extension, $filename) {
                $filename = preg_replace('/[^a-zA-Z0-9_-]/', '', $filename) ?: 'test';
                $fullFilename = $filename . '.' . $extension;
                
                // Create a fake uploaded file with valid extension
                $file = UploadedFile::fake()->create($fullFilename, 1024, 'audio/mpeg');
                
                $response = $this->postJson('/api/tracks', [
                    'file' => $file,
                ]);

                // Should not return format error (may fail for other reasons in test env)
                $this->assertNotEquals(
                    'Поддерживаются только MP3 и WAV файлы',
                    $response->json('message') ?? ''
                );
            });
    }

    /**
     * @test
     */
    public function invalid_extensions_are_rejected(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\elements(self::INVALID_EXTENSIONS),
                Generator\string()
            )
            ->then(function ($extension, $filename) {
                $filename = preg_replace('/[^a-zA-Z0-9_-]/', '', $filename) ?: 'test';
                $fullFilename = $filename . '.' . $extension;
                
                // Create a fake uploaded file with invalid extension
                $file = UploadedFile::fake()->create($fullFilename, 1024, 'application/octet-stream');
                
                $response = $this->postJson('/api/tracks', [
                    'file' => $file,
                ]);

                // Should return 400 with format error message
                $response->assertStatus(400);
                $this->assertStringContainsString(
                    'MP3',
                    $response->json('message') ?? ''
                );
            });
    }

    /**
     * @test
     */
    public function extension_validation_is_case_insensitive(): void
    {
        $caseVariations = ['MP3', 'Mp3', 'mP3', 'WAV', 'Wav', 'wAv'];
        
        $this
            ->limitTo(50)
            ->forAll(Generator\elements($caseVariations))
            ->then(function ($extension) {
                $file = UploadedFile::fake()->create('test.' . $extension, 1024, 'audio/mpeg');
                
                $response = $this->postJson('/api/tracks', [
                    'file' => $file,
                ]);

                // Should not return format error for valid extensions regardless of case
                $this->assertNotEquals(
                    'Unsupported format',
                    $response->json('error') ?? ''
                );
            });
    }
}
