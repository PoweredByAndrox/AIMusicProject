<?php

namespace Tests\Feature\Properties;

use App\Services\YouTubeImporter;
use Eris\Generator;
use Eris\TestTrait;
use Tests\TestCase;

/**
 * Property Test: YouTube URL Validation
 * 
 * @tag Feature: ai-music-recommendations, Property 16: YouTube URL Validation
 * **Validates: Requirements 2.1, 2.2, 2.3, 2.5**
 * 
 * For any entered URL, the system should correctly determine the type
 * (video, playlist, channel) or return an error for invalid URLs.
 */
class YouTubeUrlValidationPropertyTest extends TestCase
{
    use TestTrait;

    private YouTubeImporter $importer;

    protected function setUp(): void
    {
        parent::setUp();
        $this->importer = new YouTubeImporter();
    }

    /**
     * @test
     */
    public function valid_video_urls_are_recognized(): void
    {
        $videoIds = ['dQw4w9WgXcQ', 'jNQXAC9IVRw', 'kJQP7kiw5Fk', 'M7lc1UVf-VE'];
        
        $this
            ->limitTo(100)
            ->forAll(Generator\elements($videoIds))
            ->then(function ($videoId) {
                $urls = [
                    "https://www.youtube.com/watch?v={$videoId}",
                    "https://youtube.com/watch?v={$videoId}",
                    "https://youtu.be/{$videoId}",
                    "https://www.youtube.com/embed/{$videoId}",
                    "https://www.youtube.com/v/{$videoId}",
                ];

                foreach ($urls as $url) {
                    $result = $this->importer->validateUrl($url);
                    
                    $this->assertTrue($result['valid'], "URL should be valid: {$url}");
                    $this->assertEquals(YouTubeImporter::TYPE_VIDEO, $result['type']);
                    $this->assertEquals($videoId, $result['id']);
                }
            });
    }

    /**
     * @test
     */
    public function valid_playlist_urls_are_recognized(): void
    {
        $playlistIds = [
            'PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf',
            'PL4cUxeGkcC9gcy9lrvMJ75z9maRw4byYp',
            'PLDoPjvoNmBAy532K9M_fjiAmrJ0gkCyLJ',
        ];
        
        $this
            ->limitTo(50)
            ->forAll(Generator\elements($playlistIds))
            ->then(function ($playlistId) {
                $urls = [
                    "https://www.youtube.com/playlist?list={$playlistId}",
                    "https://youtube.com/playlist?list={$playlistId}",
                ];

                foreach ($urls as $url) {
                    $result = $this->importer->validateUrl($url);
                    
                    $this->assertTrue($result['valid'], "URL should be valid: {$url}");
                    $this->assertEquals(YouTubeImporter::TYPE_PLAYLIST, $result['type']);
                    $this->assertEquals($playlistId, $result['id']);
                }
            });
    }

    /**
     * @test
     */
    public function valid_channel_urls_are_recognized(): void
    {
        $channelFormats = [
            ['type' => 'channel', 'id' => 'UCuAXFkgsw1L7xaCfnd5JJOw'],
            ['type' => 'handle', 'id' => 'MrBeast'],
            ['type' => 'c', 'id' => 'PewDiePie'],
            ['type' => 'user', 'id' => 'Google'],
        ];
        
        $this
            ->limitTo(50)
            ->forAll(Generator\elements($channelFormats))
            ->then(function ($format) {
                $urlPatterns = [
                    'channel' => "https://www.youtube.com/channel/{$format['id']}",
                    'handle' => "https://www.youtube.com/@{$format['id']}",
                    'c' => "https://www.youtube.com/c/{$format['id']}",
                    'user' => "https://www.youtube.com/user/{$format['id']}",
                ];

                $url = $urlPatterns[$format['type']];
                $result = $this->importer->validateUrl($url);
                
                $this->assertTrue($result['valid'], "URL should be valid: {$url}");
                $this->assertEquals(YouTubeImporter::TYPE_CHANNEL, $result['type']);
                $this->assertEquals($format['id'], $result['id']);
            });
    }

    /**
     * @test
     */
    public function invalid_urls_are_rejected(): void
    {
        $invalidUrls = [
            'https://www.google.com',
            'https://vimeo.com/123456',
            'https://www.dailymotion.com/video/x123456',
            'not-a-url',
            '',
            'https://youtube.com',
            'https://www.youtube.com/results?search_query=test',
            'ftp://youtube.com/watch?v=test',
        ];
        
        $this
            ->limitTo(50)
            ->forAll(Generator\elements($invalidUrls))
            ->then(function ($url) {
                $result = $this->importer->validateUrl($url);
                
                $this->assertFalse($result['valid'], "URL should be invalid: {$url}");
                $this->assertEquals(YouTubeImporter::TYPE_INVALID, $result['type']);
                $this->assertNull($result['id']);
            });
    }

    /**
     * @test
     */
    public function url_type_detection_is_deterministic(): void
    {
        $testUrls = [
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'https://youtu.be/dQw4w9WgXcQ',
            'https://www.youtube.com/playlist?list=PLtest123',
            'https://www.youtube.com/@TestChannel',
        ];
        
        $this
            ->limitTo(100)
            ->forAll(Generator\elements($testUrls))
            ->then(function ($url) {
                $result1 = $this->importer->validateUrl($url);
                $result2 = $this->importer->validateUrl($url);
                
                // Same URL should always produce same result
                $this->assertEquals($result1['type'], $result2['type']);
                $this->assertEquals($result1['id'], $result2['id']);
                $this->assertEquals($result1['valid'], $result2['valid']);
            });
    }
}
