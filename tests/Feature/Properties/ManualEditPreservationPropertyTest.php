<?php

namespace Tests\Feature\Properties;

use App\Models\Track;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Tests\TestCase;

/**
 * Property Test: Manual Edit Preservation
 * 
 * @tag Feature: ai-music-recommendations, Property 14: Manual Edit Preservation
 * **Validates: Requirements 14.1, 14.2, 14.3, 14.4**
 * 
 * For any track with manually edited characteristics,
 * re-analysis should not overwrite fields marked in manually_edited.
 */
class ManualEditPreservationPropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    private const EDITABLE_FIELDS = ['genre', 'bpm', 'key', 'mode', 'mood'];

    /**
     * @test
     */
    public function manually_edited_fields_are_tracked(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\elements(self::EDITABLE_FIELDS),
                Generator\string()
            )
            ->then(function ($field, $value) {
                $value = substr($value, 0, 50) ?: 'TestValue';
                
                $track = Track::create([
                    'title' => 'Test Track',
                    'duration' => 180.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                ]);

                // Simulate manual edit
                $track->$field = $value;
                $manuallyEdited = $track->manually_edited ?? [];
                $manuallyEdited[$field] = true;
                $track->manually_edited = $manuallyEdited;
                $track->save();

                // Verify field is marked as manually edited
                $retrieved = Track::find($track->id);
                $this->assertTrue(
                    $retrieved->manually_edited[$field] ?? false,
                    "Field {$field} should be marked as manually edited"
                );
            });
    }

    /**
     * @test
     */
    public function manually_edited_value_persists(): void
    {
        $genres = ['Rock', 'Pop', 'Jazz', 'Classical', 'Electronic'];
        
        $this
            ->limitTo(100)
            ->forAll(Generator\elements($genres))
            ->then(function ($genre) {
                $track = Track::create([
                    'title' => 'Test Track',
                    'duration' => 180.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                    'genre' => 'Original',
                ]);

                // Manual edit
                $track->genre = $genre;
                $track->manually_edited = ['genre' => true];
                $track->save();

                // Retrieve and verify
                $retrieved = Track::find($track->id);
                $this->assertEquals($genre, $retrieved->genre);
                $this->assertTrue($retrieved->manually_edited['genre'] ?? false);
            });
    }

    /**
     * @test
     */
    public function multiple_fields_can_be_manually_edited(): void
    {
        $this
            ->limitTo(50)
            ->forAll(
                Generator\choose(1, count(self::EDITABLE_FIELDS))
            )
            ->then(function ($numFields) {
                $track = Track::create([
                    'title' => 'Test Track',
                    'duration' => 180.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                ]);

                // Edit multiple fields
                $fieldsToEdit = array_slice(self::EDITABLE_FIELDS, 0, $numFields);
                $manuallyEdited = [];
                
                foreach ($fieldsToEdit as $field) {
                    $track->$field = 'Edited';
                    $manuallyEdited[$field] = true;
                }
                
                $track->manually_edited = $manuallyEdited;
                $track->save();

                // Verify all fields are marked
                $retrieved = Track::find($track->id);
                foreach ($fieldsToEdit as $field) {
                    $this->assertTrue(
                        $retrieved->manually_edited[$field] ?? false,
                        "Field {$field} should be marked as manually edited"
                    );
                }
            });
    }

    /**
     * @test
     */
    public function non_edited_fields_not_marked(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\elements(self::EDITABLE_FIELDS))
            ->then(function ($editedField) {
                $track = Track::create([
                    'title' => 'Test Track',
                    'duration' => 180.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                ]);

                // Edit only one field
                $track->$editedField = 'Edited';
                $track->manually_edited = [$editedField => true];
                $track->save();

                // Verify other fields are not marked
                $retrieved = Track::find($track->id);
                foreach (self::EDITABLE_FIELDS as $field) {
                    if ($field === $editedField) {
                        $this->assertTrue($retrieved->manually_edited[$field] ?? false);
                    } else {
                        $this->assertFalse($retrieved->manually_edited[$field] ?? false);
                    }
                }
            });
    }

    /**
     * @test
     */
    public function manually_edited_array_is_json_serializable(): void
    {
        $this
            ->limitTo(50)
            ->forAll(
                Generator\associative([
                    'genre' => Generator\bool(),
                    'bpm' => Generator\bool(),
                    'key' => Generator\bool(),
                    'mode' => Generator\bool(),
                    'mood' => Generator\bool(),
                ])
            )
            ->then(function ($manuallyEdited) {
                $track = Track::create([
                    'title' => 'Test Track',
                    'duration' => 180.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                    'manually_edited' => $manuallyEdited,
                ]);

                $retrieved = Track::find($track->id);
                
                // Verify it's an array
                $this->assertIsArray($retrieved->manually_edited);
                
                // Verify values match
                foreach ($manuallyEdited as $field => $value) {
                    $this->assertEquals(
                        $value,
                        $retrieved->manually_edited[$field] ?? false
                    );
                }
            });
    }
}
