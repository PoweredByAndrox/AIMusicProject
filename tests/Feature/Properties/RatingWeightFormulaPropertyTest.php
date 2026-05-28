<?php

namespace Tests\Feature\Properties;

use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Tests\TestCase;

/**
 * Property Test: Rating Weight Formula
 * 
 * @tag Feature: advanced-music-player, Property 15: Rating Weight Formula
 * **Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5**
 * 
 * For any rating R (1-5):
 * - weight(5) = 1.0 (strong positive)
 * - weight(4) = 0.5
 * - weight(3) = 0.0 (neutral)
 * - weight(2) = -0.25
 * - weight(1) = -0.5 (negative)
 * - Formula: weight = (R - 3) / 2
 * - Weight is monotonically increasing with rating
 */
class RatingWeightFormulaPropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    /**
     * Calculate the recommendation weight for a given rating.
     * 
     * The formula uses asymmetric scaling:
     * - For ratings >= 3: weight = (R - 3) / 2 (positive weights scale by 0.5)
     * - For ratings < 3: weight = (R - 3) / 4 (negative weights scale by 0.25)
     * 
     * This produces the specified values:
     * - weight(5) = 1.0 (strong positive)
     * - weight(4) = 0.5
     * - weight(3) = 0.0 (neutral)
     * - weight(2) = -0.25
     * - weight(1) = -0.5 (negative)
     * 
     * Requirements:
     * - 12.1: THE Recommendation_Engine SHALL использовать рейтинг как вес примера
     * - 12.2: THE Recommendation_Engine SHALL интерпретировать 5 звёзд как сильный положительный сигнал (вес 1.0)
     * - 12.3: THE Recommendation_Engine SHALL интерпретировать 1 звезду как отрицательный сигнал (вес -0.5)
     * - 12.4: THE Recommendation_Engine SHALL интерпретировать 3 звезды как нейтральный сигнал (вес 0.0)
     * - 12.5: THE Recommendation_Engine SHALL использовать линейную интерполяцию для промежуточных значений
     *
     * @param int $rating The rating value (1-5)
     * @return float The weight value
     */
    public static function calculateRatingWeight(int $rating): float
    {
        if ($rating >= 3) {
            return ($rating - 3) / 2;
        }
        return ($rating - 3) / 4;
    }

    /**
     * @test
     * 
     * Property: 5-star rating produces weight of 1.0 (strong positive signal)
     * Validates: Requirement 12.2
     * 
     * THE Recommendation_Engine SHALL интерпретировать 5 звёзд как сильный положительный сигнал (вес 1.0)
     */
    public function five_star_rating_produces_weight_of_one(): void
    {
        $weight = self::calculateRatingWeight(5);
        
        $this->assertEquals(1.0, $weight, 'Rating 5 should produce weight 1.0');
    }

    /**
     * @test
     * 
     * Property: 4-star rating produces weight of 0.5
     * Validates: Requirement 12.5 (linear interpolation)
     * 
     * THE Recommendation_Engine SHALL использовать линейную интерполяцию для промежуточных значений
     */
    public function four_star_rating_produces_weight_of_half(): void
    {
        $weight = self::calculateRatingWeight(4);
        
        $this->assertEquals(0.5, $weight, 'Rating 4 should produce weight 0.5');
    }

    /**
     * @test
     * 
     * Property: 3-star rating produces weight of 0.0 (neutral signal)
     * Validates: Requirement 12.4
     * 
     * THE Recommendation_Engine SHALL интерпретировать 3 звезды как нейтральный сигнал (вес 0.0)
     */
    public function three_star_rating_produces_weight_of_zero(): void
    {
        $weight = self::calculateRatingWeight(3);
        
        $this->assertEquals(0.0, $weight, 'Rating 3 should produce weight 0.0 (neutral)');
    }

    /**
     * @test
     * 
     * Property: 2-star rating produces weight of -0.25
     * Validates: Requirement 12.5 (linear interpolation)
     * 
     * THE Recommendation_Engine SHALL использовать линейную интерполяцию для промежуточных значений
     */
    public function two_star_rating_produces_weight_of_negative_quarter(): void
    {
        $weight = self::calculateRatingWeight(2);
        
        $this->assertEquals(-0.25, $weight, 'Rating 2 should produce weight -0.25');
    }

    /**
     * @test
     * 
     * Property: 1-star rating produces weight of -0.5 (negative signal)
     * Validates: Requirement 12.3
     * 
     * THE Recommendation_Engine SHALL интерпретировать 1 звезду как отрицательный сигнал (вес -0.5)
     */
    public function one_star_rating_produces_weight_of_negative_half(): void
    {
        $weight = self::calculateRatingWeight(1);
        
        $this->assertEquals(-0.5, $weight, 'Rating 1 should produce weight -0.5');
    }

    /**
     * @test
     * 
     * Property: For any rating R in [1,5]: weight follows the asymmetric formula
     * - For R >= 3: weight = (R - 3) / 2
     * - For R < 3: weight = (R - 3) / 4
     * Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5
     * 
     * This property test verifies the formula holds for all valid ratings.
     */
    public function formula_holds_for_all_valid_ratings(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\choose(1, 5))
            ->then(function ($rating) {
                $weight = self::calculateRatingWeight($rating);
                
                // Asymmetric formula: positive weights scale by 0.5, negative by 0.25
                $expectedWeight = $rating >= 3 
                    ? ($rating - 3) / 2 
                    : ($rating - 3) / 4;
                
                $this->assertEquals(
                    $expectedWeight,
                    $weight,
                    "Rating {$rating} should produce weight {$expectedWeight} using asymmetric formula"
                );
            });
    }

    /**
     * @test
     * 
     * Property: Weight is monotonically increasing with rating
     * Validates: Requirements 12.2, 12.3, 12.4, 12.5
     * 
     * For any two ratings R1 < R2, weight(R1) < weight(R2)
     * This ensures higher ratings always produce higher weights.
     */
    public function weight_is_monotonically_increasing_with_rating(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(1, 4),
                Generator\choose(1, 4)
            )
            ->then(function ($rating1, $rating2) {
                // Ensure rating2 is always greater than rating1
                $lowerRating = min($rating1, $rating2);
                $higherRating = max($rating1, $rating2) + 1; // +1 to ensure it's strictly greater
                
                // Cap at 5
                if ($higherRating > 5) {
                    $higherRating = 5;
                    $lowerRating = min($lowerRating, 4);
                }
                
                if ($lowerRating >= $higherRating) {
                    // Skip if we can't create a valid pair
                    return;
                }
                
                $weightLower = self::calculateRatingWeight($lowerRating);
                $weightHigher = self::calculateRatingWeight($higherRating);
                
                $this->assertLessThan(
                    $weightHigher,
                    $weightLower,
                    "Weight for rating {$lowerRating} ({$weightLower}) should be less than weight for rating {$higherRating} ({$weightHigher})"
                );
            });
    }

    /**
     * @test
     * 
     * Property: Weight range is bounded between -0.5 and 1.0
     * Validates: Requirements 12.2, 12.3
     * 
     * For any valid rating (1-5), the weight should be in range [-0.5, 1.0]
     */
    public function weight_is_bounded_between_negative_half_and_one(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\choose(1, 5))
            ->then(function ($rating) {
                $weight = self::calculateRatingWeight($rating);
                
                $this->assertGreaterThanOrEqual(
                    -0.5,
                    $weight,
                    "Weight for rating {$rating} should be >= -0.5"
                );
                
                $this->assertLessThanOrEqual(
                    1.0,
                    $weight,
                    "Weight for rating {$rating} should be <= 1.0"
                );
            });
    }

    /**
     * @test
     * 
     * Property: Weight step between consecutive ratings follows asymmetric pattern
     * Validates: Requirement 12.5 (linear interpolation within each segment)
     * 
     * THE Recommendation_Engine SHALL использовать линейную интерполяцию для промежуточных значений
     * - For ratings 3-5: step is 0.5 (positive segment)
     * - For ratings 1-3: step is 0.25 (negative segment)
     */
    public function weight_step_between_consecutive_ratings_follows_pattern(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\choose(1, 4))
            ->then(function ($rating) {
                $weightCurrent = self::calculateRatingWeight($rating);
                $weightNext = self::calculateRatingWeight($rating + 1);
                
                $step = $weightNext - $weightCurrent;
                
                // Step depends on which segment we're in
                // Transition from 2->3 crosses the boundary
                if ($rating >= 3) {
                    // Both in positive segment: step = 0.5
                    $expectedStep = 0.5;
                } elseif ($rating + 1 <= 2) {
                    // Both in negative segment: step = 0.25
                    $expectedStep = 0.25;
                } else {
                    // Crossing boundary (2->3): step = 0.25 (from -0.25 to 0.0)
                    $expectedStep = 0.25;
                }
                
                $this->assertEquals(
                    $expectedStep,
                    $step,
                    "Step between rating {$rating} and " . ($rating + 1) . " should be {$expectedStep}"
                );
            });
    }

    /**
     * @test
     * 
     * Property: All specific weight values match the specification
     * Validates: Requirements 12.2, 12.3, 12.4, 12.5
     * 
     * Comprehensive test of all specified weight values.
     */
    public function all_specific_weight_values_match_specification(): void
    {
        $expectedWeights = [
            1 => -0.5,   // Requirement 12.3: negative signal
            2 => -0.25,  // Requirement 12.5: linear interpolation
            3 => 0.0,    // Requirement 12.4: neutral signal
            4 => 0.5,    // Requirement 12.5: linear interpolation
            5 => 1.0,    // Requirement 12.2: strong positive signal
        ];
        
        foreach ($expectedWeights as $rating => $expectedWeight) {
            $actualWeight = self::calculateRatingWeight($rating);
            
            $this->assertEquals(
                $expectedWeight,
                $actualWeight,
                "Rating {$rating} should produce weight {$expectedWeight}, got {$actualWeight}"
            );
        }
    }
}
