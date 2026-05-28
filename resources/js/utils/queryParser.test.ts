/**
 * Property-based tests for QueryParser.
 * 
 * Tests Properties 1, 2, 3 from design document.
 * Requirements: 1.1, 1.2, 1.5, 1.6
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import {
  parseTime,
  formatTime,
  parseQuery,
  validateTimeRange,
  isParseError,
  ParsedQuery
} from './queryParser';

describe('QueryParser', () => {
  describe('Property 1: Time parsing round-trip', () => {
    /**
     * Property 1: For any valid time string in "MM:SS" or "Ns" format,
     * parsing to seconds and formatting back should give equivalent time.
     * 
     * **Validates: Requirements 1.1, 1.2**
     */
    it('parseTime and formatTime are inverse operations for valid times', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 3599 }), // 0 to 59:59
          (seconds) => {
            const formatted = formatTime(seconds);
            const parsed = parseTime(formatted);
            return parsed === seconds;
          }
        ),
        { numRuns: 100 }
      );
    });

    it('parses MM:SS format correctly', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 59 }), // minutes
          fc.integer({ min: 0, max: 59 }), // seconds
          (minutes, seconds) => {
            const timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            const parsed = parseTime(timeStr);
            return parsed === minutes * 60 + seconds;
          }
        ),
        { numRuns: 100 }
      );
    });

    it('parses Ns format correctly', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 3600 }),
          (seconds) => {
            const timeStr = `${seconds}s`;
            const parsed = parseTime(timeStr);
            return parsed === seconds;
          }
        ),
        { numRuns: 100 }
      );
    });
  });

  describe('Property 2: Rejection of invalid time ranges', () => {
    /**
     * Property 2: For any time range where start_time >= end_time or
     * any value is negative, parseQuery should return a ParseError.
     * 
     * **Validates: Requirements 1.5**
     */
    it('rejects ranges where start >= end', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 3600 }),
          fc.integer({ min: 0, max: 100 }),
          (start, delta) => {
            // end <= start
            const end = start - delta;
            if (end < 0) return true; // Skip negative end times
            
            const query = `${formatTime(start)} - ${formatTime(end)}`;
            const result = parseQuery(query, 1);
            
            // Should be an error if start >= end
            if (start >= end) {
              return isParseError(result) && result.code === 'INVALID_RANGE';
            }
            return true;
          }
        ),
        { numRuns: 100 }
      );
    });

    it('rejects negative time values in validateTimeRange', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: -1000, max: -1 }), // negative start
          fc.integer({ min: 1, max: 3600 }),   // positive end
          fc.integer({ min: 1, max: 7200 }),   // duration
          (start, end, duration) => {
            const result = validateTimeRange(start, end, duration);
            return isParseError(result) && result.code === 'NEGATIVE_TIME';
          }
        ),
        { numRuns: 50 }
      );
    });
  });

  describe('Property 3: Time range clamping to track duration', () => {
    /**
     * Property 3: For any time range and track with known duration,
     * if end_time > duration, the resulting end_time should equal duration.
     * 
     * **Validates: Requirements 1.6**
     */
    it('clamps end_time to track duration', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 100 }),    // start
          fc.integer({ min: 200, max: 500 }),  // end (larger than duration)
          fc.integer({ min: 101, max: 199 }), // duration (between start and end)
          (start, end, duration) => {
            const result = validateTimeRange(start, end, duration);
            
            if (isParseError(result)) {
              return false; // Should not be an error
            }
            
            // end_time should be clamped to duration
            return result.endTime === duration;
          }
        ),
        { numRuns: 100 }
      );
    });

    it('parseQuery clamps end_time when trackDuration provided', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 50 }),     // start
          fc.integer({ min: 100, max: 200 }), // end
          fc.integer({ min: 51, max: 99 }),   // duration (less than end)
          (start, end, duration) => {
            const query = `${formatTime(start)} - ${formatTime(end)}`;
            const result = parseQuery(query, 1, duration);
            
            if (isParseError(result)) {
              return false;
            }
            
            return result.endTime === duration;
          }
        ),
        { numRuns: 100 }
      );
    });
  });

  describe('Unit tests for specific formats', () => {
    it('parses "00:29 - 00:35" format', () => {
      const result = parseQuery('00:29 - 00:35', 1);
      expect(isParseError(result)).toBe(false);
      if (!isParseError(result)) {
        expect(result.startTime).toBe(29);
        expect(result.endTime).toBe(35);
      }
    });

    it('parses "0:29 - 0:35" format', () => {
      const result = parseQuery('0:29 - 0:35', 1);
      expect(isParseError(result)).toBe(false);
      if (!isParseError(result)) {
        expect(result.startTime).toBe(29);
        expect(result.endTime).toBe(35);
      }
    });

    it('parses "29s - 35s" format', () => {
      const result = parseQuery('29s - 35s', 1);
      expect(isParseError(result)).toBe(false);
      if (!isParseError(result)) {
        expect(result.startTime).toBe(29);
        expect(result.endTime).toBe(35);
      }
    });

    it('parses "1:30 - 2:00" format', () => {
      const result = parseQuery('1:30 - 2:00', 1);
      expect(isParseError(result)).toBe(false);
      if (!isParseError(result)) {
        expect(result.startTime).toBe(90);
        expect(result.endTime).toBe(120);
      }
    });

    it('parses query with surrounding text', () => {
      const result = parseQuery('найди похожие на 0:29 - 0:35 момент', 1);
      expect(isParseError(result)).toBe(false);
      if (!isParseError(result)) {
        expect(result.startTime).toBe(29);
        expect(result.endTime).toBe(35);
      }
    });

    it('uses currentTrackId when no track specified', () => {
      const result = parseQuery('0:29 - 0:35', 123);
      expect(isParseError(result)).toBe(false);
      if (!isParseError(result)) {
        expect(result.trackId).toBe(123);
      }
    });

    it('returns null trackId when no currentTrackId provided', () => {
      const result = parseQuery('0:29 - 0:35');
      expect(isParseError(result)).toBe(false);
      if (!isParseError(result)) {
        expect(result.trackId).toBeNull();
      }
    });

    it('returns error for invalid format', () => {
      const result = parseQuery('find similar tracks');
      expect(isParseError(result)).toBe(true);
      if (isParseError(result)) {
        expect(result.code).toBe('INVALID_TIME_FORMAT');
      }
    });

    it('returns error for empty query', () => {
      const result = parseQuery('');
      expect(isParseError(result)).toBe(true);
    });
  });

  describe('formatTime', () => {
    it('formats seconds correctly', () => {
      expect(formatTime(0)).toBe('0:00');
      expect(formatTime(29)).toBe('0:29');
      expect(formatTime(60)).toBe('1:00');
      expect(formatTime(90)).toBe('1:30');
      expect(formatTime(3600)).toBe('1:00:00');
      expect(formatTime(3661)).toBe('1:01:01');
    });

    it('handles edge cases', () => {
      expect(formatTime(-1)).toBe('0:00');
      expect(formatTime(Infinity)).toBe('0:00');
    });
  });

  describe('parseTime', () => {
    it('parses various formats', () => {
      expect(parseTime('0:00')).toBe(0);
      expect(parseTime('0:29')).toBe(29);
      expect(parseTime('1:00')).toBe(60);
      expect(parseTime('1:30')).toBe(90);
      expect(parseTime('10:00')).toBe(600);
      expect(parseTime('29s')).toBe(29);
      expect(parseTime('90s')).toBe(90);
      expect(parseTime('90')).toBe(90);
    });

    it('returns null for invalid formats', () => {
      expect(parseTime('')).toBeNull();
      expect(parseTime('abc')).toBeNull();
      expect(parseTime('1:60')).toBeNull(); // Invalid seconds
      expect(parseTime('-1:00')).toBeNull();
    });
  });
});
