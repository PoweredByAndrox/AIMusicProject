/**
 * QueryParser - Parses user queries to extract time ranges for segment search.
 * 
 * Supports formats:
 * - "00:29 - 00:35" (MM:SS - MM:SS)
 * - "0:29 - 0:35" (M:SS - M:SS)
 * - "29s - 35s" (Ns - Ms)
 * - "1:30 - 2:00" (M:SS - M:SS)
 * 
 * Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
 */

export interface ParsedQuery {
  trackId: number | null;      // null = use current track
  trackName: string | null;    // track name for search
  startTime: number;           // in seconds
  endTime: number;             // in seconds
  originalText: string;        // original query text
}

export interface ParseError {
  code: 'INVALID_TIME_FORMAT' | 'INVALID_RANGE' | 'NEGATIVE_TIME' | 'TRACK_NOT_FOUND';
  message: string;
}

export type ParseResult = ParsedQuery | ParseError;

/**
 * Check if result is an error.
 */
export function isParseError(result: ParseResult): result is ParseError {
  return 'code' in result && 'message' in result;
}

/**
 * Parse a time string to seconds.
 * 
 * Supported formats:
 * - "MM:SS" -> seconds (e.g., "01:30" -> 90)
 * - "M:SS" -> seconds (e.g., "1:30" -> 90)
 * - "Ns" -> seconds (e.g., "90s" -> 90)
 * - "N" -> seconds (e.g., "90" -> 90)
 * 
 * Requirements: 1.1, 1.2
 * 
 * @param timeStr Time string to parse
 * @returns Seconds or null if invalid format
 */
export function parseTime(timeStr: string): number | null {
  if (!timeStr || typeof timeStr !== 'string') {
    return null;
  }
  
  const trimmed = timeStr.trim().toLowerCase();
  
  // Format: "Ns" (e.g., "90s", "35s")
  const secondsMatch = trimmed.match(/^(\d+(?:\.\d+)?)\s*s$/);
  if (secondsMatch) {
    const seconds = parseFloat(secondsMatch[1]);
    return isNaN(seconds) ? null : seconds;
  }
  
  // Format: "MM:SS" or "M:SS" or "H:MM:SS"
  const colonMatch = trimmed.match(/^(\d+):(\d{1,2})(?::(\d{1,2}))?$/);
  if (colonMatch) {
    if (colonMatch[3] !== undefined) {
      // H:MM:SS format
      const hours = parseInt(colonMatch[1], 10);
      const minutes = parseInt(colonMatch[2], 10);
      const seconds = parseInt(colonMatch[3], 10);
      if (minutes >= 60 || seconds >= 60) return null;
      return hours * 3600 + minutes * 60 + seconds;
    } else {
      // M:SS or MM:SS format
      const minutes = parseInt(colonMatch[1], 10);
      const seconds = parseInt(colonMatch[2], 10);
      if (seconds >= 60) return null;
      return minutes * 60 + seconds;
    }
  }
  
  // Format: plain number (seconds)
  const plainNumber = parseFloat(trimmed);
  if (!isNaN(plainNumber) && plainNumber >= 0) {
    return plainNumber;
  }
  
  return null;
}

/**
 * Format seconds to time string.
 * 
 * @param seconds Time in seconds
 * @returns Formatted string "M:SS" or "H:MM:SS"
 */
export function formatTime(seconds: number): string {
  if (seconds < 0 || !isFinite(seconds)) {
    return '0:00';
  }
  
  const totalSeconds = Math.floor(seconds);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;
  
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Extract time range from query text.
 * 
 * Looks for patterns like:
 * - "00:29 - 00:35"
 * - "29s - 35s"
 * - "from 0:29 to 0:35"
 * 
 * @param query Query text
 * @returns Tuple of [startTime, endTime] or null if not found
 */
function extractTimeRange(query: string): [number, number] | null {
  // Pattern: "START - END" or "START-END"
  const rangePatterns = [
    // "00:29 - 00:35" or "0:29-0:35"
    /(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*(\d{1,2}:\d{2}(?::\d{2})?)/,
    // "29s - 35s" or "29s-35s"
    /(\d+(?:\.\d+)?s)\s*[-–—]\s*(\d+(?:\.\d+)?s)/,
    // "from 0:29 to 0:35"
    /(?:from|с|от)\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:to|до|по)\s*(\d{1,2}:\d{2}(?::\d{2})?)/i,
    // "from 29s to 35s"
    /(?:from|с|от)\s*(\d+(?:\.\d+)?s)\s*(?:to|до|по)\s*(\d+(?:\.\d+)?s)/i,
  ];
  
  for (const pattern of rangePatterns) {
    const match = query.match(pattern);
    if (match) {
      const start = parseTime(match[1]);
      const end = parseTime(match[2]);
      if (start !== null && end !== null) {
        return [start, end];
      }
    }
  }
  
  return null;
}

/**
 * Parse user query to extract time range and optional track reference.
 * 
 * Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
 * 
 * @param query User query text
 * @param currentTrackId ID of currently playing track (used if no track specified)
 * @param trackDuration Duration of the track (for validation)
 * @returns ParsedQuery or ParseError
 */
export function parseQuery(
  query: string,
  currentTrackId?: number,
  trackDuration?: number
): ParseResult {
  if (!query || typeof query !== 'string') {
    return {
      code: 'INVALID_TIME_FORMAT',
      message: 'Query is empty or invalid'
    };
  }
  
  // Extract time range
  const timeRange = extractTimeRange(query);
  
  if (!timeRange) {
    return {
      code: 'INVALID_TIME_FORMAT',
      message: 'Could not find time range in query. Use format like "00:29 - 00:35" or "29s - 35s"'
    };
  }
  
  let [startTime, endTime] = timeRange;
  
  // Validate: no negative times (Requirement 1.5)
  if (startTime < 0 || endTime < 0) {
    return {
      code: 'NEGATIVE_TIME',
      message: 'Time values cannot be negative'
    };
  }
  
  // Validate: start < end (Requirement 1.5)
  if (startTime >= endTime) {
    return {
      code: 'INVALID_RANGE',
      message: 'Start time must be less than end time'
    };
  }
  
  // Clamp end_time to track duration (Requirement 1.6)
  if (trackDuration !== undefined && trackDuration > 0) {
    if (endTime > trackDuration) {
      endTime = trackDuration;
    }
    if (startTime >= trackDuration) {
      return {
        code: 'INVALID_RANGE',
        message: `Start time (${formatTime(startTime)}) exceeds track duration (${formatTime(trackDuration)})`
      };
    }
  }
  
  // Use current track if no track specified (Requirement 1.3)
  const trackId = currentTrackId ?? null;
  
  return {
    trackId,
    trackName: null,
    startTime,
    endTime,
    originalText: query
  };
}

/**
 * Validate time range against track duration.
 * 
 * @param startTime Start time in seconds
 * @param endTime End time in seconds
 * @param trackDuration Track duration in seconds
 * @returns Validated [startTime, endTime] tuple with endTime clamped if needed
 */
export function validateTimeRange(
  startTime: number,
  endTime: number,
  trackDuration: number
): ParseResult {
  if (startTime < 0 || endTime < 0) {
    return {
      code: 'NEGATIVE_TIME',
      message: 'Time values cannot be negative'
    };
  }
  
  if (startTime >= endTime) {
    return {
      code: 'INVALID_RANGE',
      message: 'Start time must be less than end time'
    };
  }
  
  // Clamp end_time to track duration
  const clampedEnd = Math.min(endTime, trackDuration);
  
  if (startTime >= trackDuration) {
    return {
      code: 'INVALID_RANGE',
      message: `Start time exceeds track duration`
    };
  }
  
  return {
    trackId: null,
    trackName: null,
    startTime,
    endTime: clampedEnd,
    originalText: ''
  };
}
