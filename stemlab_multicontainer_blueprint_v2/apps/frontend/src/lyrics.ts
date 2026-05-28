export type LyricWord = {
  start: number;
  end: number;
  text: string;
  confidence?: number;
};

export type LyricSegment = {
  start: number;
  end: number;
  text: string;
  words: LyricWord[];
};

export type NormalizedLyrics = {
  source: string;
  language?: string | null;
  text: string;
  segments: LyricSegment[];
  segmentCount: number;
  wordCount: number;
};

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function normalizeJsonPayload(payload: unknown): NormalizedLyrics {
  const data = typeof payload === "object" && payload !== null ? (payload as Record<string, unknown>) : {};
  const rawSegments = Array.isArray(data.segments) ? (data.segments as Array<Record<string, unknown>>) : [];
  const segments: LyricSegment[] = rawSegments
    .map((segment) => {
      const rawWords = Array.isArray(segment.words)
        ? (segment.words as Array<Record<string, unknown>>)
        : [];
      const words = rawWords
        .map((word) => ({
          start: asNumber(word.start),
          end: asNumber(word.end),
          text: String(word.text ?? word.word ?? "").trim(),
          confidence:
            typeof word.confidence === "number" ? word.confidence : undefined,
        }))
        .filter((word) => word.text.length > 0);
      const text = String(segment.text ?? "").trim() || words.map((word) => word.text).join(" ");
      return {
        start: asNumber(segment.start),
        end: asNumber(segment.end),
        text,
        words,
      };
    })
    .filter((segment) => segment.text.length > 0 || segment.words.length > 0);

  const fullText =
    typeof data.text === "string" && data.text.trim().length > 0
      ? data.text
      : segments.map((segment) => segment.text).join("\n");

  return {
    source: typeof data.source === "string" ? data.source : "lyrics-json",
    language: typeof data.language === "string" ? data.language : null,
    text: fullText,
    segments,
    segmentCount: segments.length,
    wordCount: segments.reduce((sum, segment) => sum + segment.words.length, 0),
  };
}

function parseLrc(text: string): NormalizedLyrics {
  const segments: LyricSegment[] = [];
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    const match = line.match(/^\[(\d+):(\d+(?:\.\d+)?)\](.*)$/);
    if (!match) continue;
    const start = Number(match[1]) * 60 + Number(match[2]);
    segments.push({
      start,
      end: start + 2.5,
      text: match[3].trim(),
      words: [],
    });
  }
  for (let index = 0; index < segments.length - 1; index += 1) {
    segments[index].end = Math.max(segments[index].start, segments[index + 1].start - 0.01);
  }
  return {
    source: "lrc",
    language: null,
    text: segments.map((segment) => segment.text).join("\n"),
    segments,
    segmentCount: segments.length,
    wordCount: 0,
  };
}

function parseSrt(text: string): NormalizedLyrics {
  const blocks = text.split(/\r?\n\r?\n/);
  const segments: LyricSegment[] = [];
  for (const block of blocks) {
    const lines = block.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (lines.length < 2) continue;
    const timeLine = lines.find((line) => line.includes("-->"));
    if (!timeLine) continue;
    const timeMatch = timeLine.match(
      /(\d\d):(\d\d):(\d\d),(\d\d\d)\s+-->\s+(\d\d):(\d\d):(\d\d),(\d\d\d)/
    );
    if (!timeMatch) continue;
    const start =
      Number(timeMatch[1]) * 3600 +
      Number(timeMatch[2]) * 60 +
      Number(timeMatch[3]) +
      Number(timeMatch[4]) / 1000;
    const end =
      Number(timeMatch[5]) * 3600 +
      Number(timeMatch[6]) * 60 +
      Number(timeMatch[7]) +
      Number(timeMatch[8]) / 1000;
    const content = lines.filter((line) => !/^\d+$/.test(line) && !line.includes("-->"));
    segments.push({ start, end, text: content.join(" "), words: [] });
  }
  return {
    source: "srt",
    language: null,
    text: segments.map((segment) => segment.text).join("\n"),
    segments,
    segmentCount: segments.length,
    wordCount: 0,
  };
}

function parsePlainText(text: string): NormalizedLyrics {
  const cleaned = text.trim();
  return {
    source: "plain-text",
    language: null,
    text: cleaned,
    segments: cleaned
      ? [
          {
            start: 0,
            end: 0,
            text: cleaned,
            words: [],
          },
        ]
      : [],
    segmentCount: cleaned ? 1 : 0,
    wordCount: 0,
  };
}

export async function loadLyricsFromUrl(url: string): Promise<NormalizedLyrics> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load lyrics artifact: ${response.status}`);
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json") || url.endsWith(".json")) {
    return normalizeJsonPayload(await response.json());
  }
  const text = await response.text();
  if (url.endsWith(".lrc")) return parseLrc(text);
  if (url.endsWith(".srt") || url.endsWith(".vtt")) return parseSrt(text);
  return parsePlainText(text);
}

export function findActiveSegmentIndex(segments: LyricSegment[], currentTime: number): number {
  return segments.findIndex((segment) => currentTime >= segment.start && currentTime <= segment.end);
}

export function findActiveWordIndex(segment: LyricSegment | null, currentTime: number): number {
  if (!segment) return -1;
  return segment.words.findIndex((word) => currentTime >= word.start && currentTime <= word.end);
}
