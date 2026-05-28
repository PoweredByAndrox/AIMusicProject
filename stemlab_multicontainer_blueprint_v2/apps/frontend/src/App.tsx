import { useEffect, useMemo, useRef, useState } from "react";
import { fetchJson, mediaUrl, postForm, postJson } from "./api";
import {
  findActiveSegmentIndex,
  findActiveWordIndex,
  loadLyricsFromUrl,
  type NormalizedLyrics,
} from "./lyrics";

type Service = { id: string; url: string };
type ServicesResponse = { services: Service[] };
type PresetsResponse = { presets: string[] };

type JobRequest = {
  track_name: string;
  input_path: string;
  preset_name: string;
  notes?: string | null;
  enable_lyrics?: boolean;
  lyrics_language?: string | null;
  lyrics_reference_path?: string | null;
};

type Job = {
  job_id: string;
  status: string;
  output_root: string;
  request: JobRequest;
  artifacts: Record<string, string>;
  invocations: Array<{
    phase_id: string;
    stem?: string | null;
    service: string;
    task: string;
    model?: string | null;
    preset?: string | null;
    prompt?: string | null;
    input_bus?: string | null;
  }>;
  run_log: Array<Record<string, unknown>>;
};

type JobsResponse = { jobs: Job[] };

type Preset = {
  name: string;
  description: string;
  targets: string[];
  phases: Array<{
    id: string;
    label: string;
    steps: Array<Record<string, unknown>>;
  }>;
};

type Artifact = {
  key: string;
  path: string;
  relative_path?: string | null;
  url?: string | null;
  kind: string;
  exists: boolean;
  size_bytes: number;
  source: string;
};

type ArtifactsResponse = {
  job_id: string;
  artifacts: Artifact[];
};

type UploadResponse = {
  filename: string;
  container_path: string;
  media_url: string;
  size_bytes: number;
};

type UploadPackResponse = {
  container_root: string;
  count: number;
  files: UploadResponse[];
};

type VirtualDjCompareItem = {
  role: string;
  role_label: string;
  note: string;
  recommended_pipeline_keys: string[];
  baseline_artifact: Artifact;
  pipeline_artifact: Artifact | null;
};

type VirtualDjCompareResponse = {
  baseline_root: string;
  resolved_baseline_root: string;
  job_id: string | null;
  checklist: {
    what_to_export: string[];
    preferred_format: string[];
    how_to_compare: string[];
  };
  items: VirtualDjCompareItem[];
  summary: {
    file_count: number;
    detected_roles: string[];
    missing_recommended_roles: string[];
  };
};

type TandemSearchRequest = {
  base_job_id: string;
  max_candidates: number;
  lyrics_validate_top_k: number;
  reference_instrumental_path?: string | null;
  force_reference_target?: boolean;
};

type SearchFocusWindow = {
  start: number;
  end: number;
  text: string;
  difficulty_score: number;
};

type SearchMetrics = {
  final_mix_vocal_bleed_db: number;
  demucs_mix_vocal_bleed_db: number;
  warble_centroid_ratio: number;
  warble_hi4_ratio: number;
  warble_flux_ratio: number;
  final_source_corr: number;
  final_diff_vs_source_rms_db: number;
  openness_centroid_delta_hz: number;
  openness_hi4_delta: number;
  lyrics_coverage_ratio: number;
  lyrics_focus_ratio: number;
  lyrics_high_activity_coverage_ratio: number;
  reference_usable?: boolean;
  reference_diff_vs_rms_db?: number;
  reference_corr?: number;
};

type SearchCandidate = {
  slug: string;
  label: string;
  hybrid_profile: string;
  other_recipe: string;
  final_recipe: string | null;
};

type SearchResultItem = {
  candidate: SearchCandidate;
  job_id: string;
  status: string;
  score: number;
  issues: string[];
  metrics: SearchMetrics;
  mode: string;
  worst_windows?: SearchFocusWindow[];
  coarse?: {
    job_id: string;
    score: number;
  };
  validation?: {
    job_id: string;
    score: number;
  };
};

type SearchResultPayload = {
  search_id: string;
  base_job_id: string;
  base_preset_name: string;
  input_path: string;
  focus_windows: SearchFocusWindow[];
  results: SearchResultItem[];
  top_candidates: SearchResultItem[];
};

type SearchStatus = {
  search_id: string;
  status: string;
  request: TandemSearchRequest;
  output_root: string;
  created_at: string;
  updated_at: string;
  result?: SearchResultPayload | null;
  stdout_tail?: string | null;
  stderr_tail?: string | null;
  error?: string | null;
};

type SearchesResponse = {
  searches: SearchStatus[];
};

const defaultInputPath = "/app/data/inputs/demo_vocals.wav";
// Switched to the residual-other + center-hard tandem variant in S3 because
// vocal_bleed.final_vocals.bleed_db_rel_total drops by ~5.4 dB on the
// validation track (-22.71 dB -> -28.11 dB) compared to the legacy default,
// which was bottlenecked by Demucs base bleed and the BS overlay safety-net.
// The legacy preset is still selectable for "openness over cleanliness" use.
const defaultPreset = "lyrics_sync_from_track_tandem_residual_other_center_hard";
const ACTIVE_JOB_STATUSES = new Set(["queued", "running"]);
const RECENT_COMPLETED_LIMIT = 4;
const defaultVirtualDjRoot = "";

function formatBytes(bytes: number): string {
  if (bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function isRenderedAudioArtifact(artifact: Artifact): boolean {
  return artifact.exists && artifact.kind.startsWith("audio") && artifact.source !== "request";
}

function chooseDefaultAudioArtifact(artifacts: Artifact[]): Artifact | null {
  const existingAudio = artifacts.filter(isRenderedAudioArtifact);
  const preferredKeys = ["vocals", "instrumental_mix", "drums", "bass", "guitar", "piano", "other"];
  for (const key of preferredKeys) {
    const match = existingAudio.find((artifact) => artifact.key === key || artifact.key.endsWith(`/${key}.wav`));
    if (match) return match;
  }
  return existingAudio[0] ?? null;
}

function chooseDefaultLyricsArtifact(artifacts: Artifact[]): Artifact | null {
  const existingLyrics = artifacts.filter(
    (artifact) => artifact.exists && (artifact.kind === "lyrics-json" || artifact.kind === "lyrics-text")
  );
  const preferredKeys = ["lyrics_json", "lyrics_lrc", "lyrics_srt", "lyrics_txt"];
  for (const key of preferredKeys) {
    const match = existingLyrics.find(
      (artifact) =>
        artifact.key === key ||
        artifact.key.startsWith("lyrics_") ||
        artifact.key.endsWith(`/${key.split("_").slice(-1)[0]}.${artifact.path.split(".").pop()}`)
    );
    if (match) return match;
  }
  return existingLyrics[0] ?? null;
}

function isSameArtifact(left: Artifact | null, right: Artifact | null): boolean {
  if (!left || !right) return left === right;
  return left.key === right.key && left.path === right.path && left.url === right.url;
}

function findArtifactByKey(artifacts: Artifact[], key: string): Artifact | null {
  return artifacts.find((artifact) => artifact.key === key) ?? null;
}

function chooseDefaultComparePair(
  artifacts: Artifact[],
  preferred: Artifact | null
): { left: Artifact | null; right: Artifact | null } {
  const audioArtifacts = artifacts.filter((artifact) => artifact.exists && artifact.kind.startsWith("audio"));
  if (audioArtifacts.length < 2) {
    return {
      left: audioArtifacts[0] ?? null,
      right: null,
    };
  }

  const preferredPairs = [
    ["instrumental_mix", "demucs_instrumental_mix"],
    ["vocals", "demucs_vocals"],
    ["other", "demucs_other"],
    ["drums", "demucs_drums"],
    ["bass", "demucs_bass"],
    ["guitar", "demucs_guitar"],
    ["piano", "demucs_piano"],
  ] as const;

  if (preferred) {
    for (const [leftKey, rightKey] of preferredPairs) {
      if (preferred.key === leftKey || preferred.key === rightKey) {
        const left = findArtifactByKey(audioArtifacts, leftKey) ?? preferred;
        const right = findArtifactByKey(audioArtifacts, rightKey) ?? audioArtifacts.find((artifact) => artifact.key !== left.key) ?? null;
        return { left, right };
      }
    }
  }

  for (const [leftKey, rightKey] of preferredPairs) {
    const left = findArtifactByKey(audioArtifacts, leftKey);
    const right = findArtifactByKey(audioArtifacts, rightKey);
    if (left && right) {
      return { left, right };
    }
  }

  return {
    left: audioArtifacts[0] ?? null,
    right: audioArtifacts.find((artifact) => artifact.key !== audioArtifacts[0].key) ?? null,
  };
}

function describePresetBehavior(presetName: string): string {
  if (presetName === "lyrics_sync_from_vocals") {
    return "This preset is lyrics-only. It does not render stem audio. It expects an already isolated vocals input.";
  }
  if (presetName === "lyrics_sync_from_track_demucs") {
    return "This preset first extracts a vocals stem with Demucs, then runs lyrics transcription on the rendered vocals.";
  }
  if (presetName === "lyrics_sync_from_track_demucs_tuned") {
    return "This preset extracts 6 stems with Demucs, then runs a real presence-lift / de-muffling pass before lyrics transcription.";
  }
  if (presetName === "lyrics_sync_from_track_demucs_open_plus") {
    return "This preset is a stronger anti-muffle variant of the tuned Demucs pipeline. Use it when the tuned pass still feels veiled, especially on other/instrumental stems, but expect a higher risk of extra brightness or edge.";
  }
  if (presetName === "lyrics_sync_from_track_tandem") {
    return "Legacy tandem preset (kept as a fallback). Demucs base + BS-RoFormer overlay with the standard center suppression. Slightly more open top-end than the new default at the cost of ~5 dB more vocal bleed in the instrumental stem.";
  }
  if (presetName === "lyrics_sync_from_track_tandem_residual_other_center_hard") {
    return "Default tandem preset (S3). Demucs base + BS-RoFormer overlay with rebuilt residual `other` and the `hard` center-vocal suppression profile. The S2 overlay safety-net (bs_overlay_safety_margin_db, default 3 dB) automatically zeroes out any BS contribution that is more vocal-contaminated than the Demucs base, so the hybrid never makes the bleed worse. Validated: vocal_bleed.final_vocals.bleed_db_rel_total ~= -28 dB on the Nine Thou test, vs ~-22 dB for the legacy preset.";
  }
  return "Inspect targets and phases carefully. If there are no audio targets, the job should not be treated as a stem-render result.";
}

export default function App() {
  const [services, setServices] = useState<Service[]>([]);
  const [presets, setPresets] = useState<string[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>(defaultPreset);
  const [preset, setPreset] = useState<Preset | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [selectedAudioArtifact, setSelectedAudioArtifact] = useState<Artifact | null>(null);
  const [selectedLyricsArtifact, setSelectedLyricsArtifact] = useState<Artifact | null>(null);
  const [compareArtifactA, setCompareArtifactA] = useState<Artifact | null>(null);
  const [compareArtifactB, setCompareArtifactB] = useState<Artifact | null>(null);
  const [compareActiveSide, setCompareActiveSide] = useState<"A" | "B">("A");
  const [compareCurrentTime, setCompareCurrentTime] = useState(0);
  const [compareDuration, setCompareDuration] = useState(0);
  const [compareIsPlaying, setCompareIsPlaying] = useState(false);
  const [compareVolume, setCompareVolume] = useState(0.1);
  const [lyricsData, setLyricsData] = useState<NormalizedLyrics | null>(null);
  const [trackName, setTrackName] = useState("Demo Vocals Stem");
  const [inputPath, setInputPath] = useState(defaultInputPath);
  const [notes, setNotes] = useState("Local test separation / lyrics sync");
  const [enableLyrics, setEnableLyrics] = useState(true);
  const [lyricsLanguage, setLyricsLanguage] = useState("en");
  const [lyricsReferencePath, setLyricsReferencePath] = useState("");
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [uploadingAudio, setUploadingAudio] = useState(false);
  const [uploadingText, setUploadingText] = useState(false);
  const [uploadingVirtualDjPack, setUploadingVirtualDjPack] = useState(false);
  const [scanningVirtualDj, setScanningVirtualDj] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [showAllJobs, setShowAllJobs] = useState(false);
  const [virtualDjRoot, setVirtualDjRoot] = useState(defaultVirtualDjRoot);
  const [virtualDjCompare, setVirtualDjCompare] = useState<VirtualDjCompareResponse | null>(null);
  const [searches, setSearches] = useState<SearchStatus[]>([]);
  const [selectedSearchId, setSelectedSearchId] = useState("");
  const [searchMaxCandidates, setSearchMaxCandidates] = useState(5);
  const [searchLyricsValidateTopK, setSearchLyricsValidateTopK] = useState(2);
  const [startingSearch, setStartingSearch] = useState(false);
  const [searchReferenceInstrumentalPath, setSearchReferenceInstrumentalPath] = useState("");
  const [searchForceReferenceTarget, setSearchForceReferenceTarget] = useState(true);
  const [uploadingSearchReference, setUploadingSearchReference] = useState(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const compareAudioARef = useRef<HTMLAudioElement | null>(null);
  const compareAudioBRef = useRef<HTMLAudioElement | null>(null);

  async function reloadJobs(): Promise<void> {
    const jobsData = await fetchJson<JobsResponse>("/api/v1/jobs");
    setJobs(jobsData.jobs);
    if (!selectedJobId && jobsData.jobs.length > 0) {
      setSelectedJobId(jobsData.jobs[0].job_id);
    }
  }

  async function reloadSearches(): Promise<void> {
    const searchData = await fetchJson<SearchesResponse>("/api/v1/searches");
    setSearches(searchData.searches);
    if (!selectedSearchId && searchData.searches.length > 0) {
      setSelectedSearchId(searchData.searches[0].search_id);
    }
  }

  async function reloadAll(): Promise<void> {
    setLoading(true);
    setError(null);
    try {
      const [servicesData, presetsData] = await Promise.all([
        fetchJson<ServicesResponse>("/api/v1/catalog/services"),
        fetchJson<PresetsResponse>("/api/v1/catalog/presets"),
      ]);
      setServices(servicesData.services);
      setPresets(presetsData.presets);
      const presetName = presetsData.presets.includes(selectedPreset)
        ? selectedPreset
        : presetsData.presets.includes(defaultPreset)
          ? defaultPreset
          : presetsData.presets[0] ?? defaultPreset;
      setSelectedPreset(presetName);
      if (presetName) {
        setPreset(await fetchJson<Preset>(`/api/v1/catalog/presets/${presetName}`));
      }
      await Promise.all([reloadJobs(), reloadSearches()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown API error");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void reloadAll();
  }, []);

  useEffect(() => {
    if (!selectedPreset) return;
    void fetchJson<Preset>(`/api/v1/catalog/presets/${selectedPreset}`)
      .then(setPreset)
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Preset load error");
      });
  }, [selectedPreset]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void Promise.all([reloadJobs(), reloadSearches()]).catch(() => {
        // keep polling quiet
      });
    }, 4000);
    return () => window.clearInterval(timer);
  }, [selectedJobId]);

  useEffect(() => {
    if (!selectedJobId) {
      setArtifacts([]);
      setSelectedAudioArtifact(null);
      return;
    }

    void fetchJson<ArtifactsResponse>(`/api/v1/jobs/${selectedJobId}/artifacts`)
      .then((data) => {
        setArtifacts(data.artifacts);
        setSelectedAudioArtifact((prev) => {
          if (prev) {
            const match = data.artifacts.find((artifact) => artifact.key === prev.key);
            if (match) return isSameArtifact(prev, match) ? prev : match;
          }
          return chooseDefaultAudioArtifact(data.artifacts);
        });
        setSelectedLyricsArtifact((prev) => {
          if (prev) {
            const match = data.artifacts.find((artifact) => artifact.key === prev.key);
            if (match) return isSameArtifact(prev, match) ? prev : match;
          }
          return chooseDefaultLyricsArtifact(data.artifacts);
        });
        const compareDefaults = chooseDefaultComparePair(data.artifacts, chooseDefaultAudioArtifact(data.artifacts));
        setCompareArtifactA((prev) => {
          if (prev) {
            const match = data.artifacts.find((artifact) => artifact.key === prev.key);
            if (match) return isSameArtifact(prev, match) ? prev : match;
          }
          return compareDefaults.left;
        });
        setCompareArtifactB((prev) => {
          if (prev) {
            const match = data.artifacts.find((artifact) => artifact.key === prev.key);
            if (match) return isSameArtifact(prev, match) ? prev : match;
          }
          return compareDefaults.right;
        });
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Artifact load error");
      });
  }, [selectedJobId, jobs]);

  useEffect(() => {
    const url = mediaUrl(selectedLyricsArtifact?.url);
    if (!url) {
      setLyricsData(null);
      return;
    }

    let active = true;
    void loadLyricsFromUrl(url)
      .then((data) => {
        if (active) setLyricsData(data);
      })
      .catch((err: unknown) => {
        if (active) setError(err instanceof Error ? err.message : "Lyrics load error");
      });

    return () => {
      active = false;
    };
  }, [selectedLyricsArtifact]);

  useEffect(() => {
    if (!virtualDjCompare || !virtualDjRoot || !selectedJobId) return;
    void scanVirtualDjBaseline(virtualDjRoot);
  }, [selectedJobId]);

  const selectedJob = useMemo(
    () => jobs.find((job) => job.job_id === selectedJobId) ?? null,
    [jobs, selectedJobId]
  );
  const selectedSearch = useMemo(
    () => searches.find((search) => search.search_id === selectedSearchId) ?? null,
    [searches, selectedSearchId]
  );
  const searchesForSelectedJob = useMemo(
    () => searches.filter((search) => search.request.base_job_id === selectedJobId),
    [searches, selectedJobId]
  );
  const visibleJobs = useMemo(() => {
    if (showAllJobs) {
      return jobs;
    }

    const activeJobs = jobs.filter((job) => ACTIVE_JOB_STATUSES.has(job.status));
    const recentFinishedJobs = jobs.filter((job) => !ACTIVE_JOB_STATUSES.has(job.status)).slice(0, RECENT_COMPLETED_LIMIT);
    const base = [...activeJobs, ...recentFinishedJobs];
    const deduped = new Map<string, Job>();

    for (const job of base) {
      deduped.set(job.job_id, job);
    }

    const selected = jobs.find((job) => job.job_id === selectedJobId);
    if (selected && !deduped.has(selected.job_id)) {
      deduped.set(selected.job_id, selected);
    }

    return Array.from(deduped.values());
  }, [jobs, selectedJobId, showAllJobs]);
  const presetHasRenderedAudioTargets = useMemo(
    () => Boolean(preset?.targets.some((target) => !target.startsWith("lyrics_"))),
    [preset]
  );
  const renderedAudioArtifacts = useMemo(
    () => artifacts.filter(isRenderedAudioArtifact),
    [artifacts]
  );
  const primaryArtifacts = useMemo(
    () => artifacts.filter((artifact) => artifact.source === "job-index"),
    [artifacts]
  );
  const requestArtifacts = useMemo(
    () => artifacts.filter((artifact) => artifact.source === "request"),
    [artifacts]
  );
  const scanArtifacts = useMemo(
    () => artifacts.filter((artifact) => artifact.source === "scan"),
    [artifacts]
  );
  const availableCompareArtifacts = useMemo(
    () => artifacts.filter((artifact) => artifact.exists && artifact.kind.startsWith("audio") && artifact.url),
    [artifacts]
  );
  const quickComparePairs = useMemo(() => {
    const pairs = [
      {
        label: "Final vs Demucs instrumental",
        left: findArtifactByKey(availableCompareArtifacts, "instrumental_mix"),
        right: findArtifactByKey(availableCompareArtifacts, "demucs_instrumental_mix"),
      },
      {
        label: "Final vs Demucs vocals",
        left: findArtifactByKey(availableCompareArtifacts, "vocals"),
        right: findArtifactByKey(availableCompareArtifacts, "demucs_vocals"),
      },
      {
        label: "Tandem other vs Demucs other",
        left: findArtifactByKey(availableCompareArtifacts, "other"),
        right: findArtifactByKey(availableCompareArtifacts, "demucs_other"),
      },
    ];
    return pairs.filter((pair): pair is { label: string; left: Artifact; right: Artifact } => Boolean(pair.left && pair.right));
  }, [availableCompareArtifacts]);

  const selectedAudioUrl = mediaUrl(selectedAudioArtifact?.url);
  const compareAudioUrlA = mediaUrl(compareArtifactA?.url);
  const compareAudioUrlB = mediaUrl(compareArtifactB?.url);
  const activeSegmentIndex = useMemo(
    () => (lyricsData ? findActiveSegmentIndex(lyricsData.segments, currentTime) : -1),
    [lyricsData, currentTime]
  );
  const activeSegment = activeSegmentIndex >= 0 && lyricsData ? lyricsData.segments[activeSegmentIndex] : null;
  const activeWordIndex = findActiveWordIndex(activeSegment, currentTime);

  useEffect(() => {
    if (!selectedJobId) return;
    const matchingSearch = searches.find((search) => search.request.base_job_id === selectedJobId);
    if (matchingSearch) {
      setSelectedSearchId((current) => (current && searches.some((search) => search.search_id === current) ? current : matchingSearch.search_id));
    }
  }, [selectedJobId, searches]);

  useEffect(() => {
    const playerA = compareAudioARef.current;
    const playerB = compareAudioBRef.current;
    if (!playerA || !playerB) return;
    playerA.muted = compareActiveSide !== "A";
    playerB.muted = compareActiveSide !== "B";
  }, [compareActiveSide, compareAudioUrlA, compareAudioUrlB]);

  useEffect(() => {
    const playerA = compareAudioARef.current;
    const playerB = compareAudioBRef.current;
    if (playerA) playerA.volume = compareVolume;
    if (playerB) playerB.volume = compareVolume;
  }, [compareVolume, compareAudioUrlA, compareAudioUrlB]);

  useEffect(() => {
    setCompareCurrentTime(0);
    setCompareDuration(0);
    setCompareIsPlaying(false);
    setCurrentTime(0);
    const playerA = compareAudioARef.current;
    const playerB = compareAudioBRef.current;
    if (playerA) playerA.pause();
    if (playerB) playerB.pause();
  }, [selectedJobId, compareAudioUrlA, compareAudioUrlB]);

  function syncComparePlayers(targetTime: number): void {
    const playerA = compareAudioARef.current;
    const playerB = compareAudioBRef.current;
    if (playerA && Number.isFinite(playerA.duration) && Math.abs(playerA.currentTime - targetTime) > 0.05) {
      playerA.currentTime = targetTime;
    }
    if (playerB && Number.isFinite(playerB.duration) && Math.abs(playerB.currentTime - targetTime) > 0.05) {
      playerB.currentTime = targetTime;
    }
    setCompareCurrentTime(targetTime);
    setCurrentTime(targetTime);
  }

  function updateCompareDuration(): void {
    const playerA = compareAudioARef.current;
    const playerB = compareAudioBRef.current;
    const durations = [playerA?.duration ?? 0, playerB?.duration ?? 0].filter((value) => Number.isFinite(value) && value > 0);
    setCompareDuration(durations.length > 0 ? Math.min(...durations) : 0);
  }

  function handleCompareTimeUpdate(side: "A" | "B"): void {
    if (compareActiveSide !== side) return;
    const activePlayer = side === "A" ? compareAudioARef.current : compareAudioBRef.current;
    const inactivePlayer = side === "A" ? compareAudioBRef.current : compareAudioARef.current;
    const nextTime = activePlayer?.currentTime ?? 0;
    setCompareCurrentTime(nextTime);
    setCurrentTime(nextTime);
    if (inactivePlayer && Math.abs(inactivePlayer.currentTime - nextTime) > 0.08) {
      inactivePlayer.currentTime = nextTime;
    }
  }

  async function setComparePlayback(shouldPlay: boolean): Promise<void> {
    const playerA = compareAudioARef.current;
    const playerB = compareAudioBRef.current;
    if (!playerA || !playerB || !compareAudioUrlA || !compareAudioUrlB) return;

    syncComparePlayers(compareCurrentTime);
    playerA.muted = compareActiveSide !== "A";
    playerB.muted = compareActiveSide !== "B";

    if (!shouldPlay) {
      playerA.pause();
      playerB.pause();
      setCompareIsPlaying(false);
      return;
    }

    await Promise.all([
      playerA.play().catch(() => undefined),
      playerB.play().catch(() => undefined),
    ]);
    setCompareIsPlaying(true);
  }

  function switchCompareSide(side: "A" | "B"): void {
    const activePlayer = compareActiveSide === "A" ? compareAudioARef.current : compareAudioBRef.current;
    const targetTime = activePlayer?.currentTime ?? compareCurrentTime;
    syncComparePlayers(targetTime);
    setCompareActiveSide(side);
  }

  function loadComparePair(left: Artifact | null, right: Artifact | null): void {
    setCompareArtifactA(left);
    setCompareArtifactB(right);
    setCompareActiveSide("A");
    setCompareCurrentTime(0);
    setCompareDuration(0);
    setCompareIsPlaying(false);
  }

  async function uploadAudioFile(file: File): Promise<void> {
    setUploadingAudio(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const result = await postForm<UploadResponse>("/api/v1/uploads/audio", formData);
      setInputPath(result.container_path);
      if (!trackName || trackName === "Demo Vocals Stem") {
        setTrackName(file.name.replace(/\.[^.]+$/, ""));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Audio upload failed");
    } finally {
      setUploadingAudio(false);
    }
  }

  async function uploadTextFile(file: File): Promise<void> {
    setUploadingText(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const result = await postForm<UploadResponse>("/api/v1/uploads/text", formData);
      setLyricsReferencePath(result.container_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Lyrics upload failed");
    } finally {
      setUploadingText(false);
    }
  }

  async function uploadSearchReferenceFile(file: File): Promise<void> {
    setUploadingSearchReference(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const result = await postForm<UploadResponse>("/api/v1/uploads/audio", formData);
      setSearchReferenceInstrumentalPath(result.container_path);
      setSearchForceReferenceTarget(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Reference instrumental upload failed");
    } finally {
      setUploadingSearchReference(false);
    }
  }

  async function scanVirtualDjBaseline(rootOverride?: string): Promise<void> {
    const targetRoot = (rootOverride ?? virtualDjRoot).trim();
    if (!targetRoot) {
      setVirtualDjCompare(null);
      return;
    }
    setScanningVirtualDj(true);
    setError(null);
    try {
      const query = new URLSearchParams({ baseline_root: targetRoot });
      if (selectedJobId) {
        query.set("job_id", selectedJobId);
      }
      const result = await fetchJson<VirtualDjCompareResponse>(`/api/v1/compare/virtualdj?${query.toString()}`);
      setVirtualDjRoot(result.baseline_root);
      setVirtualDjCompare(result);
    } catch (err) {
      setVirtualDjCompare(null);
      setError(err instanceof Error ? err.message : "VirtualDJ compare scan failed");
    } finally {
      setScanningVirtualDj(false);
    }
  }

  async function uploadVirtualDjPack(files: FileList | null): Promise<void> {
    if (!files || files.length === 0) return;
    setUploadingVirtualDjPack(true);
    setError(null);
    try {
      const formData = new FormData();
      for (const file of Array.from(files)) {
        formData.append("files", file);
      }
      formData.append("folder_name", `${trackName || "virtualdj"}_baseline`);
      const result = await postForm<UploadPackResponse>("/api/v1/uploads/audio-pack", formData);
      setVirtualDjRoot(result.container_root);
      await scanVirtualDjBaseline(result.container_root);
    } catch (err) {
      setError(err instanceof Error ? err.message : "VirtualDJ stem pack upload failed");
    } finally {
      setUploadingVirtualDjPack(false);
    }
  }

  async function createJob(): Promise<void> {
    setCreating(true);
    setError(null);
    try {
      const created = await postJson<Job>("/api/v1/jobs", {
        track_name: trackName,
        input_path: inputPath,
        preset_name: selectedPreset,
        notes,
        enable_lyrics: enableLyrics,
        lyrics_language: lyricsLanguage || null,
        lyrics_reference_path: lyricsReferencePath || null,
      });
      setSelectedJobId(created.job_id);
      await reloadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Job create error");
    } finally {
      setCreating(false);
    }
  }

  async function startPresetSearch(): Promise<void> {
    if (!selectedJob) return;
    setStartingSearch(true);
    setError(null);
    try {
      const created = await postJson<SearchStatus>("/api/v1/searches/tandem", {
        base_job_id: selectedJob.job_id,
        max_candidates: searchMaxCandidates,
        lyrics_validate_top_k: searchLyricsValidateTopK,
        reference_instrumental_path: searchReferenceInstrumentalPath.trim() || null,
        force_reference_target: Boolean(searchReferenceInstrumentalPath.trim()) && searchForceReferenceTarget,
      });
      setSelectedSearchId(created.search_id);
      await reloadSearches();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Preset search start failed");
    } finally {
      setStartingSearch(false);
    }
  }

  function seekTo(seconds: number): void {
    if (compareIsPlaying && compareAudioUrlA && compareAudioUrlB) {
      syncComparePlayers(seconds);
      if (compareIsPlaying) {
        void setComparePlayback(true);
      }
      return;
    }
    if (!audioRef.current) return;
    audioRef.current.currentTime = seconds;
    void audioRef.current.play().catch(() => {
      // autoplay may be blocked
    });
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <h1>StemLab</h1>
          <p>Single UI for multi-container separation, stem preview, and lyrics timing review.</p>
        </div>
        <button onClick={() => void reloadAll()} className="secondary">
          Refresh
        </button>
      </header>

      {error && <div className="error-box">{error}</div>}

      <section className="grid two">
        <article className="card">
          <h2>Create job</h2>
          <label>
            Track name
            <input value={trackName} onChange={(e) => setTrackName(e.target.value)} />
          </label>

          <label>
            Audio path inside containers
            <input value={inputPath} onChange={(e) => setInputPath(e.target.value)} />
          </label>

          <label className="upload-box">
            Upload audio
            <input
              type="file"
              accept="audio/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void uploadAudioFile(file);
                e.currentTarget.value = "";
              }}
            />
            <span>{uploadingAudio ? "Uploading..." : "Supported: wav / flac / mp3 / m4a / ogg"}</span>
          </label>

          <label>
            Preset
            <select value={selectedPreset} onChange={(e) => setSelectedPreset(e.target.value)}>
              {presets.map((presetName) => (
                <option key={presetName} value={presetName}>
                  {presetName}
                </option>
              ))}
            </select>
          </label>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={enableLyrics}
              onChange={(e) => setEnableLyrics(e.target.checked)}
            />
            <span>Enable lyrics worker for this job</span>
          </label>

          <label>
            Lyrics / ASR language
            <input
              value={lyricsLanguage}
              onChange={(e) => setLyricsLanguage(e.target.value)}
              placeholder="en / ru / de / ja ..."
            />
          </label>

          <label>
            Reference lyrics path (optional)
            <input
              value={lyricsReferencePath}
              onChange={(e) => setLyricsReferencePath(e.target.value)}
              placeholder="/app/data/inputs/song_lyrics.txt"
            />
          </label>

          <label className="upload-box">
            Upload lyrics.txt / .lrc / .srt
            <input
              type="file"
              accept=".txt,.lrc,.srt,.vtt,.ass,text/plain"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void uploadTextFile(file);
                e.currentTarget.value = "";
              }}
            />
            <span>{uploadingText ? "Uploading..." : "Reference lyrics give much better exact-text alignment"}</span>
          </label>

          <label>
            Notes
            <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={4} />
          </label>

          <button disabled={creating || !inputPath} onClick={() => void createJob()}>
            {creating ? "Creating..." : "Create job"}
          </button>
        </article>

        <article className="card">
          <h2>Selected preset</h2>
          {loading && <p>Loading...</p>}
          {!loading && preset && (
            <>
              <p>
                <strong>{preset.name}</strong>
              </p>
              <p>{preset.description}</p>
              <div className="pill-row">
                {preset.targets.map((target) => (
                  <span key={target} className="pill">
                    {target}
                  </span>
                ))}
              </div>
              <p>Phases: {preset.phases.length}</p>
              <div className="hint-box">{describePresetBehavior(selectedPreset)}</div>
              {!presetHasRenderedAudioTargets && (
                <div className="warning-box">
                  This preset has no audio targets. It should not be treated as a stem-render preset.
                </div>
              )}
            </>
          )}
        </article>
      </section>

      <section className="grid two">
        <article className="card">
          <div className="section-toolbar">
            <div>
              <h2>Jobs</h2>
              <div className="muted">
                Showing {visibleJobs.length} of {jobs.length}
              </div>
            </div>
            {jobs.length > visibleJobs.length && (
              <button className="secondary" onClick={() => setShowAllJobs((value) => !value)}>
                {showAllJobs ? "Show active + recent" : `Show all ${jobs.length}`}
              </button>
            )}
          </div>
          <div className="job-list">
            {jobs.length === 0 && <p>No jobs yet.</p>}
            {visibleJobs.map((job) => (
              <button
                key={job.job_id}
                className={`job-card ${selectedJobId === job.job_id ? "job-card-active" : ""}`}
                onClick={() => setSelectedJobId(job.job_id)}
              >
                <div className="job-head">
                  <div>
                    <strong>{job.request.track_name}</strong>
                    <div className="muted">{job.job_id}</div>
                  </div>
                  <span className={`status status-${job.status}`}>{job.status}</span>
                </div>
                <p>Preset: {job.request.preset_name}</p>
                <p className="muted">Input: {job.request.input_path}</p>
                <p>Steps: {job.invocations.length}</p>
              </button>
            ))}
          </div>
        </article>

        <article className="card">
          <h2>Selected job</h2>
          {!selectedJob && <p>Select a job from the left.</p>}
          {selectedJob && (
            <>
              <p>
                <strong>{selectedJob.request.track_name}</strong>
              </p>
              <p>Preset: {selectedJob.request.preset_name}</p>
              <p>
                Status: <span className={`status status-${selectedJob.status}`}>{selectedJob.status}</span>
              </p>
              <p>
                Output root: <code>{selectedJob.output_root}</code>
              </p>
              <p>Lyrics enabled: {selectedJob.request.enable_lyrics ? "yes" : "no"}</p>
              <p>Lyrics language: {selectedJob.request.lyrics_language || "auto"}</p>
              <p>Reference lyrics: {selectedJob.request.lyrics_reference_path || "none"}</p>
              <details>
                <summary>Run log</summary>
                <pre>{JSON.stringify(selectedJob.run_log, null, 2)}</pre>
              </details>
            </>
          )}
        </article>
      </section>

      <section className="grid two-wide">
        <article className="card">
          <div className="section-toolbar">
            <div>
              <h2>Analyzer-guided preset search</h2>
              <div className="muted">Run auto-search on the selected completed job instead of hand-tweaking every preset.</div>
            </div>
            <button
              onClick={() => void startPresetSearch()}
              disabled={startingSearch || !selectedJob || selectedJob.status !== "completed"}
            >
              {startingSearch ? "Starting..." : "Run preset search"}
            </button>
          </div>

          {!selectedJob && <div className="warning-box">Select a completed job first. Search uses that job as the baseline and source input.</div>}
          {selectedJob && selectedJob.status !== "completed" && (
            <div className="warning-box">Search only makes sense on a completed job. Running or failed jobs are not valid baselines.</div>
          )}

          <div className="compare-columns">
            <label>
              Max candidates
              <input
                type="number"
                min={1}
                max={6}
                value={searchMaxCandidates}
                onChange={(e) => setSearchMaxCandidates(Number(e.target.value) || 1)}
              />
            </label>
            <label>
              Lyrics validation top-K
              <input
                type="number"
                min={0}
                max={3}
                value={searchLyricsValidateTopK}
                onChange={(e) => setSearchLyricsValidateTopK(Number(e.target.value) || 0)}
              />
            </label>
          </div>

          <label>
            Reference instrumental target (optional)
            <input
              value={searchReferenceInstrumentalPath}
              onChange={(e) => setSearchReferenceInstrumentalPath(e.target.value)}
              placeholder="/app/data/inputs/Nine Thou Instrumental.mp3"
            />
          </label>

          <label className="upload-box">
            Upload reference instrumental
            <input
              type="file"
              accept="audio/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void uploadSearchReferenceFile(file);
                e.currentTarget.value = "";
              }}
            />
            <span>{uploadingSearchReference ? "Uploading..." : "Use this when you want search to chase a specific original minus."}</span>
          </label>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={searchForceReferenceTarget}
              onChange={(e) => setSearchForceReferenceTarget(e.target.checked)}
              disabled={!searchReferenceInstrumentalPath.trim()}
            />
            <span>Force-match this reference even if analyzer thinks it may be a different master/version</span>
          </label>

          <div className="hint-box">
            Coarse stage ranks candidates by bleed, warble, source similarity, segment-local worst windows, and optional reference-minus deviation. Top-K then reruns with lyrics to add drift penalty.
          </div>

          <h3>Search runs</h3>
          <div className="job-list compact-list">
            {searchesForSelectedJob.length === 0 && <p className="muted">No searches for this job yet.</p>}
            {searchesForSelectedJob.map((search) => (
              <button
                key={search.search_id}
                className={`job-card ${selectedSearchId === search.search_id ? "job-card-active" : ""}`}
                onClick={() => setSelectedSearchId(search.search_id)}
              >
                <div className="job-head">
                    <div>
                      <strong>{search.search_id}</strong>
                    <div className="muted">
                      {search.request.max_candidates} candidates | lyrics top-K {search.request.lyrics_validate_top_k}
                      {search.request.reference_instrumental_path ? " | reference target" : ""}
                    </div>
                  </div>
                  <span className={`status status-${search.status}`}>{search.status}</span>
                </div>
                <p className="muted">Updated: {new Date(search.updated_at).toLocaleString()}</p>
              </button>
            ))}
          </div>
        </article>

        <article className="card">
          <h2>Search result</h2>
          {!selectedSearch && <p>Select a search run on the left.</p>}
          {selectedSearch && (
            <>
              <p>
                Status: <span className={`status status-${selectedSearch.status}`}>{selectedSearch.status}</span>
              </p>
              <p className="muted">Output root: {selectedSearch.output_root}</p>
              {selectedSearch.request.reference_instrumental_path && (
                <p className="muted">
                  Reference target: {selectedSearch.request.reference_instrumental_path}
                  {selectedSearch.request.force_reference_target ? " | forced" : ""}
                </p>
              )}
              {selectedSearch.error && <div className="error-box">{selectedSearch.error}</div>}

              {selectedSearch.result?.focus_windows && selectedSearch.result.focus_windows.length > 0 && (
                <>
                  <h3>Worst focus windows</h3>
                  <div className="compare-grid">
                    {selectedSearch.result.focus_windows.slice(0, 5).map((window, index) => (
                      <div key={`${window.start}-${window.end}-${index}`} className="compare-card">
                        <div className="artifact-topline">
                          <strong>
                            {formatTime(window.start)} - {formatTime(window.end)}
                          </strong>
                          <span className="pill small">difficulty {window.difficulty_score.toFixed(1)}</span>
                        </div>
                        <div className="muted">{window.text}</div>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {selectedSearch.result?.top_candidates && selectedSearch.result.top_candidates.length > 0 && (
                <>
                  <h3>Top candidates</h3>
                  <div className="compare-grid">
                    {selectedSearch.result.top_candidates.map((item) => (
                      <div key={`${item.candidate.slug}-${item.job_id}`} className="compare-card">
                        <div className="artifact-topline">
                          <strong>{item.candidate.slug}</strong>
                          <span className="pill small">{item.mode}</span>
                        </div>
                        <div className="muted">{item.candidate.label}</div>
                        <div className="metric-grid">
                          <div>score: {item.score.toFixed(2)}</div>
                          <div>bleed: {item.metrics.final_mix_vocal_bleed_db.toFixed(2)} dB</div>
                          <div>warble: {item.metrics.warble_centroid_ratio.toFixed(3)} / {item.metrics.warble_hi4_ratio.toFixed(3)}</div>
                          <div>open: +{item.metrics.openness_centroid_delta_hz.toFixed(1)} Hz</div>
                          {item.metrics.lyrics_coverage_ratio > 0 && (
                            <div>lyrics coverage: {item.metrics.lyrics_coverage_ratio.toFixed(3)}</div>
                          )}
                        </div>
                        <div className="hint-box compare-note">
                          {item.issues.length > 0 ? item.issues.join(", ") : "No flagged issues."}
                        </div>
                        <div className="artifact-actions">
                          <button onClick={() => setSelectedJobId(item.job_id)}>Open job</button>
                          {item.validation && item.validation.job_id !== item.job_id && (
                            <button className="secondary" onClick={() => setSelectedJobId(item.validation!.job_id)}>
                              Open validated job
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {selectedSearch.stdout_tail && (
                <details>
                  <summary>Search log</summary>
                  <pre>{selectedSearch.stdout_tail}</pre>
                </details>
              )}
            </>
          )}
        </article>
      </section>

      <section className="grid two-wide">
        <article className="card">
          <h2>Artifacts and preview</h2>
          {!selectedJob && <p>Select a job to inspect files.</p>}
          {selectedJob && (
            <>
              {availableCompareArtifacts.length >= 2 && (
                <div className="compare-player-card">
                  <div className="artifact-topline">
                    <div>
                      <strong>A/B compare</strong>
                      <div className="muted">One transport, instant A/B switch, no manual resync.</div>
                    </div>
                    <div className="pill-row">
                      <span className="pill small">A: {compareArtifactA?.key || "none"}</span>
                      <span className="pill small">B: {compareArtifactB?.key || "none"}</span>
                    </div>
                  </div>

                  {quickComparePairs.length > 0 && (
                    <div className="compare-quick-row">
                      {quickComparePairs.map((pair) => (
                        <button
                          key={pair.label}
                          className="secondary"
                          onClick={() => loadComparePair(pair.left, pair.right)}
                        >
                          {pair.label}
                        </button>
                      ))}
                    </div>
                  )}

                  <div className="compare-select-grid">
                    <label>
                      Track A
                      <select
                        value={compareArtifactA?.key || ""}
                        onChange={(e) =>
                          setCompareArtifactA(
                            availableCompareArtifacts.find((artifact) => artifact.key === e.target.value) ?? null
                          )
                        }
                      >
                        <option value="">Select A</option>
                        {availableCompareArtifacts.map((artifact) => (
                          <option key={`a-${artifact.key}`} value={artifact.key}>
                            {artifact.key}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label>
                      Track B
                      <select
                        value={compareArtifactB?.key || ""}
                        onChange={(e) =>
                          setCompareArtifactB(
                            availableCompareArtifacts.find((artifact) => artifact.key === e.target.value) ?? null
                          )
                        }
                      >
                        <option value="">Select B</option>
                        {availableCompareArtifacts.map((artifact) => (
                          <option key={`b-${artifact.key}`} value={artifact.key}>
                            {artifact.key}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  {compareAudioUrlA && compareAudioUrlB ? (
                    <>
                      <div className="compare-transport">
                        <button onClick={() => void setComparePlayback(!compareIsPlaying)}>
                          {compareIsPlaying ? "Pause" : "Play"}
                        </button>
                        <button
                          className={compareActiveSide === "A" ? "" : "secondary"}
                          onClick={() => switchCompareSide("A")}
                        >
                          Hear A
                        </button>
                        <button
                          className={compareActiveSide === "B" ? "" : "secondary"}
                          onClick={() => switchCompareSide("B")}
                        >
                          Hear B
                        </button>
                        <button className="secondary" onClick={() => syncComparePlayers(0)}>
                          Restart
                        </button>
                      </div>

                      <label className="compare-slider">
                        <span>Volume: {Math.round(compareVolume * 100)}%</span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.01}
                          value={compareVolume}
                          onChange={(e) => setCompareVolume(Number(e.target.value))}
                        />
                      </label>

                      <label className="compare-slider">
                        <span>
                          Position: {formatTime(compareCurrentTime)} / {formatTime(compareDuration)}
                        </span>
                        <input
                          type="range"
                          min={0}
                          max={compareDuration || 0}
                          step={0.01}
                          value={Math.min(compareCurrentTime, compareDuration || compareCurrentTime)}
                          onChange={(e) => {
                            syncComparePlayers(Number(e.target.value));
                          }}
                        />
                      </label>

                      <div className="compare-columns">
                        <div className={`compare-side ${compareActiveSide === "A" ? "compare-side-active" : ""}`}>
                          <div className="muted">A {compareActiveSide === "A" ? "(live)" : "(muted)"}</div>
                          <strong>{compareArtifactA?.key}</strong>
                          <div className="muted">{compareArtifactA?.relative_path || compareArtifactA?.path}</div>
                        </div>
                        <div className={`compare-side ${compareActiveSide === "B" ? "compare-side-active" : ""}`}>
                          <div className="muted">B {compareActiveSide === "B" ? "(live)" : "(muted)"}</div>
                          <strong>{compareArtifactB?.key}</strong>
                          <div className="muted">{compareArtifactB?.relative_path || compareArtifactB?.path}</div>
                        </div>
                      </div>

                      <audio
                        ref={compareAudioARef}
                        className="compare-hidden-audio"
                        preload="auto"
                        src={compareAudioUrlA}
                        onLoadedMetadata={updateCompareDuration}
                        onTimeUpdate={() => handleCompareTimeUpdate("A")}
                        onEnded={() => setCompareIsPlaying(false)}
                      />
                      <audio
                        ref={compareAudioBRef}
                        className="compare-hidden-audio"
                        preload="auto"
                        src={compareAudioUrlB}
                        onLoadedMetadata={updateCompareDuration}
                        onTimeUpdate={() => handleCompareTimeUpdate("B")}
                        onEnded={() => setCompareIsPlaying(false)}
                      />
                    </>
                  ) : (
                    <div className="hint-box compare-note">
                      Pick any two rendered audio artifacts from this job. Then use one transport and switch A/B instantly.
                    </div>
                  )}
                </div>
              )}

              <div className="artifact-toolbar">
                <div>
                  <strong>Current audio:</strong> {selectedAudioArtifact?.key || "none"}
                </div>
                <div>
                  <strong>Current lyrics:</strong> {selectedLyricsArtifact?.key || "none"}
                </div>
              </div>

              {selectedAudioUrl ? (
                <>
                  <audio
                    ref={audioRef}
                    controls
                    src={selectedAudioUrl}
                    onTimeUpdate={() => setCurrentTime(audioRef.current?.currentTime ?? 0)}
                  />
                  <div className="muted">Position: {formatTime(currentTime)}</div>
                </>
              ) : (
                <div className="warning-box">
                  This job has no rendered audio artifact selected. Original input is no longer auto-used as a fake stem
                  result. If you want the source track, select `input_audio` manually.
                </div>
              )}

              {renderedAudioArtifacts.length === 0 && (
                <div className="hint-box">
                  No output stem/audio artifact exists for this job yet. Lyrics-only jobs are expected to produce JSON /
                  TXT / LRC / SRT but no stem audio.
                </div>
              )}

              <h3>Output artifacts</h3>
              <div className="artifact-grid">
                {primaryArtifacts.map((artifact) => (
                  <div key={artifact.key} className={`artifact-card ${artifact.exists ? "" : "artifact-missing"}`}>
                    <div className="artifact-topline">
                      <strong>{artifact.key}</strong>
                      <span className="pill small">{artifact.kind}</span>
                    </div>
                    <div className="muted">{artifact.relative_path || artifact.path}</div>
                    <div className="muted">
                      {artifact.exists ? formatBytes(artifact.size_bytes) : "planned / missing"} · {artifact.source}
                    </div>
                    <div className="artifact-actions">
                      {artifact.exists && artifact.kind.startsWith("audio") && artifact.url && (
                        <>
                          <button onClick={() => setSelectedAudioArtifact(artifact)}>Listen</button>
                          <button className="secondary" onClick={() => setCompareArtifactA(artifact)}>
                            Set A
                          </button>
                          <button className="secondary" onClick={() => setCompareArtifactB(artifact)}>
                            Set B
                          </button>
                        </>
                      )}
                      {artifact.exists && (artifact.kind === "lyrics-json" || artifact.kind === "lyrics-text") && artifact.url && (
                        <button className="secondary" onClick={() => setSelectedLyricsArtifact(artifact)}>
                          Show text
                        </button>
                      )}
                      {artifact.exists && artifact.url && (
                        <a className="link-button" href={mediaUrl(artifact.url) ?? "#"} target="_blank" rel="noreferrer">
                          Open
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {requestArtifacts.length > 0 && (
                <>
                  <h3>Input and reference files</h3>
                  <div className="artifact-grid">
                    {requestArtifacts.map((artifact) => (
                      <div key={artifact.key} className={`artifact-card ${artifact.exists ? "" : "artifact-missing"}`}>
                        <div className="artifact-topline">
                          <strong>{artifact.key}</strong>
                          <span className="pill small">{artifact.kind}</span>
                        </div>
                        <div className="muted">{artifact.relative_path || artifact.path}</div>
                        <div className="muted">
                          {artifact.exists ? formatBytes(artifact.size_bytes) : "planned / missing"} | {artifact.source}
                        </div>
                        <div className="artifact-actions">
                          {artifact.exists && artifact.kind.startsWith("audio") && artifact.url && (
                            <>
                              <button onClick={() => setSelectedAudioArtifact(artifact)}>Listen</button>
                              <button className="secondary" onClick={() => setCompareArtifactA(artifact)}>
                                Set A
                              </button>
                              <button className="secondary" onClick={() => setCompareArtifactB(artifact)}>
                                Set B
                              </button>
                            </>
                          )}
                          {artifact.exists && (artifact.kind === "lyrics-json" || artifact.kind === "lyrics-text") && artifact.url && (
                            <button className="secondary" onClick={() => setSelectedLyricsArtifact(artifact)}>
                              Show text
                            </button>
                          )}
                          {artifact.exists && artifact.url && (
                            <a className="link-button" href={mediaUrl(artifact.url) ?? "#"} target="_blank" rel="noreferrer">
                              Open
                            </a>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {scanArtifacts.length > 0 && (
                <details>
                  <summary>Debug scan artifacts ({scanArtifacts.length})</summary>
                  <div className="artifact-grid">
                    {scanArtifacts.map((artifact) => (
                      <div key={artifact.key} className={`artifact-card ${artifact.exists ? "" : "artifact-missing"}`}>
                        <div className="artifact-topline">
                          <strong>{artifact.key}</strong>
                          <span className="pill small">{artifact.kind}</span>
                        </div>
                        <div className="muted">{artifact.relative_path || artifact.path}</div>
                        <div className="muted">
                          {artifact.exists ? formatBytes(artifact.size_bytes) : "planned / missing"} | {artifact.source}
                        </div>
                        <div className="artifact-actions">
                          {artifact.exists && artifact.kind.startsWith("audio") && artifact.url && (
                            <>
                              <button onClick={() => setSelectedAudioArtifact(artifact)}>Listen</button>
                              <button className="secondary" onClick={() => setCompareArtifactA(artifact)}>
                                Set A
                              </button>
                              <button className="secondary" onClick={() => setCompareArtifactB(artifact)}>
                                Set B
                              </button>
                            </>
                          )}
                          {artifact.exists && (artifact.kind === "lyrics-json" || artifact.kind === "lyrics-text") && artifact.url && (
                            <button className="secondary" onClick={() => setSelectedLyricsArtifact(artifact)}>
                              Show text
                            </button>
                          )}
                          {artifact.exists && artifact.url && (
                            <a className="link-button" href={mediaUrl(artifact.url) ?? "#"} target="_blank" rel="noreferrer">
                              Open
                            </a>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </>
          )}
        </article>

        <article className="card">
          <h2>Lyrics sync</h2>
          {!lyricsData && <p>Select a lyrics JSON/LRC/SRT/TXT artifact from the left.</p>}
          {lyricsData && (
            <>
              <div className="lyrics-meta">
                <span className="pill small">{lyricsData.source}</span>
                <span className="pill small">{lyricsData.language || "auto"}</span>
                <span className="pill small">segments: {lyricsData.segmentCount}</span>
                <span className="pill small">words: {lyricsData.wordCount}</span>
              </div>
              <div className="lyrics-panel">
                {lyricsData.segments.map((segment, index) => {
                  const active = index === activeSegmentIndex;
                  return (
                    <button
                      key={`${segment.start}-${segment.end}-${index}`}
                      className={`lyrics-segment ${active ? "lyrics-segment-active" : ""}`}
                      onClick={() => seekTo(segment.start)}
                    >
                      <span className="time-pill">
                        {formatTime(segment.start)} - {formatTime(segment.end)}
                      </span>
                      <span className="lyrics-segment-text">
                        {segment.words.length > 0
                          ? segment.words.map((word, wordIndex) => (
                              <span
                                key={`${word.start}-${wordIndex}`}
                                className={`word-chip ${active && wordIndex === activeWordIndex ? "word-chip-active" : ""}`}
                              >
                                {word.text}
                              </span>
                            ))
                          : segment.text}
                      </span>
                    </button>
                  );
                })}
              </div>
            </>
          )}
        </article>
      </section>

      <section className="grid two">
        <article className="card">
          <div className="section-toolbar">
            <div>
              <h2>VirtualDJ baseline compare</h2>
              <div className="muted">Separate A/B mode against exported VirtualDJ stems.</div>
            </div>
            <button className="secondary" onClick={() => void scanVirtualDjBaseline()} disabled={scanningVirtualDj || !virtualDjRoot.trim()}>
              {scanningVirtualDj ? "Scanning..." : "Scan baseline"}
            </button>
          </div>

          <label>
            VirtualDJ export folder inside containers
            <input
              value={virtualDjRoot}
              onChange={(e) => setVirtualDjRoot(e.target.value)}
              placeholder="/app/data/inputs/virtualdj/my_track_pack"
            />
          </label>

          <label className="upload-box">
            Upload VirtualDJ stem pack
            <input
              type="file"
              accept="audio/*"
              multiple
              onChange={(e) => {
                void uploadVirtualDjPack(e.target.files);
                e.currentTarget.value = "";
              }}
            />
            <span>{uploadingVirtualDjPack ? "Uploading..." : "Upload Acapella / Instrumental / Beat / Melody files together."}</span>
          </label>

          {!selectedJob && (
            <div className="warning-box">
              Select one of our jobs first. Auto-match only makes sense against a concrete pipeline result.
            </div>
          )}

          {virtualDjCompare && (
            <>
              <div className="pill-row">
                <span className="pill small">files: {virtualDjCompare.summary.file_count}</span>
                <span className="pill small">roles: {virtualDjCompare.summary.detected_roles.join(", ") || "none"}</span>
                <span className="pill small">job: {virtualDjCompare.job_id || "none"}</span>
              </div>

              {virtualDjCompare.summary.missing_recommended_roles.length > 0 && (
                <div className="warning-box">
                  Missing recommended VirtualDJ exports: {virtualDjCompare.summary.missing_recommended_roles.join(", ")}
                </div>
              )}

              <div className="compare-grid">
                {virtualDjCompare.items.map((item) => (
                  <div key={`${item.role}-${item.baseline_artifact.path}`} className="compare-card">
                    <div className="artifact-topline">
                      <strong>{item.role_label}</strong>
                      <span className="pill small">{item.role}</span>
                    </div>
                    <div className="compare-columns">
                      <div className="compare-side">
                        <div className="muted">VirtualDJ baseline</div>
                        <strong>{item.baseline_artifact.key}</strong>
                        <div className="muted">{item.baseline_artifact.relative_path || item.baseline_artifact.path}</div>
                        <div className="muted">{formatBytes(item.baseline_artifact.size_bytes)}</div>
                      </div>
                      <div className="compare-side">
                        <div className="muted">Our pipeline match</div>
                        {item.pipeline_artifact ? (
                          <>
                            <strong>{item.pipeline_artifact.key}</strong>
                            <div className="muted">{item.pipeline_artifact.relative_path || item.pipeline_artifact.path}</div>
                            <div className="muted">{formatBytes(item.pipeline_artifact.size_bytes)}</div>
                          </>
                        ) : (
                          <>
                            <strong>No auto-match</strong>
                            <div className="muted">Try one of: {item.recommended_pipeline_keys.join(", ")}</div>
                          </>
                        )}
                      </div>
                    </div>
                    <div className="hint-box compare-note">{item.note}</div>
                    <div className="artifact-actions">
                      {item.baseline_artifact.url && item.pipeline_artifact?.url && (
                        <button
                          className="secondary"
                          onClick={() => loadComparePair(item.baseline_artifact, item.pipeline_artifact)}
                        >
                          A/B compare
                        </button>
                      )}
                      {item.baseline_artifact.url && (
                        <>
                          <button onClick={() => setSelectedAudioArtifact(item.baseline_artifact)}>Listen baseline</button>
                          <a
                            className="link-button"
                            href={mediaUrl(item.baseline_artifact.url) ?? "#"}
                            target="_blank"
                            rel="noreferrer"
                          >
                            Open baseline
                          </a>
                        </>
                      )}
                      {item.pipeline_artifact?.url && (
                        <>
                          <button className="secondary" onClick={() => setSelectedAudioArtifact(item.pipeline_artifact)}>
                            Listen ours
                          </button>
                          <a
                            className="link-button"
                            href={mediaUrl(item.pipeline_artifact.url) ?? "#"}
                            target="_blank"
                            rel="noreferrer"
                          >
                            Open ours
                          </a>
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}

          {!virtualDjCompare && (
            <div className="hint-box">
              This mode is for honest A/B, not vibes. Point it at one VirtualDJ export folder with Acapella / Instrumental /
              Beat / Melody files, then compare against the selected job.
            </div>
          )}
        </article>

        <article className="card">
          <h2>VirtualDJ checklist</h2>
          {!virtualDjCompare && (
            <p className="muted">
              Scan a VirtualDJ folder or upload a stem pack to load the export checklist and role mapping.
            </p>
          )}
          {virtualDjCompare && (
            <div className="checklist-grid">
              <div className="checklist-block">
                <h3>What to export</h3>
                <ul>
                  {virtualDjCompare.checklist.what_to_export.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
              <div className="checklist-block">
                <h3>Preferred format</h3>
                <ul>
                  {virtualDjCompare.checklist.preferred_format.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
              <div className="checklist-block">
                <h3>How to compare</h3>
                <ul>
                  {virtualDjCompare.checklist.how_to_compare.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </article>
      </section>

      <section className="card">
        <h2>Pipeline plan</h2>
        {preset?.phases.map((phase) => (
          <div key={phase.id} className="phase-box">
            <div className="phase-header">
              <strong>{phase.label}</strong>
              <span>{phase.id}</span>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Stem</th>
                  <th>Service</th>
                  <th>Task</th>
                  <th>Model / Preset / Prompt / Input bus</th>
                </tr>
              </thead>
              <tbody>
                {phase.steps.map((step, index) => (
                  <tr key={`${phase.id}-${index}`}>
                    <td>{String(step.stem ?? "—")}</td>
                    <td>{String(step.service ?? "—")}</td>
                    <td>{String(step.task ?? "—")}</td>
                    <td>{String(step.model ?? step.preset ?? step.prompt ?? step.input_bus ?? "—")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </section>

      <section className="card">
        <h2>Services</h2>
        <div className="service-grid">
          {services.map((service) => (
            <div key={service.id} className="service-card">
              <strong>{service.id}</strong>
              <code>{service.url}</code>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
