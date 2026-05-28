import { ChangeEvent, useEffect, useMemo, useState } from "react";
import { fetchJson, postFormData, postJson } from "./api";

type Service = { id: string; url: string };
type ServicesResponse = { services: Service[] };
type PresetsResponse = { presets: string[] };
type UploadEntry = { name: string; input_path: string; size_bytes: number };
type UploadListResponse = { files: UploadEntry[] };
type UploadResponse = {
  filename: string;
  original_filename: string;
  size_bytes: number;
  input_path: string;
};

type Job = {
  job_id: string;
  status: string;
  output_root: string;
  request: {
    track_name: string;
    input_path: string;
    preset_name: string;
    notes?: string | null;
  };
  invocations: Array<{
    phase_id: string;
    stem?: string | null;
    service: string;
    task: string;
    model?: string | null;
    preset?: string | null;
    prompt?: string | null;
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

const defaultInputPath = "/app/data/inputs/demo.wav";

export default function App() {
  const [services, setServices] = useState<Service[]>([]);
  const [presets, setPresets] = useState<string[]>([]);
  const [uploads, setUploads] = useState<UploadEntry[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>("fine_grained_14stem");
  const [preset, setPreset] = useState<Preset | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [trackName, setTrackName] = useState("Demo Track");
  const [inputPath, setInputPath] = useState(defaultInputPath);
  const [notes, setNotes] = useState("Dry-run pipeline test");
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUploadMessage, setLastUploadMessage] = useState<string | null>(null);

  async function reloadAll(): Promise<void> {
    setLoading(true);
    setError(null);
    try {
      const [servicesData, presetsData, jobsData, uploadsData] = await Promise.all([
        fetchJson<ServicesResponse>("/api/v1/catalog/services"),
        fetchJson<PresetsResponse>("/api/v1/catalog/presets"),
        fetchJson<JobsResponse>("/api/v1/jobs"),
        fetchJson<UploadListResponse>("/api/v1/uploads"),
      ]);

      setServices(servicesData.services);
      setPresets(presetsData.presets);
      setJobs(jobsData.jobs);
      setUploads(uploadsData.files);

      const presetName = presetsData.presets.includes(selectedPreset)
        ? selectedPreset
        : presetsData.presets[0];

      if (presetName) {
        const presetData = await fetchJson<Preset>(`/api/v1/catalog/presets/${presetName}`);
        setSelectedPreset(presetName);
        setPreset(presetData);
      }
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
    const hasActiveJobs = jobs.some((job) => job.status === "queued" || job.status === "running");
    if (!hasActiveJobs) return;

    const timer = window.setInterval(() => {
      void fetchJson<JobsResponse>("/api/v1/jobs")
        .then((jobsData) => {
          setJobs(jobsData.jobs);
        })
        .catch((err: unknown) => {
          setError(err instanceof Error ? err.message : "Jobs refresh error");
        });
    }, 2000);

    return () => {
      window.clearInterval(timer);
    };
  }, [jobs]);

  const phaseCount = useMemo(() => preset?.phases.length ?? 0, [preset]);

  async function createJob(): Promise<void> {
    setCreating(true);
    setError(null);
    try {
      await postJson<Job>("/api/v1/jobs", {
        track_name: trackName,
        input_path: inputPath,
        preset_name: selectedPreset,
        notes,
      });
      await reloadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Job create error");
    } finally {
      setCreating(false);
    }
  }

  async function handleFileUpload(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    setLastUploadMessage(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const result = await postFormData<UploadResponse>("/api/v1/uploads", formData);

      setInputPath(result.input_path);
      setTrackName((current) => (current === "Demo Track" ? file.name : current));
      setLastUploadMessage(`Uploaded ${result.original_filename} -> ${result.input_path}`);
      await reloadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload error");
    } finally {
      event.target.value = "";
      setUploading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <h1>StemLab</h1>
          <p>
            Multi-container dry-run orchestration surface. Upload a file, let the API place it
            into the shared inputs volume, and then create a job without guessing container paths.
          </p>
        </div>
        <button onClick={() => void reloadAll()} className="secondary">
          Reload
        </button>
      </header>

      {error && <div className="error-box">{error}</div>}
      {lastUploadMessage && <div className="success-box">{lastUploadMessage}</div>}

      <section className="grid two">
        <article className="card">
          <h2>Create job</h2>

          <label>
            Upload audio into shared container inputs
            <input type="file" accept="audio/*" onChange={(event) => void handleFileUpload(event)} />
            <small className="helper-text">
              Uploaded files are stored under <code>/app/data/inputs</code> and the correct
              container path is filled automatically.
            </small>
          </label>

          <label>
            Track name
            <input value={trackName} onChange={(e) => setTrackName(e.target.value)} />
          </label>

          <label>
            Input path inside containers
            <input value={inputPath} onChange={(e) => setInputPath(e.target.value)} />
            <small className="helper-text">
              If you upload through this UI, do not guess the path manually. Use the auto-filled
              <code> /app/data/inputs/&lt;file&gt;</code> value.
            </small>
          </label>

          {uploads.length > 0 && (
            <label>
              Existing uploaded files
              <select value={inputPath} onChange={(e) => setInputPath(e.target.value)}>
                <option value="">Select uploaded input</option>
                {uploads.map((entry) => (
                  <option key={entry.input_path} value={entry.input_path}>
                    {entry.name} ({Math.round(entry.size_bytes / 1024)} KB)
                  </option>
                ))}
              </select>
            </label>
          )}

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

          <label>
            Notes
            <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={4} />
          </label>

          <button disabled={creating || uploading || !inputPath.trim()} onClick={() => void createJob()}>
            {uploading ? "Uploading..." : creating ? "Creating..." : "Create job"}
          </button>
        </article>

        <article className="card">
          <h2>Active preset</h2>
          {loading && <p>Loading...</p>}
          {!loading && preset && (
            <>
              <p><strong>{preset.name}</strong></p>
              <p>{preset.description}</p>
              <div className="pill-row">
                {preset.targets.map((target) => (
                  <span key={target} className="pill">
                    {target}
                  </span>
                ))}
              </div>
              <p>Phases: {phaseCount}</p>
            </>
          )}
        </article>
      </section>

      <section className="grid two">
        <article className="card">
          <h2>Services</h2>
          <div className="service-grid">
            {services.map((service) => (
              <div key={service.id} className="service-card">
                <strong>{service.id}</strong>
                <code>{service.url}</code>
              </div>
            ))}
          </div>
        </article>

        <article className="card">
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
                    <th>Model / Preset / Prompt</th>
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
        </article>
      </section>

      <section className="card">
        <h2>Jobs</h2>
        <div className="job-list">
          {jobs.length === 0 && <p>No jobs yet.</p>}
          {jobs.map((job) => (
            <div key={job.job_id} className="job-card">
              <div className="job-head">
                <div>
                  <strong>{job.request.track_name}</strong>
                  <div className="muted">{job.job_id}</div>
                </div>
                <span className={`status status-${job.status}`}>{job.status}</span>
              </div>
              <p>Preset: {job.request.preset_name}</p>
              <p>Input: <code>{job.request.input_path}</code></p>
              <p>Steps: {job.invocations.length}</p>
              <details>
                <summary>Run log</summary>
                <pre>{JSON.stringify(job.run_log, null, 2)}</pre>
              </details>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
