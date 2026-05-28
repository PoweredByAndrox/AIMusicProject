#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const MODE_CONFIG_FIELD_BY_RENDER_MODE = {
    post_dsp: 'processorConfig',
    hybrid_build: 'hybridBuildConfig',
    stem_rewrite: 'stemRewriteConfig',
    vocals_rewrite: 'vocalsRewriteConfig',
};

function printHelp() {
    console.log(`Usage:
  node scripts/test-stem-tuning-api.mjs --track 111 --token <API_TOKEN> [options]

Required:
  --track <id>                    Track id.
  --token <token>                 API bearer token, or set MUSIC_AI_API_TOKEN.

Optional:
  --base-url <url>                Default: http://localhost:8000
  --preset-input <json|file>      Full preset JSON like "Apply editor JSON".
  --mode-config-input <json|file> Mode JSON like "Apply mode JSON".
  --mode-config-field <field>     One of processorConfig|hybridBuildConfig|stemRewriteConfig|vocalsRewriteConfig.
  --unsafe-input <json|file>      Unsafe internals JSON like "Apply unsafe internals JSON".
  --render-mode <mode>            One of post_dsp|hybrid_build|stem_rewrite|vocals_rewrite.
  --source-key <key>              Example: stem_vocals.
  --enable-only <csv>             Example: stem_vocals or stem_drums,stem_other
  --build-if-missing              If lab is unavailable but buildable, build stems first.
  --poll-ms <ms>                  Default: 1000
  --timeout-sec <sec>             Default: 3600
    --output-dir <dir>              Default: .tmp/stem-tuning-api
  --summary-path <file>            Optional final summary JSON path
  --dry-run                       Build and dump request JSON without POST /render
  --print-session                 Print session source summary before render
  --print-request                 Print final request JSON to stdout

Examples:
  node scripts/test-stem-tuning-api.mjs --track 111 --token %MUSIC_AI_API_TOKEN% --render-mode vocals_rewrite --source-key stem_vocals --enable-only stem_vocals --mode-config-input D:\\preset\\vocals.json --unsafe-input D:\\preset\\unsafe.json

  node scripts/test-stem-tuning-api.mjs --track 111 --token %MUSIC_AI_API_TOKEN% --preset-input D:\\preset\\full.json --print-request
`);
}

function parseArgs(argv) {
    const args = {};
    for (let index = 0; index < argv.length; index += 1) {
        const raw = argv[index];
        if (!raw.startsWith('--')) {
            continue;
        }

        const key = raw.slice(2);
        const next = argv[index + 1];
        if (!next || next.startsWith('--')) {
            args[key] = true;
            continue;
        }

        args[key] = next;
        index += 1;
    }

    return args;
}

function isPlainObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function deepClone(value) {
    return JSON.parse(JSON.stringify(value));
}

function deepMerge(baseValue, patchValue) {
    if (Array.isArray(baseValue) && Array.isArray(patchValue)) {
        return deepClone(patchValue);
    }

    if (isPlainObject(baseValue) && isPlainObject(patchValue)) {
        const merged = { ...baseValue };
        for (const [key, patchChild] of Object.entries(patchValue)) {
            if (key in merged) {
                merged[key] = deepMerge(merged[key], patchChild);
                continue;
            }
            merged[key] = deepClone(patchChild);
        }
        return merged;
    }

    return deepClone(patchValue);
}

function resolveJsonInput(value, label) {
    if (!value) {
        return null;
    }

    const maybePath = path.resolve(process.cwd(), value);
    const raw = fs.existsSync(maybePath)
        ? fs.readFileSync(maybePath, 'utf8')
        : value;
    const normalizedRaw = raw.replace(/^\uFEFF/, '');

    try {
        return JSON.parse(normalizedRaw);
    } catch (error) {
        throw new Error(`${label} is not valid JSON: ${error.message}`);
    }
}

function summarizeSources(session) {
    const sources = Array.isArray(session?.lab?.sources) ? session.lab.sources : [];
    return sources.map((source) => ({
        key: source.key,
        editable: Boolean(source.editable),
        label: source.label,
        kind: source.kind,
        is_rendered: Boolean(source.is_rendered),
    }));
}

function getEnabledLayerKeys(preset) {
    return Object.entries(preset?.layerStates ?? {})
        .filter(([, state]) => Boolean(state?.enabled))
        .map(([key]) => key);
}

function ensureDirectory(dirPath) {
    fs.mkdirSync(dirPath, { recursive: true });
}

function stableJson(value) {
    return JSON.stringify(value, null, 2);
}

async function requestJson(method, url, { token, body } = {}) {
    const headers = {
        Accept: 'application/json',
    };
    if (token) {
        headers.Authorization = `Bearer ${token}`;
    }
    if (body !== undefined) {
        headers['Content-Type'] = 'application/json';
    }

    const response = await fetch(url, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
    });

    const text = await response.text();
    let payload = null;
    if (text) {
        try {
            payload = JSON.parse(text);
        } catch {
            payload = text;
        }
    }

    return {
        ok: response.ok,
        status: response.status,
        payload,
    };
}

async function wait(ms) {
    await new Promise((resolve) => setTimeout(resolve, ms));
}

function sha256File(filePath) {
    const hash = crypto.createHash('sha256');
    const stream = fs.createReadStream(filePath);
    return new Promise((resolve, reject) => {
        stream.on('data', (chunk) => hash.update(chunk));
        stream.on('error', reject);
        stream.on('end', () => resolve(hash.digest('hex')));
    });
}

async function pollTask({ baseApiUrl, token, trackId, taskId, pollMs, timeoutSec, logPrefix }) {
    const startedAt = Date.now();
    let lastStatus = null;
    let lastLogCount = 0;

    while (true) {
        const response = await requestJson('GET', `${baseApiUrl}/tracks/${trackId}/tuning-lab/render-tasks/${taskId}`, {
            token,
        });
        if (!response.ok) {
            throw new Error(`Render status request failed (${response.status}): ${stableJson(response.payload)}`);
        }

        const task = response.payload;
        if (task.status !== lastStatus) {
            console.log(`${logPrefix} status=${task.status} progress=${task.progress ?? 0}% message=${task.message ?? ''}`);
            lastStatus = task.status;
        }

        const logs = Array.isArray(task.logs) ? task.logs : [];
        if (logs.length > lastLogCount) {
            for (const entry of logs.slice(lastLogCount)) {
                console.log(`${logPrefix} ${entry.timestamp} ${entry.progress ?? 0}% ${entry.level} ${entry.stage ?? 'render'} ${entry.message}`);
            }
            lastLogCount = logs.length;
        }

        if (['completed', 'failed', 'cancelled'].includes(task.status)) {
            return task;
        }

        const elapsedSec = (Date.now() - startedAt) / 1000;
        if (elapsedSec > timeoutSec) {
            throw new Error(`Render polling timed out after ${timeoutSec}s`);
        }

        await wait(pollMs);
    }
}

async function pollBuild({ baseApiUrl, token, trackId, taskId, pollMs, timeoutSec, logPrefix }) {
    const startedAt = Date.now();

    while (true) {
        const response = await requestJson('GET', `${baseApiUrl}/tracks/${trackId}/tuning-lab/build-stem-tasks/${taskId}`, {
            token,
        });
        if (!response.ok) {
            throw new Error(`Build status request failed (${response.status}): ${stableJson(response.payload)}`);
        }

        const task = response.payload;
        console.log(`${logPrefix} status=${task.status} message=${task.error_message ?? ''}`);
        if (['completed', 'failed', 'cancelled'].includes(task.status)) {
            return task;
        }

        const elapsedSec = (Date.now() - startedAt) / 1000;
        if (elapsedSec > timeoutSec) {
            throw new Error(`Build polling timed out after ${timeoutSec}s`);
        }

        await wait(pollMs);
    }
}

function inferModeConfigField(renderMode, explicitField) {
    if (explicitField) {
        return explicitField;
    }
    const inferred = MODE_CONFIG_FIELD_BY_RENDER_MODE[renderMode];
    if (!inferred) {
        throw new Error(`Cannot infer mode config field for renderMode=${renderMode}`);
    }
    return inferred;
}

function applyEnableOnly(preset, enableOnlyCsv) {
    if (!enableOnlyCsv) {
        return;
    }

    const enabledKeys = enableOnlyCsv
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean);

    for (const key of Object.keys(preset.layerStates ?? {})) {
        const nextState = isPlainObject(preset.layerStates[key]) ? preset.layerStates[key] : {};
        preset.layerStates[key] = {
            ...nextState,
            enabled: enabledKeys.includes(key),
        };
    }

    if (enabledKeys.length > 0 && !enabledKeys.includes(preset.sourceKey)) {
        preset.sourceKey = enabledKeys[0];
    }

    if (preset?.focusWindow && typeof preset.focusWindow === 'object' && preset.sourceKey) {
        preset.focusWindow.targetLayerKey = preset.sourceKey;
    }
}

function summarizePreset(preset) {
    return {
        renderMode: preset.renderMode,
        sourceKey: preset.sourceKey,
        enabledLayers: getEnabledLayerKeys(preset),
        modeConfigField: MODE_CONFIG_FIELD_BY_RENDER_MODE[preset.renderMode] ?? null,
        activeModeConfig: preset[MODE_CONFIG_FIELD_BY_RENDER_MODE[preset.renderMode] ?? 'processorConfig'],
    };
}

async function inspectArtifacts({ trackId, renderId, outputDir }) {
    if (!renderId) {
        return null;
    }

    const artifactDir = path.resolve(process.cwd(), 'lab_artifacts', 'tuning_lab', `track_${trackId}`, `render_${renderId}`);
    const manifestPath = path.join(artifactDir, 'stem_manifest.json');
    const result = {
        artifactDir,
        manifestPath,
        manifestExists: fs.existsSync(manifestPath),
        files: {},
    };

    if (!result.manifestExists) {
        return result;
    }

    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    result.manifest = {
        pipelineMode: manifest?.pipeline?.mode ?? null,
        renderStrategy: manifest?.pipeline?.render_strategy ?? null,
        presetRenderMode: manifest?.preset?.renderMode ?? null,
        presetSourceKey: manifest?.preset?.sourceKey ?? null,
        enabledLayers: Object.entries(manifest?.preset?.layerStates ?? {})
            .filter(([, state]) => Boolean(state?.enabled))
            .map(([key]) => key),
        sourcePipeline: manifest?.pipeline?.source_pipeline ?? null,
    };
    result.observations = [];

    for (const name of ['mix.flac', 'vocals.flac', 'drums.flac', 'bass.flac', 'other.flac', 'guitar.flac', 'piano.flac']) {
        const filePath = path.join(artifactDir, name);
        if (!fs.existsSync(filePath)) {
            continue;
        }

        const stats = fs.statSync(filePath);
        result.files[name] = {
            sizeBytes: stats.size,
            sha256: await sha256File(filePath),
        };
    }

    if (result.files['mix.flac'] && Array.isArray(result.manifest.enabledLayers) && result.manifest.enabledLayers.length === 1) {
        const onlyEnabledLayer = result.manifest.enabledLayers[0];
        const enabledStemFile = `${onlyEnabledLayer}.flac`;
        if (result.files[enabledStemFile] && result.files[enabledStemFile].sha256 === result.files['mix.flac'].sha256) {
            result.observations.push(
                `mix.flac equals ${enabledStemFile} because only one layer is enabled in the preset`,
            );
        }
    }

    ensureDirectory(outputDir);
    fs.writeFileSync(path.join(outputDir, `track-${trackId}-render-${renderId}-manifest.json`), stableJson(manifest));
    return result;
}

async function main() {
    const args = parseArgs(process.argv.slice(2));
    if (args.help || args.h) {
        printHelp();
        process.exit(0);
    }

    const trackId = Number(args.track);
    if (!Number.isInteger(trackId) || trackId <= 0) {
        throw new Error('--track is required and must be a positive integer');
    }

    const token = args.token || process.env.MUSIC_AI_API_TOKEN || process.env.MUSIC_AI_TOKEN;
    if (!token) {
        throw new Error('API token is required. Use --token or set MUSIC_AI_API_TOKEN.');
    }

    const baseUrl = String(args['base-url'] || 'http://localhost:8000').replace(/\/$/, '');
    const baseApiUrl = `${baseUrl}/api`;
    const pollMs = Number(args['poll-ms'] || 1000);
    const timeoutSec = Number(args['timeout-sec'] || 3600);
    const outputDir = path.resolve(process.cwd(), String(args['output-dir'] || path.join('.tmp', 'stem-tuning-api')));
    const explicitSummaryPath = args['summary-path']
        ? path.resolve(process.cwd(), String(args['summary-path']))
        : null;
    ensureDirectory(outputDir);

    console.log(`[stem-tuning-api] GET session for track ${trackId}`);
    let sessionResponse = await requestJson('GET', `${baseApiUrl}/tracks/${trackId}/tuning-lab`, { token });

    if (!sessionResponse.ok && sessionResponse.status === 404 && args['build-if-missing']) {
        const buildable = Boolean(sessionResponse.payload?.buildable);
        if (!buildable) {
            throw new Error(`Lab is unavailable and track is not buildable: ${stableJson(sessionResponse.payload)}`);
        }

        console.log(`[stem-tuning-api] Lab unavailable. Triggering stem build for track ${trackId}`);
        const buildResponse = await requestJson('POST', `${baseApiUrl}/tracks/${trackId}/tuning-lab/build-stems`, { token });
        if (!buildResponse.ok) {
            throw new Error(`Build request failed (${buildResponse.status}): ${stableJson(buildResponse.payload)}`);
        }

        const buildTaskId = buildResponse.payload?.task_id;
        if (!buildTaskId) {
            throw new Error(`Build request returned no task id: ${stableJson(buildResponse.payload)}`);
        }

        const buildStatus = await pollBuild({
            baseApiUrl,
            token,
            trackId,
            taskId: buildTaskId,
            pollMs,
            timeoutSec,
            logPrefix: `[stem-tuning-api][build:${buildTaskId}]`,
        });
        if (buildStatus.status !== 'completed') {
            throw new Error(`Stem build did not complete successfully: ${stableJson(buildStatus)}`);
        }

        sessionResponse = await requestJson('GET', `${baseApiUrl}/tracks/${trackId}/tuning-lab`, { token });
    }

    if (!sessionResponse.ok) {
        throw new Error(`Session request failed (${sessionResponse.status}): ${stableJson(sessionResponse.payload)}`);
    }

    const session = sessionResponse.payload;
    if (args['print-session']) {
        console.log(stableJson({
            defaultSourceKey: session.lab?.default_source_key ?? null,
            sources: summarizeSources(session),
        }));
    }

    const requestStamp = new Date().toISOString().replace(/[:.]/g, '-');
    const requestSummaryPath = path.join(outputDir, `track-${trackId}-request-${requestStamp}.json`);
    const statusSummaryPath = path.join(outputDir, `track-${trackId}-status-${requestStamp}.json`);

    let preset = deepClone(session.lab?.default_preset ?? {});
    const fullPresetPatch = resolveJsonInput(args['preset-input'], 'Preset input');
    if (fullPresetPatch) {
        preset = deepMerge(preset, fullPresetPatch);
    }

    if (args['render-mode']) {
        preset.renderMode = args['render-mode'];
    }
    if (args['source-key']) {
        preset.sourceKey = args['source-key'];
        if (preset?.focusWindow && typeof preset.focusWindow === 'object') {
            preset.focusWindow.targetLayerKey = preset.sourceKey;
        }
    }

    const modeConfigPatch = resolveJsonInput(args['mode-config-input'], 'Mode config input');
    if (modeConfigPatch) {
        const field = inferModeConfigField(preset.renderMode, args['mode-config-field']);
        preset[field] = deepMerge(preset[field] ?? {}, modeConfigPatch);
    }

    const unsafePatch = resolveJsonInput(args['unsafe-input'], 'Unsafe input');
    if (unsafePatch) {
        preset.unsafeModelInternals = deepMerge(preset.unsafeModelInternals ?? {}, unsafePatch);
    }

    applyEnableOnly(preset, args['enable-only']);

    const requestSummary = {
        trackId,
        baseUrl,
        requestBuiltAt: new Date().toISOString(),
        preset,
        summary: summarizePreset(preset),
    };

    fs.writeFileSync(requestSummaryPath, stableJson(requestSummary));
    console.log(`[stem-tuning-api] Request JSON saved to ${requestSummaryPath}`);

    if (args['print-request']) {
        console.log(stableJson(requestSummary));
    }

    if (args['dry-run']) {
        console.log('[stem-tuning-api] Dry run only. No render submitted.');
        return;
    }

    console.log(`[stem-tuning-api] POST /render for track ${trackId}`);
    const renderResponse = await requestJson('POST', `${baseApiUrl}/tracks/${trackId}/tuning-lab/render`, {
        token,
        body: { preset },
    });
    if (!renderResponse.ok) {
        throw new Error(`Render request failed (${renderResponse.status}): ${stableJson(renderResponse.payload)}`);
    }

    const taskId = renderResponse.payload?.task_id;
    if (!taskId) {
        throw new Error(`Render request returned no task id: ${stableJson(renderResponse.payload)}`);
    }

    console.log(`[stem-tuning-api] Render task queued: ${taskId}`);
    const finalTask = await pollTask({
        baseApiUrl,
        token,
        trackId,
        taskId,
        pollMs,
        timeoutSec,
        logPrefix: `[stem-tuning-api][task:${taskId}]`,
    });

    fs.writeFileSync(statusSummaryPath, stableJson(finalTask));
    console.log(`[stem-tuning-api] Final task status saved to ${statusSummaryPath}`);

    const renderId = finalTask?.payload?.render_id ?? null;
    const artifactInspection = await inspectArtifacts({
        trackId,
        renderId,
        outputDir,
    });

    const finalSummary = {
        taskId,
        status: finalTask.status,
        progress: finalTask.progress ?? null,
        message: finalTask.message ?? null,
        error: finalTask.error_message ?? null,
        renderId,
        requestSummaryPath,
        statusSummaryPath,
        artifactInspection,
    };

    if (explicitSummaryPath) {
        ensureDirectory(path.dirname(explicitSummaryPath));
        fs.writeFileSync(explicitSummaryPath, stableJson(finalSummary));
    }
    console.log(stableJson(finalSummary));
}

main().catch((error) => {
    console.error(`[stem-tuning-api] ${error.message}`);
    process.exit(1);
});
