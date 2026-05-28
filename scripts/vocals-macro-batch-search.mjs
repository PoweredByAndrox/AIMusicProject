#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

function printHelp() {
    console.log(`Usage:
  node scripts/vocals-macro-batch-search.mjs --track <id> --token <API_TOKEN> [options]

Required:
  --track <id>                    Track id.
  --token <token>                 API bearer token, or set MUSIC_AI_API_TOKEN.

Optional:
  --base-url <url>                Default: http://localhost:8000
  --unsafe-input <json|file>      Unsafe internals JSON like "Apply unsafe internals JSON".
  --output-dir <dir>              Default: .tmp/vocals-macro-batch-search
  --poll-ms <ms>                  Default: 1500
  --timeout-sec <sec>             Default: 3600
  --target-stem <stem>            Default: vocals
  --source-key <key>              Default: stem_vocals
  --enable-only <csv>             Default: stem_vocals
  --continue-on-error             Continue if some candidates fail
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

function ensureDirectory(dirPath) {
    fs.mkdirSync(dirPath, { recursive: true });
}

function readJson(filePath) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, ''));
}

function writeJson(filePath, payload) {
    fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
}

function runCommand(command, args, { cwd } = {}) {
    const result = spawnSync(command, args, {
        cwd,
        stdio: ['ignore', 'pipe', 'inherit'],
        encoding: 'utf8',
    });
    if (result.stdout) {
        process.stdout.write(result.stdout);
    }
    if (result.status !== 0) {
        throw new Error(`Command failed: ${command} ${args.join(' ')}`);
    }
    return result.stdout ?? '';
}

function findNewestFile(rootDir, fileName) {
    const matches = [];

    function walk(currentDir) {
        const entries = fs.readdirSync(currentDir, { withFileTypes: true });
        for (const entry of entries) {
            const absolutePath = path.join(currentDir, entry.name);
            if (entry.isDirectory()) {
                walk(absolutePath);
                continue;
            }
            if (entry.isFile() && entry.name === fileName) {
                matches.push({
                    path: absolutePath,
                    mtimeMs: fs.statSync(absolutePath).mtimeMs,
                });
            }
        }
    }

    walk(rootDir);
    matches.sort((left, right) => right.mtimeMs - left.mtimeMs);
    return matches[0]?.path ?? null;
}

function defaultVocalsRewriteConfig() {
    return {
        taskMode: 'custom',
        vocalIsolationIntensity: 0,
        bleedRemovalIntensity: 0,
        airRestore: 0,
        bodyPreserve: 0,
        artifactSuppression: 0,
        centerVocalSuppression: 0,
        candidateMode: 'hybrid_vocals',
        lowBsWeight: 0.75,
        highBsWeight: 0.25,
        inactiveLowBsWeight: 0.97,
        inactiveHighBsWeight: 0.9,
        crossoverHz: 3000,
        activityAbsFloorRatio: 0.08,
        activityRelFloor: 0.35,
        activitySmoothingFrames: 11,
        vocalSharpenAlpha: 0,
        tailGateStrength: 0,
        tailGateFloorRatio: 0.08,
        tailGateSmoothingFrames: 11,
        subtractiveStrength: 0,
        subtractivePower: 1,
        subtractiveResidualFloor: 0.12,
        subtractiveActiveRestore: 0,
        subtractiveTargetRmsRatio: 0,
        subtractiveMaxGain: 1.35,
        hardMaskThreshold: 0,
        hardMaskPower: 3,
        hardMaskResidualFloor: 0.02,
    };
}

function taskModePreset(mode) {
    const custom = defaultVocalsRewriteConfig();
    const presets = {
        clean_vocals: {
            taskMode: 'clean_vocals',
            vocalIsolationIntensity: 0.58,
            bleedRemovalIntensity: 0.52,
            airRestore: 0.46,
            bodyPreserve: 0.42,
            artifactSuppression: 0.38,
            centerVocalSuppression: 0.42,
            candidateMode: 'hybrid_vocals',
            lowBsWeight: 0.82,
            highBsWeight: 0.34,
            inactiveLowBsWeight: 0.98,
            inactiveHighBsWeight: 0.93,
            crossoverHz: 3200,
            activityAbsFloorRatio: 0.06,
            activityRelFloor: 0.3,
            activitySmoothingFrames: 9,
            vocalSharpenAlpha: 0.08,
            tailGateStrength: 0.12,
            tailGateFloorRatio: 0.06,
            tailGateSmoothingFrames: 9,
            subtractiveStrength: 0.22,
            subtractivePower: 1.18,
            subtractiveResidualFloor: 0.14,
            subtractiveActiveRestore: 0.04,
            subtractiveTargetRmsRatio: 0.98,
            subtractiveMaxGain: 1.28,
            hardMaskThreshold: 0,
            hardMaskPower: 3,
            hardMaskResidualFloor: 0.02,
        },
        minus_from_vocals: {
            taskMode: 'minus_from_vocals',
            vocalIsolationIntensity: 0.9,
            bleedRemovalIntensity: 0.85,
            airRestore: 0.2,
            bodyPreserve: 0.2,
            artifactSuppression: 0.62,
            centerVocalSuppression: 0.92,
            candidateMode: 'hybrid_vocals',
            lowBsWeight: 0.88,
            highBsWeight: 0.42,
            inactiveLowBsWeight: 0.99,
            inactiveHighBsWeight: 0.96,
            crossoverHz: 3400,
            activityAbsFloorRatio: 0.04,
            activityRelFloor: 0.24,
            activitySmoothingFrames: 7,
            vocalSharpenAlpha: 0.04,
            tailGateStrength: 0.22,
            tailGateFloorRatio: 0.05,
            tailGateSmoothingFrames: 7,
            subtractiveStrength: 0.42,
            subtractivePower: 1.35,
            subtractiveResidualFloor: 0.08,
            subtractiveActiveRestore: 0.01,
            subtractiveTargetRmsRatio: 0.85,
            subtractiveMaxGain: 1.12,
            hardMaskThreshold: 0,
            hardMaskPower: 3,
            hardMaskResidualFloor: 0.02,
        },
        artifact_repair: {
            taskMode: 'artifact_repair',
            vocalIsolationIntensity: 0.3,
            bleedRemovalIntensity: 0.38,
            airRestore: 0.34,
            bodyPreserve: 0.72,
            artifactSuppression: 0.86,
            centerVocalSuppression: 0.18,
            candidateMode: 'ft_vocals',
            lowBsWeight: 0.58,
            highBsWeight: 0.18,
            inactiveLowBsWeight: 0.88,
            inactiveHighBsWeight: 0.78,
            crossoverHz: 2800,
            activityAbsFloorRatio: 0.09,
            activityRelFloor: 0.42,
            activitySmoothingFrames: 15,
            vocalSharpenAlpha: 0,
            tailGateStrength: 0.05,
            tailGateFloorRatio: 0.08,
            tailGateSmoothingFrames: 15,
            subtractiveStrength: 0.1,
            subtractivePower: 1.05,
            subtractiveResidualFloor: 0.2,
            subtractiveActiveRestore: 0.08,
            subtractiveTargetRmsRatio: 1.05,
            subtractiveMaxGain: 1.4,
            hardMaskThreshold: 0,
            hardMaskPower: 3,
            hardMaskResidualFloor: 0.02,
        },
        aggressive_vocal_suppression: {
            taskMode: 'aggressive_vocal_suppression',
            vocalIsolationIntensity: 0.96,
            bleedRemovalIntensity: 0.96,
            airRestore: 0.12,
            bodyPreserve: 0.1,
            artifactSuppression: 0.72,
            centerVocalSuppression: 0.98,
            candidateMode: 'bs_vocals',
            lowBsWeight: 0.95,
            highBsWeight: 0.62,
            inactiveLowBsWeight: 1,
            inactiveHighBsWeight: 0.98,
            crossoverHz: 3600,
            activityAbsFloorRatio: 0.03,
            activityRelFloor: 0.18,
            activitySmoothingFrames: 5,
            vocalSharpenAlpha: 0.02,
            tailGateStrength: 0.18,
            tailGateFloorRatio: 0.04,
            tailGateSmoothingFrames: 5,
            subtractiveStrength: 0,
            subtractivePower: 1,
            subtractiveResidualFloor: 0.12,
            subtractiveActiveRestore: 0,
            subtractiveTargetRmsRatio: 0,
            subtractiveMaxGain: 1.2,
            hardMaskThreshold: 0.58,
            hardMaskPower: 4.8,
            hardMaskResidualFloor: 0.015,
        },
        detail_preserve: {
            taskMode: 'detail_preserve',
            vocalIsolationIntensity: 0.34,
            bleedRemovalIntensity: 0.18,
            airRestore: 0.82,
            bodyPreserve: 0.86,
            artifactSuppression: 0.22,
            centerVocalSuppression: 0.16,
            candidateMode: 'hybrid_vocals',
            lowBsWeight: 0.68,
            highBsWeight: 0.24,
            inactiveLowBsWeight: 0.94,
            inactiveHighBsWeight: 0.88,
            crossoverHz: 2600,
            activityAbsFloorRatio: 0.08,
            activityRelFloor: 0.36,
            activitySmoothingFrames: 13,
            vocalSharpenAlpha: 0.14,
            tailGateStrength: 0,
            tailGateFloorRatio: 0.08,
            tailGateSmoothingFrames: 13,
            subtractiveStrength: 0.08,
            subtractivePower: 1,
            subtractiveResidualFloor: 0.18,
            subtractiveActiveRestore: 0.12,
            subtractiveTargetRmsRatio: 1.08,
            subtractiveMaxGain: 1.55,
            hardMaskThreshold: 0,
            hardMaskPower: 3,
            hardMaskResidualFloor: 0.02,
        },
    };

    return {
        ...custom,
        ...(presets[mode] ?? {}),
    };
}

function caseConfig(mode, overrides = {}) {
    return {
        ...taskModePreset(mode),
        ...overrides,
    };
}

function main() {
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
    const pollMs = Number(args['poll-ms'] || 1500);
    const timeoutSec = Number(args['timeout-sec'] || 3600);
    const targetStem = String(args['target-stem'] || 'vocals');
    const sourceKey = String(args['source-key'] || 'stem_vocals');
    const enableOnly = String(args['enable-only'] || 'stem_vocals');
    const outputRoot = path.resolve(process.cwd(), String(args['output-dir'] || path.join('.tmp', 'vocals-macro-batch-search')));
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outputDir = path.join(outputRoot, `track-${trackId}-${stamp}`);
    ensureDirectory(outputDir);

    const unsafeInput = args['unsafe-input']
        ? path.resolve(process.cwd(), String(args['unsafe-input']))
        : null;

    const cases = [
        {
            name: 'baseline_custom_neutral',
            modeConfig: defaultVocalsRewriteConfig(),
        },
        {
            name: 'clean_vocals_reference',
            modeConfig: taskModePreset('clean_vocals'),
        },
        {
            name: 'clean_vocals_air_push',
            modeConfig: caseConfig('clean_vocals', {
                airRestore: 0.82,
                bodyPreserve: 0.55,
                artifactSuppression: 0.28,
            }),
        },
        {
            name: 'minus_from_vocals_hard',
            modeConfig: taskModePreset('minus_from_vocals'),
        },
        {
            name: 'minus_from_vocals_brutal',
            modeConfig: caseConfig('minus_from_vocals', {
                bleedRemovalIntensity: 1,
                centerVocalSuppression: 1,
                artifactSuppression: 0.74,
            }),
        },
        {
            name: 'artifact_repair_dense',
            modeConfig: taskModePreset('artifact_repair'),
        },
        {
            name: 'artifact_repair_open',
            modeConfig: caseConfig('artifact_repair', {
                airRestore: 0.58,
                bodyPreserve: 0.84,
                artifactSuppression: 0.8,
            }),
        },
        {
            name: 'detail_preserve_reference',
            modeConfig: taskModePreset('detail_preserve'),
        },
        {
            name: 'detail_preserve_open',
            modeConfig: caseConfig('detail_preserve', {
                airRestore: 1,
                bodyPreserve: 1,
                bleedRemovalIntensity: 0.22,
            }),
        },
        {
            name: 'aggressive_vocal_suppression_hardmask',
            modeConfig: taskModePreset('aggressive_vocal_suppression'),
        },
    ];

    for (const item of cases) {
        const filePath = path.join(outputDir, `${item.name}.json`);
        writeJson(filePath, item.modeConfig);
        item.modeConfigInput = filePath;
    }

    const spec = {
        trackId,
        targetStem,
        common: {
            renderMode: 'vocals_rewrite',
            sourceKey,
            enableOnly,
            pollMs,
            timeoutSec,
            ...(unsafeInput ? { unsafeInput } : {}),
        },
        cases: cases.map((item) => ({
            name: item.name,
            modeConfigInput: item.modeConfigInput,
        })),
    };
    const specPath = path.join(outputDir, 'spec.json');
    writeJson(specPath, spec);

    const batchArgs = [
        path.resolve(process.cwd(), 'scripts', 'batch-stem-tuning-api.mjs'),
        '--spec', specPath,
        '--token', token,
        '--base-url', baseUrl,
        '--output-dir', outputDir,
        '--baseline-case', 'baseline_custom_neutral',
    ];
    if (args['continue-on-error']) {
        batchArgs.push('--continue-on-error');
    }

    runCommand(process.execPath, batchArgs, { cwd: process.cwd() });

    const reportPath = findNewestFile(outputDir, 'batch-report.json');
    if (!reportPath) {
        throw new Error(`Could not locate batch-report.json inside ${outputDir}`);
    }
    const report = readJson(reportPath);
    const ranked = report.cases
        .filter((item) => !item.isBaseline && typeof item.compare?.deltasVsBaseline?.balanced_score_delta === 'number')
        .sort((left, right) => right.compare.deltasVsBaseline.balanced_score_delta - left.compare.deltasVsBaseline.balanced_score_delta);

    const summaryPath = path.join(outputDir, 'macro-search-summary.md');
    const lines = [
        '# Vocals Macro Search Summary',
        '',
        `- Track: ${trackId}`,
        `- Target stem: ${targetStem}`,
        `- Baseline: baseline_custom_neutral`,
        '',
    ];
    if (ranked.length === 0) {
        lines.push('No successful comparable candidates were produced.');
    } else {
        const best = ranked[0];
        lines.push(`## Best candidate`);
        lines.push('');
        lines.push(`- Name: ${best.name}`);
        lines.push(`- Balanced delta vs baseline: ${best.compare.deltasVsBaseline.balanced_score_delta}`);
        lines.push(`- Balanced score: ${best.compare.comparisons.scores.balanced_score}`);
        lines.push(`- Cleanliness score: ${best.compare.comparisons.scores.cleanliness_score}`);
        lines.push(`- Preservation score: ${best.compare.comparisons.scores.preservation_score}`);
        lines.push('');
        lines.push('## Ranking');
        lines.push('');
        ranked.forEach((item, index) => {
            lines.push(`${index + 1}. ${item.name}: delta ${item.compare.deltasVsBaseline.balanced_score_delta}`);
        });
    }
    fs.writeFileSync(summaryPath, lines.join('\n'));

    console.log(JSON.stringify({
        outputDir,
        specPath,
        reportPath,
        summaryPath,
    }, null, 2));
}

try {
    main();
} catch (error) {
    console.error(`[vocals-macro-batch-search] ${error.message}`);
    process.exit(1);
}
