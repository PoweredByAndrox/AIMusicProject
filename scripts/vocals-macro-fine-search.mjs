#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

function printHelp() {
    console.log(`Usage:
  node scripts/vocals-macro-fine-search.mjs --track <id> --token <API_TOKEN> [options]

Required:
  --track <id>                    Track id.
  --token <token>                 API bearer token, or set MUSIC_AI_API_TOKEN.

Optional:
  --base-url <url>                Default: http://localhost:8000
  --unsafe-input <json|file>      Unsafe internals JSON like "Apply unsafe internals JSON".
  --output-dir <dir>              Default: .tmp/vocals-macro-fine-search
  --poll-ms <ms>                  Default: 1500
  --timeout-sec <sec>             Default: 3600
  --target-stem <stem>            Default: vocals
  --source-key <key>              Default: stem_vocals
  --enable-only <csv>             Default: stem_vocals
  --top-k <n>                     Default: 3
  --compare-segments-input <file> Segment definitions JSON for compare-scoring only.
  --compare-segment-tag <tag>     Example: vocal_drop
  --compare-segment-limit <n>     Limit selected compare windows.
  --compare-use-segment-windows   Use refined segment windows instead of whole segments.
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

function writeText(filePath, contents) {
    fs.writeFileSync(filePath, contents);
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

function mergeConfig(base, overrides = {}) {
    return {
        ...base,
        ...overrides,
    };
}

function buildFineCases() {
    const baseline = defaultVocalsRewriteConfig();

    return [
        {
            name: 'baseline_custom_neutral',
            modeConfig: baseline,
            note: 'Neutral strong baseline around current best preserving region.',
        },
        {
            name: 'detail_preserve_reference',
            modeConfig: mergeConfig(baseline, {
                taskMode: 'detail_preserve',
                vocalIsolationIntensity: 0.16,
                bleedRemovalIntensity: 0.08,
                airRestore: 0.26,
                bodyPreserve: 0.24,
                artifactSuppression: 0.1,
                centerVocalSuppression: 0.08,
                vocalSharpenAlpha: 0.04,
                subtractiveStrength: 0.04,
                subtractiveActiveRestore: 0.03,
                subtractiveTargetRmsRatio: 1.02,
            }),
            note: 'Closest soft detail-preserve neighbor to baseline.',
        },
        {
            name: 'detail_preserve_air_lift',
            modeConfig: mergeConfig(baseline, {
                taskMode: 'detail_preserve',
                airRestore: 0.38,
                bodyPreserve: 0.18,
                artifactSuppression: 0.08,
                vocalSharpenAlpha: 0.06,
                subtractiveStrength: 0.03,
                subtractiveActiveRestore: 0.04,
            }),
            note: 'Soft high-end opening without aggressive cleanup.',
        },
        {
            name: 'detail_preserve_body_lock',
            modeConfig: mergeConfig(baseline, {
                taskMode: 'detail_preserve',
                airRestore: 0.12,
                bodyPreserve: 0.42,
                artifactSuppression: 0.12,
                subtractiveStrength: 0.02,
                subtractiveActiveRestore: 0.05,
                subtractiveTargetRmsRatio: 1.04,
            }),
            note: 'Keeps more body and center weight while staying soft.',
        },
        {
            name: 'detail_preserve_balanced',
            modeConfig: mergeConfig(baseline, {
                taskMode: 'detail_preserve',
                vocalIsolationIntensity: 0.12,
                bleedRemovalIntensity: 0.1,
                airRestore: 0.22,
                bodyPreserve: 0.28,
                artifactSuppression: 0.14,
                centerVocalSuppression: 0.1,
                vocalSharpenAlpha: 0.05,
                subtractiveStrength: 0.05,
                subtractiveActiveRestore: 0.04,
            }),
            note: 'Slightly more action than reference, still in the safe zone.',
        },
        {
            name: 'clean_vocals_soft',
            modeConfig: mergeConfig(baseline, {
                taskMode: 'clean_vocals',
                vocalIsolationIntensity: 0.2,
                bleedRemovalIntensity: 0.16,
                airRestore: 0.16,
                bodyPreserve: 0.12,
                artifactSuppression: 0.18,
                centerVocalSuppression: 0.1,
                vocalSharpenAlpha: 0.03,
                tailGateStrength: 0.02,
                subtractiveStrength: 0.06,
                subtractivePower: 1.04,
                subtractiveResidualFloor: 0.14,
                subtractiveActiveRestore: 0.02,
                subtractiveTargetRmsRatio: 1.0,
            }),
            note: 'Gentle cleanup profile, not the harsh clean_vocals preset.',
        },
        {
            name: 'clean_vocals_air_push',
            modeConfig: mergeConfig(baseline, {
                taskMode: 'clean_vocals',
                vocalIsolationIntensity: 0.18,
                bleedRemovalIntensity: 0.14,
                airRestore: 0.34,
                bodyPreserve: 0.14,
                artifactSuppression: 0.16,
                centerVocalSuppression: 0.08,
                vocalSharpenAlpha: 0.05,
                tailGateStrength: 0.01,
                subtractiveStrength: 0.05,
                subtractivePower: 1.03,
                subtractiveResidualFloor: 0.15,
            }),
            note: 'Opens the air band more while keeping cleanup mild.',
        },
        {
            name: 'artifact_repair_gentle',
            modeConfig: mergeConfig(baseline, {
                taskMode: 'artifact_repair',
                vocalIsolationIntensity: 0.08,
                bleedRemovalIntensity: 0.08,
                airRestore: 0.08,
                bodyPreserve: 0.24,
                artifactSuppression: 0.28,
                centerVocalSuppression: 0.04,
                vocalSharpenAlpha: 0.01,
                tailGateStrength: 0,
                subtractiveStrength: 0.03,
                subtractiveResidualFloor: 0.16,
                subtractiveActiveRestore: 0.05,
                subtractiveTargetRmsRatio: 1.05,
            }),
            note: 'Targeted artifact damping with low collateral damage.',
        },
        {
            name: 'artifact_repair_open',
            modeConfig: mergeConfig(baseline, {
                taskMode: 'artifact_repair',
                vocalIsolationIntensity: 0.06,
                bleedRemovalIntensity: 0.06,
                airRestore: 0.2,
                bodyPreserve: 0.3,
                artifactSuppression: 0.26,
                centerVocalSuppression: 0.04,
                vocalSharpenAlpha: 0.02,
                subtractiveStrength: 0.02,
                subtractiveResidualFloor: 0.17,
                subtractiveActiveRestore: 0.06,
                subtractiveTargetRmsRatio: 1.06,
            }),
            note: 'Repair-biased profile that preserves openness and body.',
        },
    ];
}

function copyTopConfigs({ outputDir, rankedCases, topK, baselineCase, baselineModeConfig }) {
    const exportDir = path.join(outputDir, 'top-3-ab-pack');
    ensureDirectory(exportDir);

    const baselineFilePath = path.join(exportDir, '0-baseline_custom_neutral.json');
    writeJson(baselineFilePath, baselineModeConfig);

    const top = rankedCases.slice(0, topK);
    const exported = [];
    top.forEach((item, index) => {
        const rank = index + 1;
        const safeName = item.name.replace(/[^a-z0-9_-]/gi, '_').toLowerCase();
        const filePath = path.join(exportDir, `${rank}-${safeName}.json`);
        writeJson(filePath, item.inputModeConfig);
        exported.push({
            rank,
            name: item.name,
            filePath,
            renderId: item.renderId,
            delta: item.compare.deltasVsBaseline.balanced_score_delta,
            balanced: item.compare.comparisons.scores.balanced_score,
            cleanliness: item.compare.comparisons.scores.cleanliness_score,
            preservation: item.compare.comparisons.scores.preservation_score,
            note: item.inputNote ?? '',
        });
    });

    return {
        exportDir,
        baseline: {
            name: baselineCase.name,
            filePath: baselineFilePath,
            renderId: baselineCase.renderId,
            balanced: baselineCase.compare.comparisons.scores.balanced_score,
            cleanliness: baselineCase.compare.comparisons.scores.cleanliness_score,
            preservation: baselineCase.compare.comparisons.scores.preservation_score,
        },
        exported,
    };
}

function buildMarkdownSummary({ trackId, targetStem, baselineCase, baselineScore, rankedCases, topExport }) {
    const lines = [
        '# Vocals Macro Fine Search',
        '',
        `- Track: ${trackId}`,
        `- Target stem: ${targetStem}`,
        `- Baseline: ${baselineCase.name}`,
        `- Baseline balanced score: ${baselineScore.toFixed(6)}`,
        `- Baseline JSON: ${topExport.baseline.filePath}`,
        '',
        '## Top 3 for A/B',
        '',
    ];

    if (topExport.exported.length === 0) {
        lines.push('No successful comparable candidates were produced.');
    } else {
        topExport.exported.forEach((item) => {
            lines.push(`### ${item.rank}. ${item.name}`);
            lines.push('');
            lines.push(`- Render id: ${item.renderId}`);
            lines.push(`- Balanced delta vs baseline: ${item.delta.toFixed(6)}`);
            lines.push(`- Balanced score: ${item.balanced.toFixed(6)}`);
            lines.push(`- Cleanliness score: ${item.cleanliness.toFixed(6)}`);
            lines.push(`- Preservation score: ${item.preservation.toFixed(6)}`);
            lines.push(`- Mode config JSON: ${item.filePath}`);
            if (item.note) {
                lines.push(`- Intent: ${item.note}`);
            }
            lines.push('');
        });
    }

    lines.push('## Full ranking');
    lines.push('');
    if (rankedCases.length === 0) {
        lines.push('No comparable successful cases.');
    } else {
        rankedCases.forEach((item, index) => {
            const scores = item.compare.comparisons.scores;
            lines.push(`${index + 1}. ${item.name}`);
            lines.push(`   delta=${item.compare.deltasVsBaseline.balanced_score_delta.toFixed(6)} balanced=${scores.balanced_score.toFixed(6)} cleanliness=${scores.cleanliness_score.toFixed(6)} preservation=${scores.preservation_score.toFixed(6)}`);
        });
    }

    lines.push('');
    lines.push('## Listening order');
    lines.push('');
    lines.push('1. Baseline');
    topExport.exported.forEach((item) => {
        lines.push(`${item.rank + 1}. ${item.name}`);
    });
    lines.push('');
    lines.push('Compare on the same short problem segment first, not the full track.');

    return `${lines.join('\n')}\n`;
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
    const topK = Math.max(1, Number(args['top-k'] || 3));
    const outputRoot = path.resolve(process.cwd(), String(args['output-dir'] || path.join('.tmp', 'vocals-macro-fine-search')));
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outputDir = path.join(outputRoot, `track-${trackId}-${stamp}`);
    ensureDirectory(outputDir);

    const unsafeInput = args['unsafe-input']
        ? path.resolve(process.cwd(), String(args['unsafe-input']))
        : null;

    const cases = buildFineCases();
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
            ...(args['compare-segments-input'] ? { compareSegmentsInput: path.resolve(process.cwd(), String(args['compare-segments-input'])) } : {}),
            ...(args['compare-segment-tag'] ? { compareSegmentTag: String(args['compare-segment-tag']) } : {}),
            ...(args['compare-segment-limit'] ? { compareSegmentLimit: Number(args['compare-segment-limit']) } : {}),
            ...(args['compare-use-segment-windows'] ? { compareUseSegmentWindows: true } : {}),
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
    const byName = new Map(cases.map((item) => [item.name, item]));
    const baselineCase = report.cases.find((item) => item.isBaseline) ?? report.cases.find((item) => item.name === 'baseline_custom_neutral');
    if (!baselineCase?.compare?.comparisons?.scores?.balanced_score) {
        throw new Error('Baseline case is missing a balanced score in batch report.');
    }

    const ranked = report.cases
        .filter((item) => !item.isBaseline && typeof item.compare?.deltasVsBaseline?.balanced_score_delta === 'number')
        .map((item) => ({
            ...item,
            inputModeConfig: byName.get(item.name)?.modeConfig ?? {},
            inputNote: byName.get(item.name)?.note ?? '',
        }))
        .sort((left, right) => right.compare.deltasVsBaseline.balanced_score_delta - left.compare.deltasVsBaseline.balanced_score_delta);

    const topExport = copyTopConfigs({
        outputDir,
        rankedCases: ranked,
        topK,
        baselineCase,
        baselineModeConfig: byName.get(baselineCase.name)?.modeConfig ?? defaultVocalsRewriteConfig(),
    });

    const abSummaryPath = path.join(outputDir, 'ab-top-3-summary.md');
    writeText(abSummaryPath, buildMarkdownSummary({
        trackId,
        targetStem,
        baselineCase,
        baselineScore: baselineCase.compare.comparisons.scores.balanced_score,
        rankedCases: ranked,
        topExport,
    }));

    const summaryJsonPath = path.join(outputDir, 'fine-search-summary.json');
    writeJson(summaryJsonPath, {
        trackId,
        targetStem,
        baselineCase: {
            name: baselineCase.name,
            renderId: baselineCase.renderId,
            balanced: baselineCase.compare.comparisons.scores.balanced_score,
            cleanliness: baselineCase.compare.comparisons.scores.cleanliness_score,
            preservation: baselineCase.compare.comparisons.scores.preservation_score,
        },
        topExport,
        outputDir,
        specPath,
        reportPath,
        abSummaryPath,
    });

    console.log(JSON.stringify({
        outputDir,
        specPath,
        reportPath,
        abSummaryPath,
        summaryJsonPath,
        topExport,
    }, null, 2));
}

try {
    main();
} catch (error) {
    console.error(`[vocals-macro-fine-search] ${error.message}`);
    process.exit(1);
}
