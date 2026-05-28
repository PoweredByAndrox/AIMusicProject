#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

function printHelp() {
    console.log(`Usage:
  node scripts/batch-stem-tuning-api.mjs --spec <file> --token <API_TOKEN> [options]

Required:
  --spec <file>                   Batch spec JSON file.
  --token <token>                 API bearer token, or set MUSIC_AI_API_TOKEN.

Optional:
  --base-url <url>                Default: http://localhost:8000
  --output-dir <dir>              Default: .tmp/stem-tuning-batch
  --continue-on-error             Continue after a failed case
  --baseline-case <name>          Explicit case name to treat as baseline

Spec shape:
{
  "trackId": 111,
  "targetStem": "vocals",
  "common": {
    "renderMode": "vocals_rewrite",
    "sourceKey": "stem_vocals",
    "enableOnly": "stem_vocals",
    "unsafeInput": "D:/path/unsafe.json",
    "pollMs": 1500,
    "compareSegmentsInput": "D:/path/segments.json",
    "compareSegmentTag": "vocal_drop",
    "compareSegmentLimit": 3,
    "compareUseSegmentWindows": true
  },
  "cases": [
    { "name": "baseline", "modeConfigInput": "D:/path/base.json" },
    { "name": "cleaner", "modeConfigInput": "D:/path/cleaner.json" }
  ]
}
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

function readJson(filePath) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, ''));
}

function ensureDirectory(dirPath) {
    fs.mkdirSync(dirPath, { recursive: true });
}

function toMlWorkerPath(filePath) {
    const absolutePath = path.resolve(String(filePath));
    const workspaceRoot = path.resolve(process.cwd());
    const relativePath = path.relative(workspaceRoot, absolutePath);
    if (relativePath && !relativePath.startsWith('..') && !path.isAbsolute(relativePath)) {
        return `/app/${relativePath.replace(/\\/g, '/')}`;
    }
    return absolutePath.replace(/\\/g, '/');
}

function slugify(value) {
    return String(value)
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '')
        || 'case';
}

function toHarnessArgs({ token, baseUrl, caseConfig, trackId, outputDir, summaryPath }) {
    const args = [
        path.resolve(process.cwd(), 'scripts', 'test-stem-tuning-api.mjs'),
        '--track', String(trackId),
        '--token', token,
        '--base-url', baseUrl,
        '--summary-path', summaryPath,
        '--output-dir', outputDir,
    ];

    const mapping = [
        ['presetInput', '--preset-input'],
        ['modeConfigInput', '--mode-config-input'],
        ['modeConfigField', '--mode-config-field'],
        ['unsafeInput', '--unsafe-input'],
        ['renderMode', '--render-mode'],
        ['sourceKey', '--source-key'],
        ['enableOnly', '--enable-only'],
        ['pollMs', '--poll-ms'],
        ['timeoutSec', '--timeout-sec'],
    ];

    for (const [key, flag] of mapping) {
        if (caseConfig[key] === undefined || caseConfig[key] === null || caseConfig[key] === '') {
            continue;
        }
        args.push(flag, String(caseConfig[key]));
    }

    if (caseConfig.buildIfMissing) {
        args.push('--build-if-missing');
    }

    return args;
}

function runCommand(command, args, { cwd, inheritStdout = true, captureStdout = false, input } = {}) {
    const result = spawnSync(command, args, {
        cwd,
        stdio: captureStdout
            ? ['pipe', 'pipe', 'inherit']
            : inheritStdout
                ? ['pipe', 'inherit', 'inherit']
                : ['pipe', 'ignore', 'inherit'],
        encoding: captureStdout ? 'utf8' : undefined,
        input,
    });
    if (result.status !== 0) {
        throw new Error(`Command failed: ${command} ${args.join(' ')}`);
    }
    return result.stdout ?? '';
}

function buildMarkdownReport(report) {
    const lines = [];
    lines.push(`# Stem Tuning Batch Report`);
    lines.push('');
    lines.push(`- Track: ${report.trackId}`);
    lines.push(`- Target stem: ${report.targetStem}`);
    lines.push(`- Baseline case: ${report.baselineCaseName ?? 'none'}`);
    lines.push(`- Generated at: ${report.generatedAt}`);
    lines.push('');
    lines.push(`| Case | Status | Render ID | Balanced | Cleanliness | Preservation | Delta vs baseline | Verdict |`);
    lines.push(`| --- | --- | --- | ---: | ---: | ---: | ---: | --- |`);
    for (const item of report.cases) {
        const balanced = item.compare?.comparisons?.scores?.balanced_score;
        const cleanliness = item.compare?.comparisons?.scores?.cleanliness_score;
        const preservation = item.compare?.comparisons?.scores?.preservation_score;
        const delta = item.compare?.deltasVsBaseline?.balanced_score_delta;
        const verdict = item.compare?.deltasVsBaseline?.verdict ?? (item.isBaseline ? 'baseline' : '');
        lines.push(`| ${item.name} | ${item.status} | ${item.renderId ?? ''} | ${balanced ?? ''} | ${cleanliness ?? ''} | ${preservation ?? ''} | ${delta ?? ''} | ${verdict} |`);
    }
    lines.push('');
    lines.push(`## Ranking`);
    lines.push('');
    const ranked = report.cases
        .filter((item) => !item.isBaseline && typeof item.compare?.deltasVsBaseline?.balanced_score_delta === 'number')
        .sort((left, right) => right.compare.deltasVsBaseline.balanced_score_delta - left.compare.deltasVsBaseline.balanced_score_delta);
    if (ranked.length === 0) {
        lines.push(`No comparable non-baseline cases completed.`);
    } else {
        ranked.forEach((item, index) => {
            lines.push(`${index + 1}. ${item.name}: balanced delta ${item.compare.deltasVsBaseline.balanced_score_delta}`);
        });
    }
    lines.push('');
    lines.push(`## Notes`);
    lines.push('');
    lines.push(`- Cleanliness rewards lower accompaniment similarity.`);
    lines.push(`- Preservation rewards closeness to the base target stem, with penalties for loudness and high-band drift.`);
    lines.push(`- Balanced score uses 60% preservation and 40% cleanliness.`);
    lines.push(`- This is still not ground-truth SDR/SIR/SAR because there is no oracle vocal truth here.`);
    return lines.join('\n');
}

function main() {
    const args = parseArgs(process.argv.slice(2));
    if (args.help || args.h) {
        printHelp();
        process.exit(0);
    }

    const specPath = args.spec ? path.resolve(process.cwd(), args.spec) : null;
    if (!specPath || !fs.existsSync(specPath)) {
        throw new Error('--spec is required and must point to an existing JSON file');
    }

    const token = args.token || process.env.MUSIC_AI_API_TOKEN || process.env.MUSIC_AI_TOKEN;
    if (!token) {
        throw new Error('API token is required. Use --token or set MUSIC_AI_API_TOKEN.');
    }

    const baseUrl = String(args['base-url'] || 'http://localhost:8000').replace(/\/$/, '');
    const batchSpec = readJson(specPath);
    const trackId = Number(batchSpec.trackId);
    const targetStem = String(batchSpec.targetStem || 'vocals');
    const cases = Array.isArray(batchSpec.cases) ? batchSpec.cases : [];
    if (!Number.isInteger(trackId) || trackId <= 0) {
        throw new Error('Spec trackId must be a positive integer');
    }
    if (cases.length === 0) {
        throw new Error('Spec cases[] is empty');
    }

    const outputRoot = path.resolve(process.cwd(), String(args['output-dir'] || path.join('.tmp', 'stem-tuning-batch')));
    const compareScriptSource = fs.readFileSync(path.resolve(process.cwd(), 'scripts', 'compare-stem-render.py'), 'utf8');
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outputDir = path.join(outputRoot, `track-${trackId}-${stamp}`);
    ensureDirectory(outputDir);

    const baselineCaseName = args['baseline-case'] ? String(args['baseline-case']) : String(cases[0].name);
    const report = {
        trackId,
        targetStem,
        baselineCaseName,
        generatedAt: new Date().toISOString(),
        specPath,
        outputDir,
        cases: [],
    };

    let baselineRenderId = null;

    for (const rawCase of cases) {
        const name = String(rawCase.name || 'unnamed');
        const caseSlug = slugify(name);
        const mergedCase = {
            ...(batchSpec.common || {}),
            ...rawCase,
        };
        const summaryPath = path.join(outputDir, `${caseSlug}-summary.json`);
        const comparePath = path.join(outputDir, `${caseSlug}-compare.json`);
        const caseRecord = {
            name,
            isBaseline: name === baselineCaseName,
            summaryPath,
            comparePath,
            status: 'pending',
            renderId: null,
            compare: null,
            error: null,
        };

        try {
            console.log(`[stem-tuning-batch] Running case ${name}`);
            const harnessArgs = toHarnessArgs({
                token,
                baseUrl,
                caseConfig: mergedCase,
                trackId,
                outputDir,
                summaryPath,
            });
            runCommand(process.execPath, harnessArgs, {
                cwd: process.cwd(),
                inheritStdout: true,
            });

            const summary = readJson(summaryPath);
            caseRecord.status = summary.status;
            caseRecord.renderId = summary.renderId ?? null;
            caseRecord.summary = summary;

            if (summary.status !== 'completed' || !summary.renderId) {
                throw new Error(`Case ${name} did not complete successfully`);
            }

            if (caseRecord.isBaseline) {
                baselineRenderId = summary.renderId;
            }

            const compareArgs = [
                'compose', 'exec', '-T', 'ml-worker',
                'python', '-',
                '--track-id', String(trackId),
                '--render-id', String(summary.renderId),
                '--target-stem', targetStem,
            ];
            if (baselineRenderId && !caseRecord.isBaseline) {
                compareArgs.push('--baseline-render-id', baselineRenderId);
            }
            if (mergedCase.compareSegmentsInput && mergedCase.compareSegmentTag) {
                compareArgs.push('--segments-json', toMlWorkerPath(String(mergedCase.compareSegmentsInput)));
                compareArgs.push('--segment-tag', String(mergedCase.compareSegmentTag));
                if (mergedCase.compareSegmentLimit !== undefined && mergedCase.compareSegmentLimit !== null && mergedCase.compareSegmentLimit !== '') {
                    compareArgs.push('--segment-limit', String(mergedCase.compareSegmentLimit));
                }
                if (mergedCase.compareUseSegmentWindows) {
                    compareArgs.push('--use-segment-windows');
                }
            }

            const compareStdout = runCommand('docker', compareArgs, {
                cwd: process.cwd(),
                captureStdout: true,
                input: compareScriptSource,
            });
            const comparePayload = JSON.parse(compareStdout.replace(/^\uFEFF/, ''));
            fs.writeFileSync(comparePath, JSON.stringify(comparePayload, null, 2));
            caseRecord.compare = comparePayload;
        } catch (error) {
            caseRecord.status = caseRecord.status === 'pending' ? 'failed' : caseRecord.status;
            caseRecord.error = error.message;
            if (!args['continue-on-error']) {
                report.cases.push(caseRecord);
                break;
            }
        }

        report.cases.push(caseRecord);
    }

    const reportJsonPath = path.join(outputDir, 'batch-report.json');
    const reportMdPath = path.join(outputDir, 'batch-report.md');
    fs.writeFileSync(reportJsonPath, JSON.stringify(report, null, 2));
    fs.writeFileSync(reportMdPath, buildMarkdownReport(report));

    console.log(JSON.stringify({
        outputDir,
        reportJsonPath,
        reportMdPath,
        baselineCaseName,
        baselineRenderId,
    }, null, 2));
}

try {
    main();
} catch (error) {
    console.error(`[stem-tuning-batch] ${error.message}`);
    process.exit(1);
}
