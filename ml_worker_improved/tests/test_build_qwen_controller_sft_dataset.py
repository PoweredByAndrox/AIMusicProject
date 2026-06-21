from __future__ import annotations

import json
from pathlib import Path

import tools.build_qwen_controller_sft_dataset as qwen_sft
from tools.build_qwen_controller_sft_dataset import build_qwen_controller_sft_dataset, main


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def controller_row(
    row_id: str,
    *,
    stem: str = "vocal",
    route: str = "route-good",
    accepted: bool = True,
    manifest_path: str | None = None,
    failure: str = "",
) -> dict:
    return {
        "schema": "generic_finetuning_controller_decision_example.v1",
        "id": row_id,
        "track_id": f"track-{row_id}",
        "stem_type": stem,
        "input": {
            "manifest_path": manifest_path or f"/app/lab_artifacts/controller/{row_id}/manifest.json",
            "model_id": route,
            "delivery_status": "accepted" if accepted else "blocked",
            "presentation_status": "accepted_for_listen_pack" if accepted else "blocked_from_listen_pack",
            "failure_labels": [failure] if failure else [],
        },
        "target": {
            "action": "reuse_route_candidate" if accepted else "avoid_route_and_generate_alternative",
            "chosen_model": route,
            "acceptance_tests": ["artifact_status_contract"],
        },
        "result": {
            "accepted": accepted,
            "human_verified": accepted,
            "human_rating": "ideal" if accepted else "bad",
            "reference_verified": False,
        },
        "trainable": True,
        "training_target": "llm_controller",
    }


def test_builds_qwen_sft_dataset_with_provenance_and_progress(tmp_path: Path) -> None:
    train_examples = write_jsonl(
        tmp_path / "controller_train.jsonl",
        [
            controller_row("train-accept", route="route-a", accepted=True),
            controller_row("train-reject", route="route-b", accepted=False, failure="severe_bleed"),
        ],
    )
    eval_examples = write_jsonl(
        tmp_path / "controller_eval.jsonl",
        [controller_row("eval-reject", route="route-b", accepted=False, failure="phase_collapse")],
    )

    manifest = build_qwen_controller_sft_dataset(
        controller_examples_jsonl=train_examples,
        output_dir=tmp_path / "out",
        eval_examples_jsonl=eval_examples,
    )

    assert manifest["schema"] == "qwen_controller_sft_dataset_manifest.v1"
    assert manifest["dataset_status"] == "ready"
    assert manifest["training_started"] is False
    assert manifest["fine_tune_started"] is False
    assert manifest["models_changed"] is False
    assert manifest["model_registry_edited"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["safety"]["metadata_only"] is True
    assert manifest["safety"]["blocks_train_eval_overlap"] is True
    assert manifest["summary"]["train_examples"] == 2
    assert manifest["summary"]["eval_examples"] == 1
    assert Path(manifest["outputs"]["train_jsonl"]).name == "train.jsonl"
    assert Path(manifest["outputs"]["eval_jsonl"]).name == "eval.jsonl"
    assert Path(manifest["outputs"]["manifest_json"]).name == "manifest.json"

    train_rows = read_jsonl(Path(manifest["outputs"]["train_jsonl"]))
    assert train_rows[0]["schema"] == "qwen_controller_sft_example.v1"
    assert [message["role"] for message in train_rows[0]["messages"]] == ["system", "user", "assistant"]
    assistant_payload = json.loads(train_rows[0]["messages"][-1]["content"])
    assert assistant_payload == {
        "accepted": True,
        "action": "reuse_route_candidate",
        "route_id": "route-a",
        "stem_type": "vocal",
    }
    assert train_rows[0]["metadata"]["split"] == "train"
    assert train_rows[0]["metadata"]["provenance"]["source_row_id"] == "train-accept"
    assert train_rows[0]["metadata"]["provenance"]["manifest_path"].endswith("/train-accept/manifest.json")
    assert train_rows[0]["metadata"]["source"]["controller_examples_jsonl"] == str(train_examples)
    assert train_rows[0]["training_target"] == "qwen_controller_sft"
    assert train_rows[0]["delivery_allowed"] is False
    assert train_rows[0]["release_ready"] is False

    progress_rows = read_jsonl(Path(manifest["outputs"]["progress_jsonl"]))
    assert [row["event"] for row in progress_rows] == ["start", "examples_loaded", "write_outputs", "finish"]
    assert progress_rows[0]["gpu_inference_skipped"] is True
    assert progress_rows[0]["expected_acceleration_mode"] == "external_only_cpu_metadata_dataset_build"

    written_manifest = read_json(Path(manifest["outputs"]["manifest_json"]))
    assert written_manifest["fingerprint"] == manifest["fingerprint"]


def test_blocks_train_eval_manifest_overlap(tmp_path: Path) -> None:
    shared_manifest = "/app/lab_artifacts/shared/manifest.json"
    train_examples = write_jsonl(
        tmp_path / "controller_train.jsonl",
        [controller_row("train-shared", manifest_path=shared_manifest)],
    )
    eval_examples = write_jsonl(
        tmp_path / "controller_eval.jsonl",
        [controller_row("eval-shared", manifest_path=shared_manifest)],
    )

    manifest = build_qwen_controller_sft_dataset(
        controller_examples_jsonl=train_examples,
        output_dir=tmp_path / "out",
        eval_examples_jsonl=eval_examples,
    )

    assert manifest["dataset_status"] == "blocked"
    assert manifest["failed_checks"][0]["check"] == "train_eval_manifest_overlap"
    assert manifest["failed_checks"][0]["overlap_manifest_paths"] == [shared_manifest]
    assert manifest["summary"]["eval_examples"] == 0
    assert manifest["summary"]["excluded_examples"] == 1
    assert read_jsonl(Path(manifest["outputs"]["eval_jsonl"])) == []
    excluded_rows = read_jsonl(Path(manifest["outputs"]["excluded_jsonl"]))
    assert excluded_rows[0]["source_row_id"] == "eval-shared"
    assert excluded_rows[0]["exclude_reason"] == "train_eval_manifest_overlap"


def test_blocks_train_eval_manifest_overlap_after_path_normalization(tmp_path: Path) -> None:
    train_examples = write_jsonl(
        tmp_path / "controller_train.jsonl",
        [controller_row("train-shared", manifest_path=r"\app\lab\shared\..\shared\manifest.json")],
    )
    eval_examples = write_jsonl(
        tmp_path / "controller_eval.jsonl",
        [controller_row("eval-shared", manifest_path="/app/lab/shared/manifest.json")],
    )

    manifest = build_qwen_controller_sft_dataset(
        controller_examples_jsonl=train_examples,
        output_dir=tmp_path / "out",
        eval_examples_jsonl=eval_examples,
    )

    assert manifest["dataset_status"] == "blocked"
    assert manifest["failed_checks"][0]["check"] == "train_eval_manifest_overlap"
    assert manifest["failed_checks"][0]["overlap_manifest_paths"] == ["/app/lab/shared/manifest.json"]
    excluded_rows = read_jsonl(Path(manifest["outputs"]["excluded_jsonl"]))
    assert excluded_rows[0]["source_row_id"] == "eval-shared"
    assert excluded_rows[0]["exclude_reason"] == "train_eval_manifest_overlap"


def test_excludes_rows_that_claim_delivery_or_release(tmp_path: Path) -> None:
    unsafe = controller_row("unsafe", accepted=True)
    unsafe["delivery_allowed"] = True
    unsafe["release_ready"] = True
    nested_unsafe = controller_row("nested-unsafe", accepted=True)
    nested_unsafe["input"]["delivery_allowed"] = True
    nested_unsafe["input"]["release_ready"] = True
    safe = controller_row("safe", accepted=False, route="route-bad", failure="bleed")
    train_examples = write_jsonl(tmp_path / "controller_train.jsonl", [unsafe, nested_unsafe, safe])

    manifest = build_qwen_controller_sft_dataset(
        controller_examples_jsonl=train_examples,
        output_dir=tmp_path / "out",
    )

    assert manifest["dataset_status"] == "ready"
    assert manifest["summary"]["train_examples"] == 1
    assert manifest["summary"]["excluded_examples"] == 2
    train_rows = read_jsonl(Path(manifest["outputs"]["train_jsonl"]))
    excluded_rows = read_jsonl(Path(manifest["outputs"]["excluded_jsonl"]))
    assert train_rows[0]["metadata"]["provenance"]["source_row_id"] == "safe"
    assert {row["source_row_id"] for row in excluded_rows} == {"nested-unsafe", "unsafe"}
    assert {row["exclude_reason"] for row in excluded_rows} == {"source_row_claims_delivery_or_release"}


def test_excludes_wrong_schema_rows(tmp_path: Path) -> None:
    wrong_schema = controller_row("wrong-schema", accepted=True)
    wrong_schema["schema"] = "unrelated_training_example.v1"
    safe = controller_row("safe", accepted=False, route="route-bad", failure="bleed")
    train_examples = write_jsonl(tmp_path / "controller_train.jsonl", [wrong_schema, safe])

    manifest = build_qwen_controller_sft_dataset(
        controller_examples_jsonl=train_examples,
        output_dir=tmp_path / "out",
    )

    assert manifest["dataset_status"] == "ready"
    assert manifest["summary"]["train_examples"] == 1
    train_rows = read_jsonl(Path(manifest["outputs"]["train_jsonl"]))
    excluded_rows = read_jsonl(Path(manifest["outputs"]["excluded_jsonl"]))
    assert train_rows[0]["metadata"]["provenance"]["source_row_id"] == "safe"
    assert excluded_rows[0]["source_row_id"] == "wrong-schema"
    assert excluded_rows[0]["exclude_reason"] == "unexpected_source_schema"


def test_qwen_sft_fingerprint_is_stable(tmp_path: Path) -> None:
    examples = write_jsonl(
        tmp_path / "controller.jsonl",
        [
            controller_row("a", route="route-a", accepted=True),
            controller_row("b", route="route-b", accepted=False, failure="bleed"),
        ],
    )

    first = build_qwen_controller_sft_dataset(
        controller_examples_jsonl=examples,
        output_dir=tmp_path / "out1",
    )
    second = build_qwen_controller_sft_dataset(
        controller_examples_jsonl=examples,
        output_dir=tmp_path / "out2",
    )

    assert first["fingerprint"] == second["fingerprint"]


def test_cli_checks_docker_runtime_before_parsing_args(monkeypatch) -> None:
    calls = []

    def reject_non_docker_runtime(**kwargs):
        calls.append(("require_docker_runtime", kwargs))
        raise RuntimeError("Docker-only test sentinel")

    def fail_if_args_are_parsed(self, argv=None):
        raise AssertionError("parse_args must not run before Docker runtime validation")

    monkeypatch.setattr(qwen_sft, "require_docker_runtime", reject_non_docker_runtime)
    monkeypatch.setattr(qwen_sft.argparse.ArgumentParser, "parse_args", fail_if_args_are_parsed)

    try:
        main(["--controller-examples-jsonl", "/tmp/controller.jsonl", "--output-dir", "/tmp/out"])
    except RuntimeError as exc:
        assert str(exc) == "Docker-only test sentinel"
    else:
        raise AssertionError("main must fail through Docker runtime guard")

    assert calls == [
        (
            "require_docker_runtime",
            {
                "tool_name": "build_qwen_controller_sft_dataset.py",
                "entrypoint_path": "/app/tools/build_qwen_controller_sft_dataset.py",
            },
        )
    ]


def test_cli_writes_qwen_sft_manifest_and_short_outputs(tmp_path: Path) -> None:
    examples = write_jsonl(
        tmp_path / "controller.jsonl",
        [controller_row("cli-a", route="route-a", accepted=True)],
    )

    assert main(["--controller-examples-jsonl", str(examples), "--output-dir", str(tmp_path / "out")]) == 0

    manifest = read_json(tmp_path / "out" / "manifest.json")
    assert manifest["dataset_status"] == "ready"
    assert manifest["summary"]["train_examples"] == 1
    assert (tmp_path / "out" / "train.jsonl").exists()
    assert (tmp_path / "out" / "eval.jsonl").exists()
    assert (tmp_path / "out" / "excluded.jsonl").exists()
    assert (tmp_path / "out" / "progress.jsonl").exists()
