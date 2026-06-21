from __future__ import annotations

import json
from pathlib import Path

import tools.qwen_controller_qlora_eval as qwen_eval
from tools.qwen_controller_qlora_eval import main, run_qwen_controller_qlora_eval


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sft_row(row_id: str, *, accepted: bool, route_id: str = "route-a") -> dict:
    action = "reuse_route_candidate" if accepted else "avoid_route_and_generate_alternative"
    return {
        "schema": "qwen_controller_sft_example.v1",
        "id": row_id,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": json.dumps({"task": "decide", "route_id": route_id}, sort_keys=True)},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "accepted": accepted,
                        "action": action,
                        "route_id": route_id,
                        "stem_type": "vocal",
                    },
                    sort_keys=True,
                ),
            },
        ],
        "metadata": {
            "split": "eval",
            "provenance": {
                "source_row_id": row_id,
                "track_id": f"track-{row_id}",
                "stem_type": "vocal",
                "route_id": route_id,
                "manifest_path": f"/app/lab/qwen/{row_id}/manifest.json",
            },
        },
        "delivery_allowed": False,
        "release_ready": False,
        "training_target": "qwen_controller_sft",
    }


def write_sft_dataset(tmp_path: Path, *, eval_rows: list[dict] | None = None) -> Path:
    train_jsonl = write_jsonl(tmp_path / "dataset" / "train.jsonl", [sft_row("train-a", accepted=True)])
    eval_jsonl = write_jsonl(
        tmp_path / "dataset" / "eval.jsonl",
        eval_rows if eval_rows is not None else [sft_row("eval-a", accepted=True), sft_row("eval-b", accepted=False)],
    )
    return write_json(
        tmp_path / "dataset" / "manifest.json",
        {
            "schema": "qwen_controller_sft_dataset_manifest.v1",
            "manifest_type": "qwen_controller_sft_dataset",
            "dataset_status": "ready",
            "target_model_id": "Qwen/Qwen3-4B-Thinking-2507",
            "fingerprint": "dataset-fingerprint",
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "delivery_allowed": False,
            "release_ready": False,
            "outputs": {
                "train_jsonl": str(train_jsonl),
                "eval_jsonl": str(eval_jsonl),
                "manifest_json": str(tmp_path / "dataset" / "manifest.json"),
                "progress_jsonl": str(tmp_path / "dataset" / "progress.jsonl"),
            },
        },
    )


def write_training_manifest(tmp_path: Path, *, status: str = "trained", adapter_exists: bool = True) -> Path:
    adapter_dir = tmp_path / "train" / "adapter"
    if adapter_exists:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter.txt").write_text("saved\n", encoding="utf-8")
    return write_json(
        tmp_path / "train" / "manifest.json",
        {
            "schema": "qwen_controller_qlora_training_manifest.v1",
            "manifest_type": "qwen_controller_qlora_training",
            "training_status": status,
            "failed_checks": [] if status == "trained" else ["preflight_not_passed"],
            "target_model_id": "Qwen/Qwen3-4B-Thinking-2507",
            "training_started": status == "trained",
            "fine_tune_started": status == "trained",
            "models_changed": status == "trained",
            "model_registry_edited": False,
            "delivery_allowed": False,
            "release_ready": False,
            "outputs": {
                "manifest_json": str(tmp_path / "train" / "manifest.json"),
                "adapter_dir": str(adapter_dir),
                "metrics_json": str(tmp_path / "train" / "metrics.json"),
                "progress_jsonl": str(tmp_path / "train" / "progress.jsonl"),
            },
        },
    )


def fake_predictions(*, eval_rows: list[dict], model_role: str, **kwargs) -> list[dict]:
    predictions: list[dict] = []
    for row in eval_rows:
        expected = qwen_eval.expected_payload(row)
        if model_role == "adapter":
            predicted = dict(expected)
        else:
            predicted = {
                "accepted": not bool(expected["accepted"]),
                "action": "avoid_route_and_generate_alternative",
                "route_id": "baseline-wrong",
                "stem_type": "vocal",
            }
        predictions.append(
            {
                "id": row["id"],
                "expected": expected,
                "predicted": predicted,
                "raw_text": json.dumps(predicted, sort_keys=True),
                "parse_error": "",
            }
        )
    return predictions


class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakePromptTokens:
    shape = (1, 3)

    def to(self, device):
        return self


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token = None

    def apply_chat_template(self, prompts, add_generation_prompt: bool, return_tensors: str):
        return FakePromptTokens()

    def decode(self, tokens, skip_special_tokens: bool) -> str:
        return json.dumps(
            {
                "accepted": True,
                "action": "reuse_route_candidate",
                "route_id": "route-a",
                "stem_type": "vocal",
            },
            sort_keys=True,
        )


class FakeAutoTokenizer:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append({"model_id": model_id, "kwargs": kwargs})
        return FakeTokenizer()


class FakeModel:
    device = "cuda"

    def eval(self):
        return None

    def generate(self, prompt_tokens, max_new_tokens: int, do_sample: bool):
        return [[1, 2, 3, 4]]


class FakeAutoModelForCausalLM:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append({"model_id": model_id, "kwargs": kwargs})
        return FakeModel()


class FakeEvalRuntime:
    BitsAndBytesConfig = FakeConfig
    AutoTokenizer = FakeAutoTokenizer
    AutoModelForCausalLM = FakeAutoModelForCausalLM
    PeftModel = object
    torch = type("FakeTorch", (), {"bfloat16": "bfloat16", "float16": "float16"})()


def test_generate_predictions_loads_base_model_from_local_cache_only(tmp_path: Path) -> None:
    FakeAutoTokenizer.calls = []
    FakeAutoModelForCausalLM.calls = []
    eval_rows = [sft_row("eval-a", accepted=True)]

    predictions = qwen_eval.generate_predictions(
        runtime=FakeEvalRuntime,
        eval_rows=eval_rows,
        model_role="baseline",
        adapter_dir=None,
        max_new_tokens=8,
        compute_dtype="bfloat16",
    )

    assert predictions[0]["id"] == "eval-a"
    assert FakeAutoTokenizer.calls[0]["kwargs"]["local_files_only"] is True
    assert FakeAutoModelForCausalLM.calls[0]["kwargs"]["local_files_only"] is True


def test_eval_blocks_when_training_manifest_not_trained(tmp_path: Path) -> None:
    dataset_manifest = write_sft_dataset(tmp_path)
    training_manifest = write_training_manifest(tmp_path, status="blocked", adapter_exists=False)

    manifest = run_qwen_controller_qlora_eval(
        sft_dataset_manifest_json=dataset_manifest,
        training_manifest_json=training_manifest,
        output_dir=tmp_path / "eval",
        min_eval_examples=1,
    )

    assert manifest["eval_status"] == "blocked"
    assert "training_not_trained" in manifest["failed_checks"]
    assert manifest["baseline_comparison"]["adapter_accuracy"] == 0.0
    assert manifest["model_registry_edited"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert Path(manifest["outputs"]["manifest_json"]).name == "manifest.json"
    assert Path(manifest["outputs"]["rows_jsonl"]).name == "rows.jsonl"
    progress_rows = read_jsonl(Path(manifest["outputs"]["progress_jsonl"]))
    assert [row["event"] for row in progress_rows] == ["start", "dataset_checked", "training_manifest_checked", "finish"]


def test_eval_passes_adapter_when_it_beats_baseline(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(qwen_eval, "load_eval_runtime", lambda: object())
    monkeypatch.setattr(qwen_eval, "generate_predictions", fake_predictions)
    dataset_manifest = write_sft_dataset(tmp_path)
    training_manifest = write_training_manifest(tmp_path)

    manifest = run_qwen_controller_qlora_eval(
        sft_dataset_manifest_json=dataset_manifest,
        training_manifest_json=training_manifest,
        output_dir=tmp_path / "eval",
        min_eval_examples=2,
        min_adapter_accuracy=0.9,
    )

    assert manifest["eval_status"] == "passed"
    assert manifest["failed_checks"] == []
    assert manifest["baseline_comparison"]["baseline_accuracy"] == 0.0
    assert manifest["baseline_comparison"]["adapter_accuracy"] == 1.0
    assert manifest["baseline_comparison"]["adapter_accuracy_improvement"] == 1.0
    assert manifest["promotion"]["can_promote_adapter"] is False
    assert manifest["promotion"]["next_required_step"] == "manual_registry_review"
    assert manifest["model_registry_edited"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    rows = read_jsonl(Path(manifest["outputs"]["rows_jsonl"]))
    assert {row["model_role"] for row in rows} == {"adapter", "baseline"}
    assert all(Path(path).name in {"manifest.json", "rows.jsonl", "progress.jsonl"} for path in manifest["outputs"].values())


def test_eval_blocks_when_adapter_does_not_beat_baseline(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(qwen_eval, "load_eval_runtime", lambda: object())
    monkeypatch.setattr(qwen_eval, "generate_predictions", lambda **kwargs: fake_predictions(**{**kwargs, "model_role": "adapter"}))
    dataset_manifest = write_sft_dataset(tmp_path)
    training_manifest = write_training_manifest(tmp_path)

    manifest = run_qwen_controller_qlora_eval(
        sft_dataset_manifest_json=dataset_manifest,
        training_manifest_json=training_manifest,
        output_dir=tmp_path / "eval",
        min_eval_examples=2,
        min_adapter_accuracy=0.9,
    )

    assert manifest["eval_status"] == "blocked"
    assert "adapter_not_better_than_baseline" in manifest["failed_checks"]
    assert manifest["baseline_comparison"]["baseline_accuracy"] == 1.0
    assert manifest["baseline_comparison"]["adapter_accuracy"] == 1.0


def test_cli_checks_docker_runtime_before_parsing_args(monkeypatch) -> None:
    calls = []

    def reject_non_docker_runtime(**kwargs):
        calls.append(("require_docker_runtime", kwargs))
        raise RuntimeError("Docker-only test sentinel")

    def fail_if_args_are_parsed(self, argv=None):
        raise AssertionError("parse_args must not run before Docker runtime validation")

    monkeypatch.setattr(qwen_eval, "require_docker_runtime", reject_non_docker_runtime)
    monkeypatch.setattr(qwen_eval.argparse.ArgumentParser, "parse_args", fail_if_args_are_parsed)

    try:
        main(["--sft-dataset-manifest-json", "/tmp/dataset.json", "--training-manifest-json", "/tmp/train.json", "--output-dir", "/tmp/out"])
    except RuntimeError as exc:
        assert str(exc) == "Docker-only test sentinel"
    else:
        raise AssertionError("main must fail through Docker runtime guard")

    assert calls == [
        (
            "require_docker_runtime",
            {
                "tool_name": "qwen_controller_qlora_eval.py",
                "entrypoint_path": "/app/tools/qwen_controller_qlora_eval.py",
            },
        )
    ]
