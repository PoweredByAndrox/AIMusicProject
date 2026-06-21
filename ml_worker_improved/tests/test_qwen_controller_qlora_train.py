from __future__ import annotations

import json
from pathlib import Path

import tools.qwen_controller_qlora_train as qwen_train
from tools.qwen_controller_qlora_train import main, run_qwen_controller_qlora_training


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


def sft_row(row_id: str, *, split: str, manifest_path: str, route_id: str = "route-a") -> dict:
    return {
        "schema": "qwen_controller_sft_example.v1",
        "id": row_id,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "{}"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "accepted": True,
                        "action": "reuse_route_candidate",
                        "route_id": route_id,
                        "stem_type": "vocal",
                    },
                    sort_keys=True,
                ),
            },
        ],
        "metadata": {
            "split": split,
            "provenance": {
                "source_row_id": row_id,
                "track_id": f"track-{row_id}",
                "stem_type": "vocal",
                "route_id": route_id,
                "manifest_path": manifest_path,
            },
        },
        "delivery_allowed": False,
        "release_ready": False,
        "training_target": "qwen_controller_sft",
    }


def write_sft_dataset(tmp_path: Path, *, status: str = "ready") -> Path:
    train_jsonl = write_jsonl(
        tmp_path / "dataset" / "train.jsonl",
        [sft_row("train-a", split="train", manifest_path="/app/lab/qwen/t001/manifest.json")],
    )
    eval_jsonl = write_jsonl(
        tmp_path / "dataset" / "eval.jsonl",
        [sft_row("eval-a", split="eval", manifest_path="/app/lab/qwen/t002/manifest.json")],
    )
    excluded_jsonl = write_jsonl(tmp_path / "dataset" / "excluded.jsonl", [])
    return write_json(
        tmp_path / "dataset" / "manifest.json",
        {
            "schema": "qwen_controller_sft_dataset_manifest.v1",
            "manifest_type": "qwen_controller_sft_dataset",
            "dataset_status": status,
            "target_model_id": "Qwen/Qwen3-4B-Thinking-2507",
            "fingerprint": "dataset-fingerprint",
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "delivery_allowed": False,
            "release_ready": False,
            "summary": {
                "train_examples": 1,
                "eval_examples": 1,
                "excluded_examples": 0,
            },
            "outputs": {
                "train_jsonl": str(train_jsonl),
                "eval_jsonl": str(eval_jsonl),
                "excluded_jsonl": str(excluded_jsonl),
                "manifest_json": str(tmp_path / "dataset" / "manifest.json"),
                "progress_jsonl": str(tmp_path / "dataset" / "progress.jsonl"),
            },
        },
    )


def write_preflight(tmp_path: Path, *, status: str, failed_checks: list[str] | None = None) -> Path:
    return write_json(
        tmp_path / "preflight" / "manifest.json",
        {
            "schema": "qwen_controller_qlora_preflight_manifest.v1",
            "manifest_type": "qwen_controller_qlora_preflight",
            "preflight_status": status,
            "failed_checks": failed_checks or [],
            "target_model_id": "Qwen/Qwen3-4B-Thinking-2507",
            "training_started": False,
            "fine_tune_started": False,
            "models_changed": False,
            "model_registry_edited": False,
            "delivery_allowed": False,
            "release_ready": False,
            "dependency_report": {"missing_required_modules": []},
            "cuda": {
                "cuda_available": True,
                "requested_device": "cuda:0",
                "device_count": 1,
                "device_name": "Unit Test GPU",
                "free_memory_mb": 24000,
                "total_memory_mb": 32000,
                "memory_guard_passed": True,
            },
            "dataset": {
                "train_examples": 1,
                "eval_examples": 1,
                "train_eval_overlap_count": 0,
            },
        },
    )


class FakeDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    @classmethod
    def from_list(cls, rows: list[dict]) -> "FakeDataset":
        return cls(rows)


class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token = None

    def save_pretrained(self, output_dir: str) -> None:
        Path(output_dir, "tokenizer.txt").write_text("saved\n", encoding="utf-8")


class FakeModel:
    config = type("Config", (), {"use_cache": True})()


class FakeAutoTokenizer:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append({"model_id": model_id, "kwargs": kwargs})
        return FakeTokenizer()


class FakeAutoModelForCausalLM:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append({"model_id": model_id, "kwargs": kwargs})
        return FakeModel()


class FakeTrainer:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.state = type("State", (), {"log_history": [{"loss": 0.01}]})()
        FakeTrainer.calls.append(kwargs)

    def train(self):
        return type("TrainOutput", (), {"metrics": {"train_loss": 0.01, "epoch": 1.0}})()

    def save_model(self, output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "adapter.txt").write_text("saved\n", encoding="utf-8")


class FakeRuntime:
    Dataset = FakeDataset
    BitsAndBytesConfig = FakeConfig
    LoraConfig = FakeConfig
    SFTConfig = FakeConfig
    SFTTrainer = FakeTrainer
    AutoTokenizer = FakeAutoTokenizer
    AutoModelForCausalLM = FakeAutoModelForCausalLM
    torch = type("FakeTorch", (), {"bfloat16": "bfloat16", "float16": "float16"})()


def test_training_blocks_when_preflight_is_not_passed(tmp_path: Path) -> None:
    dataset_manifest = write_sft_dataset(tmp_path)
    preflight_manifest = write_preflight(tmp_path, status="blocked", failed_checks=["missing_required_python_modules"])

    manifest = run_qwen_controller_qlora_training(
        sft_dataset_manifest_json=dataset_manifest,
        preflight_manifest_json=preflight_manifest,
        output_dir=tmp_path / "out",
    )

    assert manifest["schema"] == "qwen_controller_qlora_training_manifest.v1"
    assert manifest["training_status"] == "blocked"
    assert "preflight_not_passed" in manifest["failed_checks"]
    assert manifest["training_started"] is False
    assert manifest["fine_tune_started"] is False
    assert manifest["models_changed"] is False
    assert manifest["model_registry_edited"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["safety"]["blocks_failed_preflight"] is True
    assert Path(manifest["outputs"]["manifest_json"]).name == "manifest.json"
    assert Path(manifest["outputs"]["adapter_dir"]).name == "adapter"
    assert not Path(manifest["outputs"]["adapter_dir"]).exists()
    progress_rows = read_jsonl(Path(manifest["outputs"]["progress_jsonl"]))
    assert [row["event"] for row in progress_rows] == ["start", "dataset_checked", "preflight_checked", "finish"]
    assert progress_rows[0]["gpu_training_required"] is True


def test_training_runs_qlora_runtime_and_writes_short_artifacts(monkeypatch, tmp_path: Path) -> None:
    FakeTrainer.calls = []
    FakeAutoTokenizer.calls = []
    FakeAutoModelForCausalLM.calls = []
    monkeypatch.setattr(qwen_train, "load_qlora_runtime", lambda: FakeRuntime)
    dataset_manifest = write_sft_dataset(tmp_path)
    preflight_manifest = write_preflight(tmp_path, status="passed")

    manifest = run_qwen_controller_qlora_training(
        sft_dataset_manifest_json=dataset_manifest,
        preflight_manifest_json=preflight_manifest,
        output_dir=tmp_path / "out",
        max_steps=1,
    )

    assert manifest["training_status"] == "trained"
    assert manifest["training_started"] is True
    assert manifest["fine_tune_started"] is True
    assert manifest["models_changed"] is True
    assert manifest["model_registry_edited"] is False
    assert manifest["delivery_allowed"] is False
    assert manifest["release_ready"] is False
    assert manifest["promotion"]["can_promote_adapter"] is False
    assert manifest["promotion"]["next_required_step"] == "external_eval_gate"
    assert manifest["summary"]["train_examples"] == 1
    assert manifest["summary"]["eval_examples"] == 1
    assert manifest["training_config"]["quantization"]["load_in_4bit"] is True
    assert manifest["training_config"]["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert Path(manifest["outputs"]["adapter_dir"]).name == "adapter"
    assert (tmp_path / "out" / "adapter" / "adapter.txt").exists()
    assert (tmp_path / "out" / "metrics.json").exists()
    assert FakeAutoTokenizer.calls[0]["kwargs"]["local_files_only"] is True
    assert FakeAutoModelForCausalLM.calls[0]["model_id"] == "Qwen/Qwen3-4B-Thinking-2507"
    assert FakeAutoModelForCausalLM.calls[0]["kwargs"]["device_map"] == "auto"
    assert FakeAutoModelForCausalLM.calls[0]["kwargs"]["local_files_only"] is True
    assert FakeTrainer.calls[0]["train_dataset"].rows[0]["id"] == "train-a"
    assert FakeTrainer.calls[0]["eval_dataset"].rows[0]["id"] == "eval-a"
    assert FakeTrainer.calls[0]["args"].kwargs["output_dir"] == str(tmp_path / "out" / "trainer")
    progress_rows = read_jsonl(Path(manifest["outputs"]["progress_jsonl"]))
    assert [row["event"] for row in progress_rows] == [
        "start",
        "dataset_checked",
        "preflight_checked",
        "runtime_loaded",
        "model_loaded",
        "trainer_started",
        "adapter_saved",
        "finish",
    ]


def test_cli_checks_docker_runtime_before_parsing_args(monkeypatch) -> None:
    calls = []

    def reject_non_docker_runtime(**kwargs):
        calls.append(("require_docker_runtime", kwargs))
        raise RuntimeError("Docker-only test sentinel")

    def fail_if_args_are_parsed(self, argv=None):
        raise AssertionError("parse_args must not run before Docker runtime validation")

    monkeypatch.setattr(qwen_train, "require_docker_runtime", reject_non_docker_runtime)
    monkeypatch.setattr(qwen_train.argparse.ArgumentParser, "parse_args", fail_if_args_are_parsed)

    try:
        main(["--sft-dataset-manifest-json", "/tmp/dataset.json", "--preflight-manifest-json", "/tmp/pre.json", "--output-dir", "/tmp/out"])
    except RuntimeError as exc:
        assert str(exc) == "Docker-only test sentinel"
    else:
        raise AssertionError("main must fail through Docker runtime guard")

    assert calls == [
        (
            "require_docker_runtime",
            {
                "tool_name": "qwen_controller_qlora_train.py",
                "entrypoint_path": "/app/tools/qwen_controller_qlora_train.py",
            },
        )
    ]
