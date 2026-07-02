from __future__ import annotations

import json
from pathlib import Path

import tools.build_generic_finetuning_corpus_from_lab_artifacts as corpus_builder
from tools.build_generic_finetuning_corpus_from_lab_artifacts import build_corpus, main


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def accepted_vocal_manifest(tmp_path: Path) -> Path:
    audio = tmp_path / "audio" / "vocal.flac"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"vocal")
    return write_json(
        tmp_path / "lab" / "nothing" / "no_reference_vocal_listen_preview_manifest.json",
        {
            "track_id": "nothing-can-wait",
            "stem_type": "vocal",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "selected_vocal_path": str(audio),
            "base_candidate_id": "mdx23c_vocals",
        },
    )


def accepted_machine_gate_vocal_manifest(tmp_path: Path) -> Path:
    audio = tmp_path / "audio" / "machine_gate_vocal.flac"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"machine-gate-vocal")
    return write_json(
        tmp_path / "lab" / "accepted_gate" / "machine_stem_acceptance_gate.json",
        {
            "schema": "machine_stem_acceptance_gate.v1",
            "track_id": "machine-gated-track",
            "stem_type": "vocal",
            "gate_status": "accepted",
            "final_result_allowed": True,
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "selected_vocal_path": str(audio),
            "required_machine_gates": [
                "aggregate_pack_validation_passed",
                "full_required_stem_coverage",
                "required_stem_machine_quality_evidence",
                "source_vocal_minus_master_pair_compatible",
                "minus_quality_audit_passed_without_warnings",
            ],
            "failed_checks": [],
            "blocker_count": 0,
        },
    )


def blocked_minus_manifest(tmp_path: Path) -> Path:
    audio = tmp_path / "audio" / "bad_minus.flac"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"minus")
    return write_json(
        tmp_path / "lab" / "blocked" / "no_reference_minus_listen_preview_manifest.json",
        {
            "track_id": "bad-track",
            "stem_type": "minus",
            "delivery_status": "blocked",
            "delivery_allowed": False,
            "presentation_status": "blocked_from_listen_pack",
            "release_status": "not_release_ready",
            "selected_minus_path": str(audio),
            "dominant_failure_reason": "severe_vocal_bleed_in_minus",
            "base_candidate_id": "melband_roformer_vocals",
        },
    )


def accepted_claim_with_nested_failure_labels_manifest(tmp_path: Path) -> Path:
    audio = tmp_path / "audio" / "claimed_good_minus.flac"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"minus")
    return write_json(
        tmp_path / "lab" / "claimed_good" / "no_reference_minus_listen_preview_manifest.json",
        {
            "track_id": "claimed-good-track",
            "stem_type": "minus",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "selected_minus_path": str(audio),
            "base_candidate_id": "leaky_minus_model",
            "quality_audit": {
                "issue_labels": [
                    "foreign_source_recovered:minus:vocals,mix_conservation_failed",
                    "candidate_manifest_created",
                ],
            },
        },
    )


def accepted_claim_with_structured_blockers_manifest(tmp_path: Path) -> Path:
    audio = tmp_path / "audio" / "structured_blocked_minus.flac"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"minus")
    return write_json(
        tmp_path / "lab" / "structured_blocked" / "no_reference_minus_listen_preview_manifest.json",
        {
            "track_id": "structured-blocked-track",
            "stem_type": "minus",
            "delivery_status": "accepted",
            "delivery_allowed": True,
            "presentation_status": "accepted_for_listen_pack",
            "release_status": "listen_preview_only",
            "selected_minus_path": str(audio),
            "base_candidate_id": "structured_blocked_minus_model",
            "quality_audit": {
                "blockers": [
                    {"code": "foreign_source_recovered:minus:vocals", "severity": "hard"},
                    {"label": "mix_conservation_failed"},
                ],
                "failed_checks": [
                    {"name": "voice_presence_guard", "failed": True},
                ],
                "is_blocked": True,
            },
        },
    )


def verified_ground_truth_manifest(tmp_path: Path, *, source_mix_1_to_1: bool = True) -> Path:
    root = tmp_path / "lab" / "golden_audio" / "verified_gt"
    (root / "stems").mkdir(parents=True, exist_ok=True)
    (root / "source_mix.flac").write_bytes(b"mix")
    (root / "stems" / "vocals.flac").write_bytes(b"vocals")
    (root / "stems" / "instrumental.flac").write_bytes(b"minus")
    return write_json(
        root / "ground_truth_manifest.json",
        {
            "track_name": "verified-song",
            "ground_truth_contract": {"verified": True, "mode": "verified_isolated_stems"},
            "mix": {"path": "source_mix.flac", "role": "authoritative_full_mix_input"},
            "stems": {
                "vocals": {"path": "stems/vocals.flac", "sha256": "vocalsha"},
                "instrumental": {"path": "stems/instrumental.flac", "sha256": "minussha"},
            },
            "full_mix_reconstruction_contract": {
                "status": "accepted" if source_mix_1_to_1 else "rejected",
                "can_claim_source_mix_1_to_1": source_mix_1_to_1,
            },
            "report_alignment_contract": {"rights_status": "user_owned"},
        },
    )


def test_human_positive_listen_preview_without_machine_gate_is_not_audio_trainable(tmp_path: Path) -> None:
    vocal_manifest = accepted_vocal_manifest(tmp_path)
    blocked_minus_manifest(tmp_path)
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "track_id": "nothing-can-wait",
                "stem_type": "vocal",
                "manifest_path": str(vocal_manifest),
                "rating": "ideal",
                "human_verified": True,
                "permission_status": "owned",
                "split": "train",
                "notes": "User confirmed this vocal by listening.",
            }
        ],
    )

    manifest = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        human_feedback_jsonls=[feedback],
        permission_default="requires_review",
    )

    assert manifest["summary"]["accepted_like"] == 1
    assert manifest["summary"]["blocked_like"] == 1
    assert manifest["summary"]["audio_training_examples"] == 0
    assert manifest["summary"]["hard_eval_cases"] == 1
    assert manifest["summary"]["excluded_examples"] == 1
    assert manifest["training_started"] is False
    excluded = read_jsonl(tmp_path / "out" / "excluded_examples.jsonl")
    assert excluded[0]["exclude_reason"] == "accepted_but_missing_accepted_machine_gate_provenance"
    hard_eval = read_jsonl(tmp_path / "out" / "hard_eval_cases.jsonl")
    assert hard_eval[0]["dominant_failure_reason"] == "severe_vocal_bleed_in_minus"


def test_builds_curated_corpus_with_machine_gate_human_verified_trainable_audio(tmp_path: Path) -> None:
    vocal_manifest = accepted_machine_gate_vocal_manifest(tmp_path)
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "track_id": "machine-gated-track",
                "stem_type": "vocal",
                "manifest_path": str(vocal_manifest),
                "rating": "ideal",
                "human_verified": True,
                "permission_status": "owned",
                "split": "train",
                "notes": "User confirmed this post-gate vocal by listening.",
            }
        ],
    )

    manifest = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        human_feedback_jsonls=[feedback],
        permission_default="requires_review",
    )

    assert manifest["summary"]["accepted_like"] == 1
    assert manifest["summary"]["accepted_machine_gate_like"] == 1
    assert manifest["summary"]["audio_training_examples"] == 1
    assert manifest["summary"]["excluded_examples"] == 0
    audio_rows = read_jsonl(tmp_path / "out" / "audio_training_examples.jsonl")
    assert audio_rows[0]["track_id"] == "machine-gated-track"
    assert audio_rows[0]["stem_type"] == "vocal"
    assert audio_rows[0]["trainable"] is True
    assert audio_rows[0]["permission_status"] == "owned"
    assert audio_rows[0]["source_gate"]["accepted_machine_gate"] is True


def test_builds_audio_examples_from_verified_ground_truth_bundle(tmp_path: Path) -> None:
    verified_ground_truth_manifest(tmp_path)

    manifest = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        human_feedback_jsonls=[],
        permission_default="requires_review",
    )

    assert manifest["summary"]["ground_truth_manifests"] == 1
    assert manifest["summary"]["ground_truth_audio_training_examples"] == 2
    assert manifest["summary"]["audio_training_examples"] == 2
    rows = read_jsonl(tmp_path / "out" / "audio_training_examples.jsonl")
    assert {row["stem_type"] for row in rows} == {"minus", "vocal"}
    assert {row["permission_status"] for row in rows} == {"project_owned"}
    assert all(row["reference_verified"] is True for row in rows)
    assert all(row["safety"]["source_mix_1_to_1_claim_allowed"] is True for row in rows)
    assert all(row["source_gate"]["trainable_ground_truth"] is True for row in rows)


def test_rejected_ground_truth_reconstruction_is_not_audio_trainable(tmp_path: Path) -> None:
    verified_ground_truth_manifest(tmp_path, source_mix_1_to_1=False)

    manifest = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        human_feedback_jsonls=[],
        permission_default="requires_review",
    )

    assert manifest["summary"]["ground_truth_manifests"] == 1
    assert manifest["summary"]["ground_truth_audio_training_examples"] == 0
    assert manifest["summary"]["audio_training_examples"] == 0
    excluded = read_jsonl(tmp_path / "out" / "excluded_examples.jsonl")
    assert excluded[0]["exclude_reason"] == "ground_truth_reconstruction_not_trainable"
    assert excluded[0]["source_gate"]["trainable_ground_truth"] is False


def test_hard_eval_rows_are_not_controller_training_examples_by_default(tmp_path: Path) -> None:
    blocked_minus_manifest(tmp_path)

    manifest = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        human_feedback_jsonls=[],
        permission_default="requires_review",
    )

    assert manifest["summary"]["hard_eval_cases"] == 1
    assert manifest["summary"]["controller_decision_examples"] == 0
    hard_eval = read_jsonl(tmp_path / "out" / "hard_eval_cases.jsonl")
    controller_rows = read_jsonl(tmp_path / "out" / "controller_decision_examples.jsonl")
    assert hard_eval[0]["split"] == "hard_eval"
    assert controller_rows == []


def test_negative_train_feedback_can_still_teach_controller(tmp_path: Path) -> None:
    blocked_manifest = blocked_minus_manifest(tmp_path)
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "track_id": "bad-track",
                "stem_type": "minus",
                "manifest_path": str(blocked_manifest),
                "rating": "bad",
                "human_verified": True,
                "permission_status": "requires_review",
                "split": "train",
                "notes": "Use this as a negative controller route example, not as audio training.",
            }
        ],
    )

    manifest = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        human_feedback_jsonls=[feedback],
        permission_default="requires_review",
    )

    assert manifest["summary"]["hard_eval_cases"] == 1
    assert manifest["summary"]["controller_decision_examples"] == 1
    hard_eval = read_jsonl(tmp_path / "out" / "hard_eval_cases.jsonl")
    controller_rows = read_jsonl(tmp_path / "out" / "controller_decision_examples.jsonl")
    assert hard_eval[0]["split"] == "train"
    assert controller_rows[0]["result"]["accepted"] is False
    assert controller_rows[0]["target"]["action"] == "avoid_route_and_generate_alternative"


def test_deep_failure_labels_override_accepted_claim_for_training_surfaces(tmp_path: Path) -> None:
    leaky_manifest = accepted_claim_with_nested_failure_labels_manifest(tmp_path)
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "track_id": "claimed-good-track",
                "stem_type": "minus",
                "manifest_path": str(leaky_manifest),
                "rating": "ideal",
                "human_verified": True,
                "permission_status": "owned",
                "split": "train",
                "notes": "Positive feedback must not override machine blocking labels.",
            }
        ],
    )

    manifest = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        human_feedback_jsonls=[feedback],
        permission_default="requires_review",
    )

    assert manifest["summary"]["accepted_like"] == 1
    assert manifest["summary"]["blocked_like"] == 1
    assert manifest["summary"]["hard_eval_cases"] == 1
    assert manifest["summary"]["audio_training_examples"] == 0
    hard_eval = read_jsonl(tmp_path / "out" / "hard_eval_cases.jsonl")
    controller_rows = read_jsonl(tmp_path / "out" / "controller_decision_examples.jsonl")
    assert hard_eval[0]["dominant_failure_reason"] == "foreign_source_recovered:minus:vocals"
    assert "mix_conservation_failed" in hard_eval[0]["failure_labels"]
    assert controller_rows[0]["result"]["accepted"] is False
    assert controller_rows[0]["target"]["action"] == "avoid_route_and_generate_alternative"


def test_structured_blockers_override_accepted_claim_for_training_surfaces(tmp_path: Path) -> None:
    structured_manifest = accepted_claim_with_structured_blockers_manifest(tmp_path)
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "track_id": "structured-blocked-track",
                "stem_type": "minus",
                "manifest_path": str(structured_manifest),
                "rating": "ideal",
                "human_verified": True,
                "permission_status": "owned",
                "split": "train",
                "notes": "Structured machine blockers must win over accepted-looking status fields.",
            }
        ],
    )

    manifest = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        human_feedback_jsonls=[feedback],
        permission_default="requires_review",
    )

    assert manifest["summary"]["accepted_like"] == 1
    assert manifest["summary"]["blocked_like"] == 1
    assert manifest["summary"]["hard_eval_cases"] == 1
    assert manifest["summary"]["audio_training_examples"] == 0
    hard_eval = read_jsonl(tmp_path / "out" / "hard_eval_cases.jsonl")
    controller_rows = read_jsonl(tmp_path / "out" / "controller_decision_examples.jsonl")
    assert hard_eval[0]["dominant_failure_reason"] == "foreign_source_recovered:minus:vocals"
    assert "mix_conservation_failed" in hard_eval[0]["failure_labels"]
    assert "failed" in hard_eval[0]["failure_labels"]
    assert controller_rows[0]["result"]["accepted"] is False
    assert controller_rows[0]["target"]["action"] == "avoid_route_and_generate_alternative"


def test_negative_feedback_only_hard_eval_has_synthetic_failure_reason(tmp_path: Path) -> None:
    accepted_manifest = accepted_vocal_manifest(tmp_path)
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "track_id": "nothing-can-wait",
                "stem_type": "vocal",
                "manifest_path": str(accepted_manifest),
                "rating": "bad",
                "human_verified": True,
                "permission_status": "requires_review",
                "split": "train",
                "notes": "Human rejection without machine labels must still produce a labeled hard eval case.",
            }
        ],
    )

    manifest = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        human_feedback_jsonls=[feedback],
        permission_default="requires_review",
    )

    assert manifest["summary"]["hard_eval_cases"] == 1
    assert manifest["summary"]["audio_training_examples"] == 0
    hard_eval = read_jsonl(tmp_path / "out" / "hard_eval_cases.jsonl")
    controller_rows = read_jsonl(tmp_path / "out" / "controller_decision_examples.jsonl")
    assert hard_eval[0]["dominant_failure_reason"] == "human_negative_feedback:bad"
    assert hard_eval[0]["failure_labels"] == ["human_negative_feedback:bad"]
    assert controller_rows[0]["result"]["accepted"] is False


def test_accepted_without_human_or_reference_is_not_audio_trainable(tmp_path: Path) -> None:
    accepted_vocal_manifest(tmp_path)

    manifest = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        human_feedback_jsonls=[],
        permission_default="requires_review",
    )

    assert manifest["summary"]["accepted_like"] == 1
    assert manifest["summary"]["audio_training_examples"] == 0
    assert manifest["summary"]["excluded_examples"] == 1
    excluded = read_jsonl(tmp_path / "out" / "excluded_examples.jsonl")
    assert excluded[0]["exclude_reason"] == "accepted_but_not_human_or_reference_verified"


def test_auto_discovers_human_feedback_jsonl_from_artifact_root(tmp_path: Path) -> None:
    vocal_manifest = accepted_machine_gate_vocal_manifest(tmp_path)
    manifest_payload = read_json(vocal_manifest)
    write_jsonl(
        tmp_path / "lab" / "golden_audio" / "adaptive_vocal_feedback.jsonl",
        [
            {
                "target_stem": "vocal",
                "candidate_audio_path": manifest_payload["selected_vocal_path"],
                "rating": "better",
                "human_verified": True,
                "permission_status": "owned",
                "feedback_split": "train",
            }
        ],
    )

    manifest = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        human_feedback_jsonls=[],
        permission_default="requires_review",
    )

    assert manifest["summary"]["human_feedback_files"] == 1
    assert manifest["summary"]["auto_discovered_human_feedback_files"] == 1
    assert manifest["summary"]["audio_training_examples"] == 1
    audio_rows = read_jsonl(tmp_path / "out" / "audio_training_examples.jsonl")
    assert audio_rows[0]["rating"] == "better"
    assert audio_rows[0]["permission_status"] == "owned"


def test_unverified_inventory_does_not_hash_audio(tmp_path: Path, monkeypatch) -> None:
    accepted_vocal_manifest(tmp_path)

    def reject_hash(path: Path) -> str:
        raise AssertionError(f"unexpected audio hash for inventory-only row: {path}")

    monkeypatch.setattr(corpus_builder, "sha256_file", reject_hash)

    manifest = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out",
        human_feedback_jsonls=[],
        permission_default="requires_review",
    )

    assert manifest["summary"]["accepted_like"] == 1
    assert manifest["summary"]["audio_training_examples"] == 0


def test_fingerprint_is_stable_for_same_inputs(tmp_path: Path) -> None:
    vocal_manifest = accepted_vocal_manifest(tmp_path)
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "track_id": "nothing-can-wait",
                "stem_type": "vocal",
                "manifest_path": str(vocal_manifest),
                "rating": "ideal",
                "human_verified": True,
                "permission_status": "owned",
            }
        ],
    )

    first = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out1",
        human_feedback_jsonls=[feedback],
        permission_default="requires_review",
    )
    second = build_corpus(
        artifact_root=tmp_path / "lab",
        output_dir=tmp_path / "out2",
        human_feedback_jsonls=[feedback],
        permission_default="requires_review",
    )

    assert first["fingerprint"] == second["fingerprint"]


def test_cli_writes_manifest_and_jsonl_outputs(tmp_path: Path) -> None:
    vocal_manifest = accepted_machine_gate_vocal_manifest(tmp_path)
    feedback = write_jsonl(
        tmp_path / "feedback.jsonl",
        [
            {
                "track_id": "machine-gated-track",
                "stem_type": "vocal",
                "manifest_path": str(vocal_manifest),
                "rating": "ideal",
                "human_verified": True,
                "permission_status": "owned",
            }
        ],
    )

    assert main(
        [
            "--artifact-root",
            str(tmp_path / "lab"),
            "--output-dir",
            str(tmp_path / "out"),
            "--human-feedback-jsonl",
            str(feedback),
        ]
    ) == 0

    manifest = read_json(tmp_path / "out" / "generic_finetuning_corpus_manifest.json")
    assert manifest["summary"]["audio_training_examples"] == 1
    assert (tmp_path / "out" / "controller_decision_examples.jsonl").exists()
    assert (tmp_path / "out" / "inventory.jsonl").exists()
