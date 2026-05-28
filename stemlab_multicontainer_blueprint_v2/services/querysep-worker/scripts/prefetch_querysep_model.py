from __future__ import annotations

from app.banquet_runtime import banquet_ready_state, prefetch_runtime as prefetch_banquet_runtime
from app.runtime import prefetch_runtime as prefetch_clap_runtime


def main() -> None:
    clap_marker = prefetch_clap_runtime()
    print(f"querysep CLAP prefetch completed: {clap_marker['model_id']} on {clap_marker['device']}")

    banquet_state = banquet_ready_state()
    if banquet_state["enabled"]:
        banquet_marker = prefetch_banquet_runtime()
        print(
            "querysep Banquet prefetch completed: "
            f"{banquet_marker['checkpoint_path']} on {banquet_marker['device']}"
        )


if __name__ == "__main__":
    main()
