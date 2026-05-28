from __future__ import annotations

from pathlib import Path

import bs_roformer


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"unable to patch {label}: expected snippet not found")
    return text.replace(old, new, 1)


def main() -> None:
    package_root = Path(bs_roformer.__file__).resolve().parent

    attend_path = package_root / "attend.py"
    attend_text = attend_path.read_text(encoding="utf-8")
    attend_text = replace_once(
        attend_text,
        "import torch\nfrom torch import nn, einsum\nimport torch.nn.functional as F\n",
        "import torch\nfrom torch import nn, einsum\nimport torch.nn.functional as F\n\n"
        "try:\n"
        "    from torch.nn.attention import SDPBackend, sdpa_kernel\n"
        "except ImportError:  # pragma: no cover\n"
        "    SDPBackend = None\n"
        "    sdpa_kernel = None\n",
        "attend imports",
    )
    attend_text = replace_once(
        attend_text,
        "        with torch.backends.cuda.sdp_kernel(**config._asdict()):\n"
        "            out = F.scaled_dot_product_attention(\n"
        "                q, k, v,\n"
        "                dropout_p = self.dropout if self.training else 0.\n"
        "            )\n",
        "        if sdpa_kernel is not None and SDPBackend is not None:\n"
        "            backends = []\n"
        "            if config.enable_flash:\n"
        "                backends.append(SDPBackend.FLASH_ATTENTION)\n"
        "            if config.enable_math:\n"
        "                backends.append(SDPBackend.MATH)\n"
        "            if config.enable_mem_efficient:\n"
        "                backends.append(SDPBackend.EFFICIENT_ATTENTION)\n"
        "            with sdpa_kernel(backends):\n"
        "                out = F.scaled_dot_product_attention(\n"
        "                    q, k, v,\n"
        "                    dropout_p = self.dropout if self.training else 0.\n"
        "                )\n"
        "        else:  # pragma: no cover\n"
        "            with torch.backends.cuda.sdp_kernel(**config._asdict()):\n"
        "                out = F.scaled_dot_product_attention(\n"
        "                    q, k, v,\n"
        "                    dropout_p = self.dropout if self.training else 0.\n"
        "                )\n",
        "attend sdpa kernel block",
    )
    attend_path.write_text(attend_text, encoding="utf-8")

    utils_path = package_root / "utils.py"
    utils_text = utils_path.read_text(encoding="utf-8")
    utils_text = replace_once(
        utils_text,
        "    with torch.cuda.amp.autocast():\n"
        "        with torch.no_grad():\n",
        "    autocast_ctx = (\n"
        "        torch.amp.autocast(device_type=device.type, enabled=device.type != 'cpu')\n"
        "        if hasattr(torch, 'amp')\n"
        "        else torch.cuda.amp.autocast(enabled=device.type == 'cuda')\n"
        "    )\n"
        "\n"
        "    with autocast_ctx:\n"
        "        with torch.no_grad():\n",
        "utils autocast block",
    )
    utils_path.write_text(utils_text, encoding="utf-8")

    model_path = package_root / "bs_roformer.py"
    model_text = model_path.read_text(encoding="utf-8")
    model_text = replace_once(
        model_text,
        "        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, return_complex = True).shape[1]\n",
        "        init_window = self.stft_window_fn(device = torch.device('cpu'))\n"
        "        freqs = torch.stft(\n"
        "            torch.randn(1, 4096),\n"
        "            **self.stft_kwargs,\n"
        "            window = init_window,\n"
        "            return_complex = True\n"
        "        ).shape[1]\n",
        "bs_roformer init stft window",
    )
    model_path.write_text(model_text, encoding="utf-8")


if __name__ == "__main__":
    main()
