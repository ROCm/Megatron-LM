"""Print the environment table for `kimi_k3/PINS.md` §5.

    python -m kimi_k3.tools.capture_env

Run it at every pin bump and paste the output into PINS.md, so the recorded
environment is observed rather than remembered (rule R10.2).
"""

import importlib


def _version(mod_name: str) -> str:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        return f"not importable ({type(exc).__name__})"
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "installed (no __version__)"


def main() -> None:
    import torch

    rows = [
        ("torch", torch.__version__),
        ("hip", getattr(torch.version, "hip", None) or "n/a"),
        ("triton", _version("triton")),
        ("transformer_engine", _version("transformer_engine")),
        ("fla", _version("fla")),
        ("aiter", _version("aiter")),
    ]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        rows.append(
            ("GPUs", f"{torch.cuda.device_count()} x {torch.cuda.get_device_name(0)} "
                     f"({getattr(props, 'gcnArchName', '?')})")
        )
    else:
        rows.append(("GPUs", "none visible"))

    print("| Component | Version |")
    print("|---|---|")
    for name, value in rows:
        print(f"| {name} | {value} |")


if __name__ == "__main__":
    main()
