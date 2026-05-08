import importlib
import platform
import sys
from typing import Any, Optional, Tuple


def format_bool(value: bool) -> str:
    return "YES" if value else "NO"


def safe_import(module_name: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        module = importlib.import_module(module_name)
        return True, module, None
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def module_version(module: Any, fallback_attr: str = "__version__") -> str:
    if module is None:
        return "N/A"
    return str(getattr(module, fallback_attr, "unknown"))


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def check_python() -> None:
    print_header("Python")
    print(f"python.version      : {sys.version.split()[0]}")
    print(f"python.full_version : {sys.version.replace(chr(10), ' ')}")
    print(f"platform            : {platform.platform()}")


def check_torch() -> None:
    print_header("PyTorch / CUDA")
    ok, torch, err = safe_import("torch")
    print(f"torch.imported      : {format_bool(ok)}")
    if not ok:
        print(f"torch.error         : {err}")
        return

    print(f"torch.version       : {module_version(torch)}")
    cuda_available = bool(torch.cuda.is_available())
    print(f"cuda.available      : {format_bool(cuda_available)}")
    print(f"cuda.version        : {getattr(torch.version, 'cuda', None)}")

    if cuda_available:
        try:
            gpu_count = int(torch.cuda.device_count())
            print(f"gpu.count           : {gpu_count}")
            for idx in range(gpu_count):
                print(f"gpu.{idx}.name         : {torch.cuda.get_device_name(idx)}")
        except Exception as exc:  # noqa: BLE001
            print(f"gpu.info_error      : {exc}")
    else:
        print("gpu.name            : N/A (CUDA unavailable)")


def check_module(name: str, display_name: Optional[str] = None) -> None:
    label = display_name or name
    ok, module, err = safe_import(name)
    print(f"{label}.imported      : {format_bool(ok)}")
    if ok:
        print(f"{label}.version       : {module_version(module)}")
    else:
        print(f"{label}.error         : {err}")


def check_fitz_or_pymupdf() -> None:
    print_header("Other Dependencies")
    check_module("transformers")
    check_module("qwen_vl_utils")

    fitz_ok, fitz_module, fitz_err = safe_import("fitz")
    if fitz_ok:
        print("fitz.imported        : YES")
        print(f"fitz.version         : {module_version(fitz_module)}")
    else:
        pymupdf_ok, pymupdf_module, pymupdf_err = safe_import("pymupdf")
        print(f"fitz.imported        : {format_bool(pymupdf_ok)}")
        if pymupdf_ok:
            print(f"fitz.version         : {module_version(pymupdf_module)}")
        else:
            print("fitz.error           : fitz and pymupdf both unavailable")
            print(f"fitz.error_detail    : {fitz_err}")
            print(f"pymupdf.error_detail : {pymupdf_err}")

    pil_ok, pil_module, pil_err = safe_import("PIL")
    print(f"PIL.imported         : {format_bool(pil_ok)}")
    if pil_ok:
        print(f"PIL.version          : {module_version(pil_module)}")
    else:
        print(f"PIL.error            : {pil_err}")

    check_module("faiss")
    check_module("paddleocr")


def main() -> None:
    print("DocVisRAG Stage 0 Environment Check")
    check_python()
    check_torch()
    check_fitz_or_pymupdf()


if __name__ == "__main__":
    main()
