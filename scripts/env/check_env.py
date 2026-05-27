import argparse
import importlib
import importlib.metadata as md
import platform
import sys
from typing import Any, Optional, Tuple


CORE_PACKAGES = [
    "torch",
    "torchvision",
    "transformers",
    "sentence-transformers",
    "accelerate",
    "huggingface-hub",
    "qwen-vl-utils",
    "pymupdf",
    "pillow",
    "faiss-cpu",
    "paddleocr",
]

VISUAL_PACKAGES = [
    "byaldi",
    "colpali-engine",
    "peft",
    "ml-dtypes",
    "pytrec-eval-terrier",
]


def format_bool(value: bool) -> str:
    return "YES" if value else "NO"


def safe_import(module_name: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        module = importlib.import_module(module_name)
        return True, module, None
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def dist_version(dist_name: str) -> str:
    try:
        return md.version(dist_name)
    except md.PackageNotFoundError:
        return "not installed"


def module_version(module: Any) -> str:
    if module is None:
        return "N/A"
    return str(getattr(module, "__version__", "unknown"))


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


def check_package_versions(packages: list[str]) -> None:
    print_header("Installed Package Versions")
    for package in packages:
        print(f"{package:24}: {dist_version(package)}")


def check_import(module_name: str, display_name: Optional[str] = None) -> None:
    label = display_name or module_name
    ok, module, err = safe_import(module_name)
    print(f"{label}.imported      : {format_bool(ok)}")
    if ok:
        print(f"{label}.version       : {module_version(module)}")
    else:
        print(f"{label}.error         : {err}")


def check_core_imports() -> None:
    print_header("Core Imports")
    check_import("transformers")
    check_import("qwen_vl_utils")

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

    check_import("PIL", "PIL")
    check_import("faiss")
    check_import("paddleocr")
    check_import("sentence_transformers")


def check_visual_imports() -> None:
    print_header("Visual Retrieval Imports")
    check_import("byaldi")
    check_import("colpali_engine")
    check_import("peft")

    ok, peft_save_and_load, err = safe_import("peft.utils.save_and_load")
    print(f"peft.save_load.imported : {format_bool(ok)}")
    if ok:
        has_tp_helper = hasattr(peft_save_and_load, "_maybe_shard_state_dict_for_tp")
        print(f"peft.tp_helper          : {format_bool(has_tp_helper)}")
        if not has_tp_helper:
            print(
                "peft.tp_helper.note     : expected for the pinned relaxed visual stack; "
                "set DOCVISRAG_STRICT_VISUAL_CHECK=1 to treat this as an error"
            )
    else:
        print(f"peft.save_load.error    : {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check DocVisRAG runtime environment.")
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Also check optional Byaldi/ColPali stage-9 dependencies.",
    )
    args = parser.parse_args()

    print("DocVisRAG Environment Check")
    check_python()
    check_torch()
    check_package_versions(CORE_PACKAGES + (VISUAL_PACKAGES if args.visual else []))
    check_core_imports()
    if args.visual:
        check_visual_imports()


if __name__ == "__main__":
    main()
