from dataclasses import dataclass


@dataclass
class ProjectConfig:
    """Central project configuration for DocVisRAG."""

    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    data_dir: str = "data"
    output_dir: str = "data/outputs"
    index_dir: str = "data/indexes"
    device: str = "cuda"
    top_k: int = 3
