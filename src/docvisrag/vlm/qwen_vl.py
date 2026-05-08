import logging
import os
from pathlib import Path
from typing import Any, Optional


LOGGER = logging.getLogger(__name__)


class QwenVLClient:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        max_pixels: Optional[int] = None,
    ) -> None:
        env_model_id = os.getenv("DOCVISRAG_MODEL_ID")
        self.model_id = env_model_id or model_id
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.max_pixels = max_pixels
        self.local_files_only = os.getenv("DOCVISRAG_LOCAL_FILES_ONLY", "").lower() in {
            "1",
            "true",
            "yes",
        }

        self.model: Any = None
        self.processor: Any = None

        LOGGER.info("Initializing QwenVLClient with model: %s", self.model_id)
        self._log_cache_locations()

    def _log_cache_locations(self) -> None:
        hf_home = os.getenv("HF_HOME")
        hf_hub_cache = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
        transformers_cache = os.getenv("TRANSFORMERS_CACHE")
        LOGGER.info(
            "Cache env | HF_HOME=%s | HF_HUB_CACHE=%s | TRANSFORMERS_CACHE=%s",
            hf_home,
            hf_hub_cache,
            transformers_cache,
        )

        model_dir_name = f"models--{self.model_id.replace('/', '--')}"
        candidates = []
        if hf_hub_cache:
            candidates.append(Path(hf_hub_cache) / model_dir_name)
        if hf_home:
            candidates.append(Path(hf_home) / "hub" / model_dir_name)
            candidates.append(Path(hf_home) / model_dir_name)
            candidates.append(Path(hf_home) / "transformers" / model_dir_name)

        for c in candidates:
            if c.exists():
                LOGGER.info("Cache candidate exists: %s", c)

    def _resolve_model_class(self, model_ref: str, local_only: bool = False) -> Any:
        from transformers import AutoModelForVision2Seq

        model_type = ""
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_ref, local_files_only=local_only)
            model_type = str(getattr(cfg, "model_type", ""))
            LOGGER.info("Detected model_type: %s", model_type)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not detect model_type from config: %s", exc)

        # Qwen3-VL should not be loaded with Qwen2.5-VL class.
        if "qwen3_vl" in model_type or "Qwen3-VL" in model_ref:
            LOGGER.info("Using AutoModelForVision2Seq for Qwen3-VL.")
            return AutoModelForVision2Seq

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            LOGGER.info("Using Qwen2_5_VLForConditionalGeneration.")
            return Qwen2_5_VLForConditionalGeneration
        except Exception:  # noqa: BLE001
            LOGGER.info("Falling back to AutoModelForVision2Seq.")
            return AutoModelForVision2Seq

    def _build_model_kwargs(self) -> dict:
        kwargs = {
            "device_map": self.device_map,
            "torch_dtype": "auto",
        }

        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "load_in_4bit=True requires transformers with BitsAndBytesConfig "
                    "and bitsandbytes installed."
                ) from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        return kwargs

    def _load_model(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        try:
            from transformers import AutoProcessor
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import transformers.AutoProcessor. "
                "Please install/upgrade transformers."
            ) from exc

        processor_kwargs = {}
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        if self.local_files_only:
            processor_kwargs["local_files_only"] = True

        model_kwargs = self._build_model_kwargs()
        if self.local_files_only:
            model_kwargs["local_files_only"] = True

        try:
            LOGGER.info("Loading processor for model: %s", self.model_id)
            self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
            model_cls = self._resolve_model_class(
                self.model_id,
                local_only=bool(model_kwargs.get("local_files_only", False)),
            )
            LOGGER.info("Loading model weights (this may take a while on first run)...")
            self.model = model_cls.from_pretrained(self.model_id, **model_kwargs)
            try:
                from huggingface_hub import snapshot_download

                snapshot_path = snapshot_download(repo_id=self.model_id, local_files_only=True)
                LOGGER.info("Resolved local snapshot path: %s", snapshot_path)
            except Exception:  # noqa: BLE001
                pass
            LOGGER.info("Model loaded successfully.")
            return
        except Exception as first_exc:  # noqa: BLE001
            LOGGER.warning("Primary model loading failed, trying local cached snapshot: %s", first_exc)

        try:
            from huggingface_hub import snapshot_download

            snapshot_path = snapshot_download(
                repo_id=self.model_id,
                local_files_only=True,
            )
            LOGGER.info("Using cached snapshot: %s", snapshot_path)
            self.processor = AutoProcessor.from_pretrained(snapshot_path, local_files_only=True)
            model_cls = self._resolve_model_class(snapshot_path, local_only=True)
            self.model = model_cls.from_pretrained(snapshot_path, **self._build_model_kwargs())
            LOGGER.info("Model loaded successfully from local snapshot.")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load VLM model.\n"
                f"model_id: {self.model_id}\n"
                "Possible reasons:\n"
                "1) model not downloaded and network is unreachable;\n"
                "2) model id is wrong;\n"
                "3) GPU memory is insufficient;\n"
                "4) transformers version does not support this model family.\n"
                f"Original error: {exc}"
            ) from exc

    def answer_image(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 512,
    ) -> str:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        image_file = Path(image_path).expanduser()
        if not image_file.exists():
            raise FileNotFoundError(
                f"Image file does not exist: {image_file}. "
                "Please provide a valid local image path."
            )
        if not image_file.is_file():
            raise FileNotFoundError(f"Path is not a file: {image_file}")

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import qwen_vl_utils.process_vision_info. "
                "Please install qwen-vl-utils."
            ) from exc

        image_uri = str(image_file.resolve())
        self._load_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": question.strip()},
                ],
            }
        ]

        LOGGER.info("Preparing multimodal inputs...")
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        LOGGER.info("Running generation...")
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answer = outputs[0].strip() if outputs else ""
        LOGGER.info("Generation completed.")
        return answer
