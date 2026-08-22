from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from typing import Any

from worldcache.utils.gpu import memory_stats, synchronize, tensor_nbytes
from worldcache.utils.profiling import profiling_range

from .base import GenerationConfig, GenerationResult, StageMetrics, VLMAdapter


@dataclass
class SmolProcessedScene:
    vision_inputs: dict[str, Any]
    token_prefix: Any
    token_suffix: Any


@dataclass
class SmolVisualState:
    image_hidden_states: Any
    token_prefix: Any
    token_suffix: Any


class SmolVLMAdapter(VLMAdapter):
    """SmolVLM2 adapter with post-vision-encoder embedding reuse.

    Compatibility is checked at startup. The implementation splices question tokens into a
    processor-generated sentinel prompt, preserving the exact image-token expansion while
    avoiding image preprocessing on visual-cache hits.
    """

    SENTINEL = "WCQUESTION_SENTINEL_9F70C8E1"

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        revision: str = "main",
        processor_revision: str = "main",
        dtype: str = "float16",
        device: str = "cuda:0",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Install WorldCache model dependencies: pip install -e '.[model]'"
            ) from exc
        self.torch = torch
        self.model_id = model_id
        self.revision = revision
        self.processor_revision = processor_revision
        self.dtype_name = dtype
        self.device = device
        torch_dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=processor_revision)
        self.model = (
            AutoModelForVision2Seq.from_pretrained(
                model_id, revision=revision, torch_dtype=torch_dtype
            )
            .to(device)
            .eval()
        )
        get_features = getattr(self.model, "get_image_features", None)
        if get_features is None:
            get_features = getattr(getattr(self.model, "model", None), "get_image_features", None)
        if get_features is None:
            raise RuntimeError("selected Transformers model does not expose get_image_features")
        self._get_image_features = get_features

    def _messages(self) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": "Answer questions about the physical scene accurately and concisely.",
            },
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": self.SENTINEL}],
            },
        ]

    @staticmethod
    def _find_subsequence(values: list[int], needle: list[int]) -> int:
        for index in range(len(values) - len(needle) + 1):
            if values[index : index + len(needle)] == needle:
                return index
        raise RuntimeError("could not locate question sentinel in processed chat template")

    def preprocess_scene(self, scene: bytes) -> tuple[SmolProcessedScene, StageMetrics]:
        from PIL import Image

        synchronize(self.device)
        started = time.perf_counter_ns()
        image = Image.open(io.BytesIO(scene)).convert("RGB")
        prompt = self.processor.apply_chat_template(
            self._messages(), add_generation_prompt=True, tokenize=False
        )
        batch = self.processor(text=prompt, images=[image], return_tensors="pt")
        sentinel_ids = self.processor.tokenizer.encode(self.SENTINEL, add_special_tokens=False)
        ids = batch["input_ids"][0].tolist()
        offset = self._find_subsequence(ids, sentinel_ids)
        prefix = batch["input_ids"][:, :offset]
        suffix = batch["input_ids"][:, offset + len(sentinel_ids) :]
        vision_inputs = {
            key: value
            for key, value in batch.items()
            if key in {"pixel_values", "pixel_attention_mask"}
        }
        synchronize(self.device)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return SmolProcessedScene(vision_inputs, prefix, suffix), StageMetrics(
            preprocessing_ms=elapsed
        )

    def encode_scene(
        self, processed_scene: SmolProcessedScene
    ) -> tuple[SmolVisualState, StageMetrics]:
        torch = self.torch
        synchronize(self.device)
        started = time.perf_counter_ns()
        with profiling_range("host_to_device_transfer"):
            inputs = {
                key: value.to(self.device) for key, value in processed_scene.vision_inputs.items()
            }
        after_copy = time.perf_counter_ns()
        with torch.inference_mode():
            hidden = self._get_image_features(**inputs)
        synchronize(self.device)
        finished = time.perf_counter_ns()
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        metrics = StageMetrics(
            host_to_device_ms=(after_copy - started) / 1_000_000,
            vision_encoding_ms=(finished - after_copy) / 1_000_000,
            vision_encoder_executions=1,
            visual_tokens=int(hidden.shape[-2]),
        )
        return SmolVisualState(
            hidden, processed_scene.token_prefix, processed_scene.token_suffix
        ), metrics

    def build_multimodal_prefix(self, visual_state: SmolVisualState, system_prompt: str) -> Any:
        return visual_state

    def _prompt_inputs(self, state: SmolVisualState, question: str) -> dict[str, Any]:
        torch = self.torch
        question_ids = self.processor.tokenizer(
            question, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        input_ids = torch.cat((state.token_prefix, question_ids, state.token_suffix), dim=1).to(
            self.device
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "image_hidden_states": state.image_hidden_states,
        }

    def generate_from_scene(
        self, visual_state: SmolVisualState, question: str, generation_config: GenerationConfig
    ) -> GenerationResult:
        from transformers import TextIteratorStreamer

        torch = self.torch
        torch.manual_seed(generation_config.seed)
        inputs = self._prompt_inputs(visual_state, question)
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            **inputs,
            max_new_tokens=generation_config.max_new_tokens,
            do_sample=generation_config.temperature > 0,
            streamer=streamer,
        )
        if generation_config.temperature > 0:
            kwargs["temperature"] = generation_config.temperature
        result_box: list[Any] = []
        errors: list[BaseException] = []

        def run() -> None:
            try:
                with torch.inference_mode():
                    result_box.append(self.model.generate(**kwargs))
            except BaseException as exc:
                errors.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        synchronize(self.device)
        started = time.perf_counter_ns()
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        chunks: list[str] = []
        first_token_ns: int | None = None
        with profiling_range("language_prefill_and_token_decoding"):
            for text in streamer:
                if first_token_ns is None:
                    synchronize(self.device)
                    first_token_ns = time.perf_counter_ns()
                chunks.append(text)
        worker.join()
        if errors:
            raise RuntimeError("model generation failed") from errors[0]
        synchronize(self.device)
        finished = time.perf_counter_ns()
        if not result_box:
            raise RuntimeError("generation worker failed")
        output = result_box[0]
        generated = output[0, inputs["input_ids"].shape[1] :].tolist()
        ttft = ((first_token_ns or finished) - started) / 1_000_000
        allocated, reserved = memory_stats(self.device)
        metrics = StageMetrics(
            prefill_ms=ttft,
            time_to_first_token_ms=ttft,
            decode_ms=(finished - (first_token_ns or finished)) / 1_000_000,
            end_to_end_ms=(finished - started) / 1_000_000,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            visual_tokens=int(visual_state.image_hidden_states.shape[-2]),
            generated_tokens=len(generated),
            gpu_memory_allocated=allocated,
            gpu_memory_reserved=reserved,
        )
        return GenerationResult("".join(chunks).strip(), generated, metrics)

    def model_metadata(self) -> dict[str, Any]:
        image_processor = getattr(self.processor, "image_processor", None)
        return {
            "model_id": self.model_id,
            "model_revision": self.revision,
            "processor_revision": self.processor_revision,
            "dtype": self.dtype_name,
            "device": self.device,
            "processor_config": image_processor.to_dict() if image_processor else {},
            "visual_tokenization": {
                "image_token_id": getattr(self.model.config, "image_token_id", None),
                "adapter": "smolvlm-sentinel-splice-v1",
            },
            "supports_visual_embedding_cache": True,
            "embedding_location": "vision_encoder_output",
        }

    def value_metadata(self, value: Any) -> tuple[list[int], int, str]:
        if isinstance(value, SmolProcessedScene):
            size = (
                tensor_nbytes(value.vision_inputs)
                + tensor_nbytes(value.token_prefix)
                + tensor_nbytes(value.token_suffix)
            )
            return [], size, "cpu"
        if isinstance(value, SmolVisualState):
            tensor = value.image_hidden_states
            return list(tensor.shape), tensor_nbytes(value), str(tensor.device)
        return super().value_metadata(value)
