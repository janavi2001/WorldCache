# Architecture

WorldCache separates scene identity, bounded storage, model-specific execution, and serving.
`WorldCacheEngine` owns a byte-aware representation cache and a separately bounded raw-scene
registry. A cache key covers normalized scene bytes plus model, processor, precision, visual
tokenization, and representation-level configuration. Revision changes therefore cannot reuse an
old entry.

The `VLMAdapter` boundary exposes preprocessing, vision encoding, multimodal-prefix construction,
cached generation, and uncached generation. `SmolVLMAdapter` caches post-vision-encoder image hidden
states on their existing device. It creates the image-token portion of the prompt once with a unique
sentinel, then splices tokenized questions into that exact prompt. A cache hit does not decode the
image, create pixel tensors, transfer pixels, or invoke `get_image_features`.

Level 0 runs all stages. Level 1 stores processor outputs and skips CPU image work. Level 2 stores
image hidden states and skips preprocessing, pixel transfer, and vision feature calculation. Level 3
is deliberately rejected because exact prefix-KV reuse is not implemented.

The branch endpoint resolves the requested representation once and runs language generation for
each branch. This is textual VLM-based physical-scene reasoning, not future-world simulation.

