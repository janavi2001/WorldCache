# Limitations

- No GPU or model dependencies were available in the repository-construction environment, so the
  checked-in measured results are clearly labeled smoke results. No numerical VLM speedup claim is
  made from them.
- Prefix-KV caching, video, multi-GPU disaggregation, and a generative world-model backend are not
  implemented. Requests for `prefix_kv` fail explicitly.
- The selected model's public image-feature method combines architecture-specific vision and
  connector work; this adapter reports that combined duration as vision encoding rather than
  inventing a separate projector measurement.
- The in-process scene registry and representation cache are bounded but not durable. Restarting the
  server loses registered scenes.
- Concurrent GPU calls should be serialized by a deployment worker or scheduler until adapter-level
  stream isolation is implemented.
- Text answers about possible actions are qualitative VLM reasoning, not physically accurate future
  prediction.

