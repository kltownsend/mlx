# Windowed SDPA backward for MLX (experimental fork)

This branch adds a memory-efficient backward pass for sliding-window attention to MLX's `mx.fast.scaled_dot_product_attention`. It sits on top of ml-explore/mlx#4734, which adds the `window_size` option to the forward kernels. Everything else is upstream MLX, under its MIT license.

I'm not maintaining this. It's an experiment from a lab, published so anyone who hits the same wall can use it or pick it up. If the MLX team wants any of it, great.

## What was broken?

Gemma 4 uses sliding-window attention in 50 of its 60 layers: each token only looks at the 1,024 tokens before it. MLX's tiled attention backward (#4563) only handled the plain causal case. Sliding-window layers fell back to building the full score matrix, so backward memory grew with the square of the sequence length. A LoRA training step on a 30,553-token example through stock mlx-lm ran out of memory at a 224 GB footprint on a 256 GB M5 Ultra Mac Studio.

#4734 doesn't change that for training. With `window_size=1024` on #4734 alone, backward memory still goes from 12.94 GB at 8,192 tokens to 50.5 GB at 16,384.

## What this changes

- The Metal forward is allowed to emit the logsumexp under a window. #4734's kernels already compute it; a guard blocked it.
- The backward primitive carries `window_size`, including in its cache key.
- The tiled backward skips key tiles entirely outside each query tile's window, and masks the edge tiles element by element.
- Two bookkeeping fixes the causal-only code never needed: dQ is initialized on each query tile's first visited key tile (it used to assume tile 0), and dK and dV are zeroed when a window plus `qL < kL` leaves key tiles unvisited.
- CUDA throws "NYI" for a windowed backward. #4734 has no CUDA window kernels, so there's nothing to extend there.
- A new test, `test_sdpa_window_grad`.

There's also an opt-in experiment, `MLX_SDPA_VJP_FP32_TILES=1`, off by default. It computes the backward's score and dP tiles in fp32 instead of the input dtype. On real Gemma 4 activations it cut attention-gradient error about 4x per layer, for the windowed and the existing causal path. It costs about 1.6x the backward's memory as written (it upcasts whole inputs, not tiles) and it didn't produce a better trained adapter.

## How I tested it

All on an M5 Ultra Mac Studio, bf16, batch 1.

- Gradient parity against an fp32 reference: 867 shapes (fp32/fp16/bf16, head dims 64 to 512, GQA, `qL < kL`, windows 1 to 5,000, causal and not, tile sizes 256 and 1,024). Every window from 7 up holds within 2x the shipped causal path's own error, except one bf16 case that missed by 0.0004. Window 1 doesn't, because its true dQ and dK are zero and only rounding is left; with TF32 off it's exactly zero.
- Memory, windowed backward, D=256, 16 KV heads: 1.24 GB at 8,192 tokens, 2.49 GB at 16,384, 4.74 GB at 32,768. The same layer with a boolean mask needs 198 GB at 32,768.
- `python/tests/test_fast_sdpa.py` passes.
- A real job: an 86-step LoRA fine-tune of Gemma 4 31B (longest example 30,553 tokens) through mlx-lm, with a small patch so Gemma's sliding layers pass `window_size` instead of a boolean mask. It trained in 2 h 17 min against 2 h 41 min for my earlier model-side workaround, with the same held-out validation loss (0.1496 against 0.1510).

Is it production-ready? I don't know. It passed everything I threw at it on one machine. It hasn't been reviewed by anyone who maintains MLX.

The full method and numbers will be written up as a lab on labs.layer2c.com.

---

Disclosure: the code, tests and this README were generated with Claude (Anthropic) working in my lab, and I reviewed them.
