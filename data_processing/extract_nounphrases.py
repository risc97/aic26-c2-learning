import os
import gc

# Prevent HuggingFace tokenizers from spawning child processes — avoids deadlocks
# when the DataLoader workers also use multiprocessing.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.utils
from argparse import ArgumentParser
import pickle as pkl
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


def filter_nounphrases(nps):
    if len(nps) == 0:
        return []
    phrases = {}
    for np in nps:
        if np not in phrases and len(np) > 1:
            phrases[np] = 1
    return [key for key in phrases]


def post_process_nounphrases(raw_outputs: str, org_sentence: str):
    lines = raw_outputs.splitlines()
    outputs = []
    for item in lines:
        line = item.strip()
        spans = line.split(". ")
        if len(spans) != 2:
            continue
        try:
            int(spans[0])
        except ValueError:
            continue
        if len(spans[1]) > len(org_sentence):
            continue
        if "(" in spans[1] or ")" in spans[1] or "excluded" in spans[1]:
            continue
        outputs.append(spans[1].strip())
    return filter_nounphrases(outputs)


def get_model(device=None, use_int4=False, target_batch_size: int = 5):
    torch.random.manual_seed(0)

    # ------------------------------------------------------------------
    # 1. CUDA / cuDNN performance settings
    # ------------------------------------------------------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------
    # 2. SDPA backend: prefer Flash > memory-efficient > fallback math
    #    Flash SDP is the fastest attention backend; mem-efficient is the
    #    next best.  Disable the vanilla math backend so PyTorch never
    #    silently falls back to it (it would be >3× slower and OOM-prone).
    # ------------------------------------------------------------------
    if device is not None:
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        except Exception:
            pass  # older PyTorch — the _attn_implementation kwarg still governs

    # ------------------------------------------------------------------
    # 3. Inductor autotuning — aggressive: worth the upfront compile cost
    #    for a multi-hour run.  max_autotune profiles dozens of Triton /
    #    ATEN / CUTLASS variants per matmul shape.  coordinate_descent
    #    adds another round of search over tile sizes.
    # ------------------------------------------------------------------
    try:
        torch._inductor.config.max_autotune = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = True       # fuse pointwise after matmul
        torch._inductor.config.fx_graph_cache = True         # persist compiled graphs across calls
        torch._inductor.config.triton.cudagraph_trees = True # PyTorch ≥2.3: tree of CUDA graphs
    except AttributeError:
        pass

    # Increase Dynamo cache so more recompilations stay hot (default 64).
    # A generation loop sees several shapes (prefill vs decode, varying batch).
    try:
        torch._dynamo.config.cache_size_limit = 256
    except AttributeError:
        pass

    # ------------------------------------------------------------------
    # 4. Flash Attention 2 — try the dedicated package first (less VRAM,
    #    faster), fall back to torch's built-in SDPA.
    # ------------------------------------------------------------------
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        print("Using Flash Attention 2")
    except ImportError:
        attn_impl = "sdpa"
        print("Flash Attention 2 not found, using SDPA "
              "(pip install flash-attn --no-build-isolation for speedup)")

    load_kwargs = {
        "device_map": device if device is not None else "cpu",
        "torch_dtype": torch.float16,
        "_attn_implementation": attn_impl if device is not None else "eager",
    }

    if use_int4:
        # INT4 weight-only quantization: ~4× less VRAM for weights, ~1.5–2× faster.
        # torch.compile can't optimize bitsandbytes layers, so we skip it below.
        load_kwargs.update({
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        })
        load_kwargs.pop("torch_dtype")  # bitsandbytes manages its own dtypes
        print("Using INT4 quantization (bitsandbytes NF4)")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        **load_kwargs,
    )
    model.eval()

    # ------------------------------------------------------------------
    # 5. torch.compile — max-autotune + is_big_gpu bypass
    # ------------------------------------------------------------------
    if device is not None and not use_int4:
        # ------------------------------------------------------------------
        # max-autotune bypass: torch._inductor.utils.is_big_gpu() has a
        # hard-coded floor of 68 SMs (RTX 3080 baseline).  GPUs below that
        # are rejected for max_autotune_gemm — the inductor's GEMM kernel
        # autotuner that profiles Triton / ATEN / CUTLASS variants per
        # matmul shape and picks the fastest.  Monkey-patching it → True
        # forces it on regardless of SM count.  Cost: first-forward
        # compilation is MUCH slower, but the selected kernels are optimal
        # for THIS GPU's specific tensor shapes.
        # ------------------------------------------------------------------
        if torch.cuda.is_available():
            import torch._inductor.utils as _inductor_utils
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            # Bust the lru_cache on the original (if decorated) so stale
            # cached results don't survive, then replace with our forced-True.
            if hasattr(_inductor_utils.is_big_gpu, "cache_clear"):
                _inductor_utils.is_big_gpu.cache_clear()
            _inductor_utils.is_big_gpu = lambda _device=0: True
            print(f"max-autotune GEMM forced ON  "
                  f"(GPU has {sms} SMs; threshold is 68)")
            print("First-forward compilation will be SLOW — expect 1–5 min "
                  "of autotuning, then fast inference afterwards.")

        compile_mode = "max-autotune"
        print(f"Compiling model with torch.compile (mode={compile_mode}) …")
        try:
            model = torch.compile(model, mode=compile_mode, fullgraph=False)
        except Exception as exc:
            print(f"  torch.compile failed ({exc}), falling back to eager mode")

        # ------------------------------------------------------------------
        # 6. Multi-pass warm-up — several forward passes to trigger all
        #    compilation stages, CUDA graph capture, and kernel autotuning.
        #    We warm up with the target batch size so the CUDA graphs are
        #    captured at the shape they'll actually see.
        # ------------------------------------------------------------------
        print("Warming up (compilation + CUDA graph capture) …")
        dummy_ids = torch.randint(0, 100, (target_batch_size, 64), device=device)
        dummy_mask = torch.ones(target_batch_size, 64, device=device)
        for _ in range(5):
            model.generate(
                dummy_ids, attention_mask=dummy_mask,
                max_new_tokens=1, do_sample=False,
                pad_token_id=model.config.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Warm-up complete.")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    tokenizer.padding_side = "left"  # mandatory for decoder-only batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_chat_message(sentence):
    query_template = (
        "Extract the primary noun phrases from the sentence below.\n\n"
        "RULES:\n"
        "- Target: Only unique noun phrases representing the main subjects.\n"
        "- Exclude: General location nouns (e.g., home, street, school, university).\n"
        "- Exclude: Image tags (e.g., [stock photo], [news photo]).\n\n"
        "SENTENCE: \"{SENTENCE}\"\n\n"
        "OUTPUT FORMAT:\n"
        "Provide a strictly numbered list. Start immediately with '1.'"
    )
    query = query_template.replace("{SENTENCE}", sentence)

    message = [
        {
            "role": "system", 
            "content": "You are an automated data extraction script. You output raw data arrays only. You never generate conversational text, greetings, or conclusions."
        },
        {
            "role": "user", 
            "content": (
                "Extract the primary noun phrases from the sentence below.\n\n"
                "RULES:\n"
                "- Target: Only unique noun phrases representing the main subjects.\n"
                "- Exclude: General location nouns (e.g., home, street, school, university).\n"
                "- Exclude: Image tags (e.g., [stock photo], [news photo]).\n\n"
                "SENTENCE: \"[stock photo] A woman wearing a red dress stands next to a fireplace likely at a school.\"\n\n"
                "OUTPUT FORMAT:\n"
                "Provide a strictly numbered list. Start immediately with '1.'"
            )
        },
        {
            "role": "assistant", 
            # This perfect 1-shot mapping forces the model to understand what to keep and what to throw away
            "content": "1. A woman\n2. a red dress\n3. fireplace"
        },
        {
            "role": "user", 
            "content": query
        },
    ]
    return message


def get_chat_messages(sentences):
    return [get_chat_message(s) for s in sentences]


class CaptionDataset(torch.utils.data.Dataset):
    """Lightweight dataset: only stores index->caption mapping, not the full PKL."""

    def __init__(self, pkl_path: str):
        print(f"Loading metadata from {pkl_path} ...")
        with open(pkl_path, "rb") as fp:
            data = pkl.load(fp)

        self.items = []  # list of (index, caption)
        for k, v in data.items():
            caption = v["captions"]["shortLLA_captions"]
            self.items.append((int(k), caption))

        # Free the full dict immediately
        del data
        gc.collect()

        self.items.sort(key=lambda x: x[0])
        print(f"  {len(self.items)} captions loaded, metadata freed from RAM")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        index, caption = self.items[idx]
        return index, caption


class DataCollator:
    """Collates caption batches for generation.

    When *target_batch_size* is set and the batch is smaller (last batch of an
    epoch), the collator pads it by duplicating the first real item so every
    forward pass sees the same shape.  This keeps torch.compile CUDA graphs
    stable — without it the last batch causes a one-off recompilation on every
    shard.  *real_count* is threaded through so ``run_batch`` can strip the
    padding before post-processing.
    """

    def __init__(self, tokenizer, target_batch_size: int | None = None):
        self.tokenizer = tokenizer
        self.target_batch_size = target_batch_size

    def __call__(self, batch):
        indices = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        real_count = len(batch)

        # Pad to target_batch_size for CUDA-graph-friendly static shapes
        if self.target_batch_size is not None and real_count < self.target_batch_size:
            pad = self.target_batch_size - real_count
            # Duplicate the first item as padding — we'll strip results later
            indices = indices + [indices[0]] * pad
            captions = captions + [captions[0]] * pad

        messages = get_chat_messages(captions)
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        inputs = self.tokenizer(
            prompts, padding=True, return_tensors="pt", max_length=1024, truncation=True
        )
        return inputs, indices, captions, real_count


def run_batch(model, tokenizer, device, batch, max_new_tokens=50):
    model_inputs, indices, captions, real_count = batch
    # Pin-to-device transfer in one shot (non_blocking lets the CPU continue)
    model_inputs = {k: v.to(device, non_blocking=True) for k, v in model_inputs.items()}

    with torch.inference_mode():
        generate_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False,
            output_scores=False,
            output_hidden_states=False,
            output_attentions=False,
        )

    # Slice off the input tokens, keep only generated
    input_len = model_inputs["input_ids"].shape[1]
    generate_ids = generate_ids[:, input_len:]
    model_outputs = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )

    # Strip padding items (added by DataCollator for CUDA graph consistency)
    indices = indices[:real_count]
    model_outputs = model_outputs[:real_count]
    captions = captions[:real_count]

    outputs = {}
    for index, model_output, cap in zip(indices, model_outputs, captions):
        nps = post_process_nounphrases(model_output, cap)
        outputs[index] = nps
    return outputs


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--metadata-pkl-file", type=str, default="./data_dir/images/cc3m_images.pkl",
        help="Path to the metadata pickle from Step 2"
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Max tokens to generate per caption (most nounphrase lists are short)")
    parser.add_argument("--int4", default=False, action="store_true",
                        help="Use bitsandbytes INT4 quantization (less VRAM, allows larger batches)")
    parser.add_argument("--overwrite-metadata-file", default=False, action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu is not None and args.gpu >= 0 else "cpu"
    metadata_file = Path(args.metadata_pkl_file)

    # Checkpoint file for nounphrases only (small, ~200 MB)
    np_checkpoint_path = metadata_file.parent / f"{metadata_file.stem}_np_checkpoint{metadata_file.suffix}"

    if args.overwrite_metadata_file:
        output_file = metadata_file
    else:
        output_file = metadata_file.parent / f"{metadata_file.stem}_nounphrases{metadata_file.suffix}"

    # --- Load model ---
    print("Loading Qwen3.5-0.8B ...")
    model, tokenizer = get_model(device=device, use_int4=args.int4,
                                 target_batch_size=args.batch_size)
    print("Model loaded.")

    # --- Load captions (lightweight, discards full PKL after extracting captions) ---
    dataset = CaptionDataset(pkl_path=str(metadata_file))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=DataCollator(tokenizer=tokenizer,
                                target_batch_size=args.batch_size),
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=(args.workers > 0),
        prefetch_factor=4 if args.workers > 0 else None,
        persistent_workers=False,  # avoid worker memory accumulation on long runs
    )

    # --- Resume from nounphrase checkpoint if available ---
    nounphrases = {}
    if np_checkpoint_path.exists():
        print(f"Resuming from nounphrase checkpoint: {np_checkpoint_path}")
        with open(np_checkpoint_path, "rb") as fp:
            nounphrases = pkl.load(fp)
        print(f"  {len(nounphrases)} nounphrase entries already computed")

    # --- Run inference ---
    print(f"Processing {len(dataset)} captions (batch_size={args.batch_size}) ...")
    for i, batch in enumerate(tqdm(data_loader)):
        # Unpack: (inputs, indices, captions, real_count)
        # real_count strips padding duplicates added for CUDA graph consistency
        batch_indices = batch[1][:batch[3]]
        if all(idx in nounphrases for idx in batch_indices):
            continue

        outputs = run_batch(model, tokenizer, device, batch, max_new_tokens=args.max_new_tokens)

        for k, v in outputs.items():
            nounphrases[k] = v

        if (i + 1) % 100 == 0:
            # Save nounphrase checkpoint to disk
            with open(np_checkpoint_path, "wb") as fp:
                pkl.dump(nounphrases, fp)
            # Periodic cleanup to prevent GC pressure and CUDA memory fragmentation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Final merge: add nounphrases to the full PKL ---
    print(f"Saving nounphrase checkpoint ({len(nounphrases)} entries) ...")
    with open(np_checkpoint_path, "wb") as fp:
        pkl.dump(nounphrases, fp)

    # Free GPU resources before CPU-heavy merge
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading full metadata for merge ...")
    with open(metadata_file, "rb") as fp:
        metadata = pkl.load(fp)

    missing = 0
    for k in metadata:
        k_int = int(k)
        if k_int in nounphrases:
            metadata[k]["nounphrases"] = nounphrases[k_int]
        else:
            metadata[k]["nounphrases"] = []
            missing += 1

    print(f"Nounphrases merged: {len(nounphrases)} found, {missing} missing")
    print(f"Saving to {output_file} ...")
    with open(output_file, "wb") as fp:
        pkl.dump(metadata, fp)

    # Clean up checkpoint on success
    if np_checkpoint_path.exists():
        np_checkpoint_path.unlink()

    print("Done.")


if __name__ == "__main__":
    main()
