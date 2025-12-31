#!/usr/bin/env python3
"""
Memory Analysis for DeepSeek-R1-Distill-Qwen-7B GRPO Training
Calculates detailed memory requirements for training with DeepSpeed ZeRO-3
"""

def bytes_to_gb(bytes_val):
    """Convert bytes to GB"""
    return bytes_val / (1024**3)

def calculate_memory_requirements():
    """Calculate memory requirements for GRPO training"""

    print("=" * 80)
    print("MEMORY ANALYSIS: DeepSeek-R1-Distill-Qwen-7B with GRPO")
    print("=" * 80)

    # Model parameters
    model_params = 7_000_000_000  # 7B parameters

    # Configuration
    num_gpus = 2
    use_8bit = True
    use_4bit = False
    use_deepspeed_zero3 = True
    gradient_checkpointing = True

    # Training config
    per_device_train_batch = 1
    per_device_eval_batch = 2
    num_generations = 2  # Current config
    max_prompt_length = 1024
    max_completion_length = 1024
    max_seq_length = max_prompt_length + max_completion_length

    # LoRA config
    lora_r = 32
    lora_alpha = 32
    # Target modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj (7 modules)
    num_lora_modules = 7
    num_layers = 28  # DeepSeek-R1-Distill-Qwen-7B has 28 layers
    hidden_size = 3584  # Hidden dimension

    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Model: DeepSeek-R1-Distill-Qwen-7B ({model_params:,} parameters)")
    print(f"Quantization: {'8-bit' if use_8bit else '4-bit' if use_4bit else 'None (FP16/BF16)'}")
    print(f"DeepSpeed: ZeRO-{'3' if use_deepspeed_zero3 else '2'}")
    print(f"GPUs: {num_gpus}")
    print(f"Gradient Checkpointing: {gradient_checkpointing}")
    print(f"Training batch size per GPU: {per_device_train_batch}")
    print(f"Eval batch size per GPU: {per_device_eval_batch}")
    print(f"Num generations: {num_generations}")
    print(f"Max sequence length: {max_seq_length} tokens ({max_prompt_length} prompt + {max_completion_length} completion)")
    print(f"LoRA rank: {lora_r}, modules per layer: {num_lora_modules}")

    # ========================================================================
    # 1. MODEL WEIGHTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. MODEL WEIGHTS")
    print("=" * 80)

    if use_8bit:
        bytes_per_param = 1  # INT8
        model_memory_total = model_params * bytes_per_param
        print(f"8-bit quantization: {bytes_per_param} byte per parameter")
    elif use_4bit:
        bytes_per_param = 0.5
        model_memory_total = model_params * bytes_per_param
        print(f"4-bit quantization: {bytes_per_param} bytes per parameter")
    else:
        bytes_per_param = 2  # FP16/BF16
        model_memory_total = model_params * bytes_per_param
        print(f"FP16/BF16: {bytes_per_param} bytes per parameter")

    print(f"Total model memory: {bytes_to_gb(model_memory_total):.2f} GB")

    if use_deepspeed_zero3:
        model_memory_per_gpu = model_memory_total / num_gpus
        print(f"With ZeRO-3 sharding across {num_gpus} GPUs: {bytes_to_gb(model_memory_per_gpu):.2f} GB per GPU")
    else:
        model_memory_per_gpu = model_memory_total
        print(f"Without ZeRO-3 (replicated): {bytes_to_gb(model_memory_per_gpu):.2f} GB per GPU")

    # ========================================================================
    # 2. LORA ADAPTERS
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. LORA ADAPTERS")
    print("=" * 80)

    # Each LoRA module has: input_dim x r + r x output_dim
    # For attention: input_dim = output_dim = hidden_size
    # For MLP: gate/up proj are hidden_size x (4*hidden_size), down_proj is reverse

    # Attention projections (q,k,v,o): 4 modules
    attn_lora_params = 4 * num_layers * (hidden_size * lora_r + lora_r * hidden_size)

    # MLP projections (gate, up, down): 3 modules
    # gate_proj, up_proj: hidden_size -> intermediate (4*hidden_size)
    # down_proj: intermediate -> hidden_size
    intermediate_size = 4 * hidden_size
    mlp_lora_params = num_layers * (
        2 * (hidden_size * lora_r + lora_r * intermediate_size) +  # gate, up
        (intermediate_size * lora_r + lora_r * hidden_size)  # down
    )

    total_lora_params = attn_lora_params + mlp_lora_params
    lora_memory = total_lora_params * 2  # FP16 (LoRA adapters not quantized)

    print(f"LoRA attention parameters: {attn_lora_params:,}")
    print(f"LoRA MLP parameters: {mlp_lora_params:,}")
    print(f"Total LoRA parameters: {total_lora_params:,}")
    print(f"LoRA memory (FP16): {bytes_to_gb(lora_memory):.2f} GB")
    print(f"LoRA percentage of model: {100 * total_lora_params / model_params:.2f}%")

    if use_deepspeed_zero3:
        lora_memory_per_gpu = lora_memory / num_gpus
        print(f"With ZeRO-3 sharding: {bytes_to_gb(lora_memory_per_gpu):.2f} GB per GPU")
    else:
        lora_memory_per_gpu = lora_memory
        print(f"Without sharding: {bytes_to_gb(lora_memory_per_gpu):.2f} GB per GPU")

    # ========================================================================
    # 3. OPTIMIZER STATES
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. OPTIMIZER STATES (AdamW)")
    print("=" * 80)

    # Only LoRA parameters are trained
    trainable_params = total_lora_params

    # AdamW: 2 states (momentum + variance) × 4 bytes (FP32) per parameter
    optimizer_memory = trainable_params * 2 * 4

    print(f"Trainable parameters (LoRA only): {trainable_params:,}")
    print(f"Optimizer states: 2 × 4 bytes (FP32) per parameter")
    print(f"Total optimizer memory: {bytes_to_gb(optimizer_memory):.2f} GB")

    if use_deepspeed_zero3:
        print(f"With ZeRO-3: Offloaded to CPU (0 GB GPU memory)")
        optimizer_memory_per_gpu = 0
    else:
        optimizer_memory_per_gpu = optimizer_memory / num_gpus
        print(f"With ZeRO-2 sharding: {bytes_to_gb(optimizer_memory_per_gpu):.2f} GB per GPU")

    # ========================================================================
    # 4. GRADIENTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. GRADIENTS")
    print("=" * 80)

    # Gradients stored in FP16/BF16 for LoRA parameters
    gradient_memory = trainable_params * 2  # FP16

    print(f"Gradient memory (FP16): {bytes_to_gb(gradient_memory):.2f} GB")

    if use_deepspeed_zero3:
        gradient_memory_per_gpu = gradient_memory / num_gpus
        print(f"With ZeRO-3 sharding: {bytes_to_gb(gradient_memory_per_gpu):.2f} GB per GPU")
    else:
        gradient_memory_per_gpu = gradient_memory / num_gpus
        print(f"With ZeRO-2 sharding: {bytes_to_gb(gradient_memory_per_gpu):.2f} GB per GPU")

    # ========================================================================
    # 5. ACTIVATIONS (Training)
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. ACTIVATIONS (During Training)")
    print("=" * 80)

    batch_size = per_device_train_batch

    if gradient_checkpointing:
        # With gradient checkpointing: only store activations for current layer
        # Rough estimate: ~2 layers worth of activations
        activation_layers = 2
        activation_memory_train = (
            batch_size * max_seq_length * hidden_size * 2 *  # 2 bytes (FP16)
            activation_layers
        )
        print(f"Gradient checkpointing enabled: storing ~{activation_layers} layers")
    else:
        # Without gradient checkpointing: store all layer activations
        activation_memory_train = (
            batch_size * max_seq_length * hidden_size * 2 *  # 2 bytes (FP16)
            num_layers
        )
        print(f"No gradient checkpointing: storing all {num_layers} layers")

    print(f"Batch size: {batch_size}, Sequence length: {max_seq_length}")
    print(f"Activation memory (training): {bytes_to_gb(activation_memory_train):.2f} GB per GPU")

    # ========================================================================
    # 6. KV CACHE (Generation)
    # ========================================================================
    print("\n" + "=" * 80)
    print("6. KV CACHE (During Generation)")
    print("=" * 80)

    # KV cache stores key and value tensors for each layer
    # Size: num_layers × 2 (key+value) × batch × num_heads × seq_len × head_dim
    num_heads = 28  # DeepSeek-R1-Distill-Qwen-7B
    head_dim = hidden_size // num_heads

    # During GRPO, we generate multiple completions per prompt
    generation_batch = per_device_eval_batch * num_generations

    kv_cache_memory_per_completion = (
        num_layers * 2 *  # key + value
        max_completion_length *  # Only cache for generated tokens
        num_heads * head_dim *
        2  # FP16
    )

    kv_cache_memory = kv_cache_memory_per_completion * generation_batch

    print(f"Generation batch: {per_device_eval_batch} × {num_generations} generations = {generation_batch}")
    print(f"KV cache per completion: {bytes_to_gb(kv_cache_memory_per_completion):.2f} GB")
    print(f"Total KV cache: {bytes_to_gb(kv_cache_memory):.2f} GB per GPU")

    # ========================================================================
    # 7. GATHERED PARAMETERS (ZeRO-3 Generation)
    # ========================================================================
    print("\n" + "=" * 80)
    print("7. GATHERED PARAMETERS (ZeRO-3 During Generation)")
    print("=" * 80)

    if use_deepspeed_zero3:
        # During generation, ZeRO-3 must gather ALL parameters to one GPU
        if use_8bit:
            # 8-bit parameters gathered as FP16 for computation
            gathered_memory = model_params * 2  # Converted to FP16
            print(f"WARNING: 8-bit params converted to FP16 during gather!")
            print(f"Gathered model memory: {bytes_to_gb(gathered_memory):.2f} GB")
        elif use_4bit:
            # 4-bit parameters gathered as FP16 for computation
            gathered_memory = model_params * 2  # Converted to FP16
            print(f"WARNING: 4-bit params converted to FP16 during gather!")
            print(f"Gathered model memory: {bytes_to_gb(gathered_memory):.2f} GB")
        else:
            gathered_memory = model_memory_total
            print(f"Gathered model memory: {bytes_to_gb(gathered_memory):.2f} GB")

        gathered_memory_per_gpu = gathered_memory  # All on one GPU during generation!
    else:
        gathered_memory_per_gpu = 0
        print("No gathering needed (not using ZeRO-3)")

    # ========================================================================
    # 8. TOTAL MEMORY SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("TOTAL MEMORY REQUIREMENTS PER GPU")
    print("=" * 80)

    print("\nDURING TRAINING:")
    training_total = (
        model_memory_per_gpu +
        lora_memory_per_gpu +
        optimizer_memory_per_gpu +
        gradient_memory_per_gpu +
        activation_memory_train
    )
    print(f"  Model weights:     {bytes_to_gb(model_memory_per_gpu):>8.2f} GB")
    print(f"  LoRA adapters:     {bytes_to_gb(lora_memory_per_gpu):>8.2f} GB")
    print(f"  Optimizer states:  {bytes_to_gb(optimizer_memory_per_gpu):>8.2f} GB")
    print(f"  Gradients:         {bytes_to_gb(gradient_memory_per_gpu):>8.2f} GB")
    print(f"  Activations:       {bytes_to_gb(activation_memory_train):>8.2f} GB")
    print(f"  {'─' * 35}")
    print(f"  TOTAL:             {bytes_to_gb(training_total):>8.2f} GB")

    print("\nDURING GENERATION (GRPO evaluation step):")
    if use_deepspeed_zero3:
        generation_total = (
            gathered_memory_per_gpu +  # Full model gathered!
            lora_memory +  # LoRA not sharded
            kv_cache_memory
        )
        print(f"  Gathered model:    {bytes_to_gb(gathered_memory_per_gpu):>8.2f} GB  ⚠️  (ALL params on one GPU!)")
        print(f"  LoRA adapters:     {bytes_to_gb(lora_memory):>8.2f} GB")
        print(f"  KV cache:          {bytes_to_gb(kv_cache_memory):>8.2f} GB")
    else:
        generation_total = (
            model_memory_per_gpu +
            lora_memory_per_gpu +
            kv_cache_memory
        )
        print(f"  Model weights:     {bytes_to_gb(model_memory_per_gpu):>8.2f} GB")
        print(f"  LoRA adapters:     {bytes_to_gb(lora_memory_per_gpu):>8.2f} GB")
        print(f"  KV cache:          {bytes_to_gb(kv_cache_memory):>8.2f} GB")

    print(f"  {'─' * 35}")
    print(f"  TOTAL:             {bytes_to_gb(generation_total):>8.2f} GB")

    # ========================================================================
    # 9. RECOMMENDATIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)

    gpu_memory = 16  # 16GB GPUs

    print(f"\nGPU Memory Available: {gpu_memory} GB")
    print(f"Training requirement:  {bytes_to_gb(training_total):.2f} GB - {'✅ FITS' if training_total <= gpu_memory * (1024**3) else '❌ TOO LARGE'}")
    print(f"Generation requirement: {bytes_to_gb(generation_total):.2f} GB - {'✅ FITS' if generation_total <= gpu_memory * (1024**3) else '❌ TOO LARGE'}")

    if generation_total > gpu_memory * (1024**3):
        print("\n⚠️  PROBLEM: Generation phase exceeds GPU memory!")
        print("\nThe issue is ZeRO-3 parameter gathering during generation:")
        print(f"  - Model sharded: {bytes_to_gb(model_memory_per_gpu):.2f} GB per GPU")
        print(f"  - Model gathered: {bytes_to_gb(gathered_memory_per_gpu):.2f} GB (all on one GPU!)")

        if use_8bit or use_4bit:
            print(f"  - Additionally, quantized params are converted to FP16 during gather")
            print(f"  - This doubles the memory: {bytes_to_gb(model_memory_total):.2f} GB → {bytes_to_gb(gathered_memory_per_gpu):.2f} GB")

        print("\nSOLUTIONS:")
        print("\n1. Use ZeRO-2 instead of ZeRO-3 (no parameter sharding)")
        print("   - Keeps model replicated, no gathering needed")
        print("   - Trade-off: Higher training memory, but stable generation")

        print("\n2. Reduce generation memory:")
        print(f"   - Current: num_generations={num_generations}, eval_batch={per_device_eval_batch}")
        print(f"   - Try: num_generations=1, eval_batch=1")
        print(f"   - This reduces KV cache: {bytes_to_gb(kv_cache_memory):.2f} GB → ~{bytes_to_gb(kv_cache_memory_per_completion):.2f} GB")

        print("\n3. Disable quantization (rely on DeepSpeed alone)")
        print("   - Quantization causes conversion to FP16 during gathering")
        print("   - ZeRO-3 already provides memory savings")

        print("\n4. Use larger GPUs (24GB or 40GB)")
    else:
        print("\n✅ Configuration should fit in GPU memory!")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    calculate_memory_requirements()