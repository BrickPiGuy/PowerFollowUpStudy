import torch, time

def bench_matmul(M=8192, N=8192, K=8192, dtype=torch.float16, iters=50, warmup=10):
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)

    # Warmup
    torch.cuda.synchronize()
    for _ in range(warmup):
        C = A @ B
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        C = A @ B
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    flops_per_gemm = 2.0 * M * N * K
    total_flops = flops_per_gemm * iters
    tflops = total_flops / elapsed / 1e12

    return tflops, elapsed

# Run FP16 benchmark
tflops_fp16, t_fp16 = bench_matmul(dtype=torch.float16)
print(f"FP16: {tflops_fp16:.2f} TFLOPS over {t_fp16:.2f}s")

# Run BF16 benchmark
if torch.cuda.is_bf16_supported():
    tflops_bf16, t_bf16 = bench_matmul(dtype=torch.bfloat16)
    print(f"BF16: {tflops_bf16:.2f} TFLOPS over {t_bf16:.2f}s")
else:
    print("BF16 not supported on this GPU.")