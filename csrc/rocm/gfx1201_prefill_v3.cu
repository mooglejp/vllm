// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// R1 benchmark-only implementation for
// docs/design/gfx1201_prefill_rearchitecture.md.  This source is loaded out
// of tree by the benchmark and is not part of CMake or a production import.
// The mapping is independently specified from raw FP8 bytes: K=64 staging,
// padded LDS, register accumulators, and direct global output for full tiles.

#ifdef __HIP_NO_HALF_OPERATORS__
  #undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
  #undef __HIP_NO_HALF_CONVERSIONS__
#endif

#include <torch/extension.h>

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cstdint>
#include <string>

namespace {

constexpr int kWaveSize = 32;
constexpr int kTile = 16;
constexpr int kKSlab = 64;
constexpr int kKAStride = 80;
constexpr int kKBStride = 72;

#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx1200__) || defined(__gfx1201__))

using Fp8 = rocwmma::float8_t;

__device__ __forceinline__ Fp8 fp8_from_bits(uint8_t bits) {
  Fp8 value;
  value.__x = bits;
  return value;
}

__device__ __forceinline__ uint8_t byte_at(const uint4& value, int index) {
  const uint32_t words[4] = {value.x, value.y, value.z, value.w};
  return static_cast<uint8_t>(words[index / 4] >> (8 * (index % 4)));
}

template <int BlockM, int WaveCount, bool TailStore, bool DirectBf16>
__global__ void raw_fp8_prefill_kernel(const uint8_t* a, const uint8_t* b,
                                       float* output,
                                       __hip_bfloat16* bf16_output, int m,
                                       int n, int k, int output_stride) {
  static_assert(BlockM == WaveCount * 32);
  static_assert(WaveCount == 4 || WaveCount == 8);

  constexpr int kBlockN = 64;
  constexpr int kThreads = WaveCount * kWaveSize;
  constexpr int kFragM = 2;
  constexpr int kFragN = kBlockN / kTile;
  constexpr int kWaveTileM = BlockM / WaveCount;
  constexpr int kAElements = BlockM * kKAStride;
  constexpr int kBElements = kKSlab * kKBStride;
  constexpr int kTailElements = kWaveTileM * kBlockN;

  const int thread = threadIdx.x;
  if (thread >= kThreads) {
    return;
  }
  const int wave = thread / kWaveSize;
  const int lane = thread % kWaveSize;
  const int64_t block_m = static_cast<int64_t>(blockIdx.y) * BlockM;
  const int64_t block_n = static_cast<int64_t>(blockIdx.x) * kBlockN;
  const int wave_m_start = wave * kWaveTileM;

  __shared__ alignas(16) Fp8 tile_a[kAElements];
  __shared__ alignas(16) Fp8 tile_b[kBElements];
  // TailStore uses one wave-sized tile sequentially.  Full tiles write
  // directly to global output and never use this buffer.
  __shared__ alignas(16) float tail_output[kTailElements];

  using FragA = rocwmma::fragment<rocwmma::matrix_a, kTile, kTile, kTile, Fp8,
                                  rocwmma::row_major>;
  using FragB = rocwmma::fragment<rocwmma::matrix_b, kTile, kTile, kTile, Fp8,
                                  rocwmma::row_major>;
  using FragC = rocwmma::fragment<rocwmma::accumulator, kTile, kTile, kTile,
                                  float, rocwmma::row_major>;

  FragC accum[kFragM * kFragN];
  for (int index = 0; index < kFragM * kFragN; ++index) {
    rocwmma::fill_fragment(accum[index], 0.0f);
  }

  for (int slab_start = 0; slab_start < k; slab_start += kKSlab) {
    // A is [M,K].  Each assignment handles one contiguous 16-byte K chunk.
    // The aligned path is deliberately explicit; tails use scalar zero-fill.
    constexpr int kAChunks = BlockM * (kKSlab / 16);
    for (int index = thread; index < kAChunks; index += kThreads) {
      const int row = index / (kKSlab / 16);
      const int chunk = index % (kKSlab / 16);
      const int k_local = chunk * 16;
      const int64_t global_m = block_m + row;
      const int global_k = slab_start + k_local;
      Fp8* destination = tile_a + row * kKAStride + k_local;
      if (global_m < m && global_k + 16 <= k &&
          ((static_cast<int64_t>(global_m) * k + global_k) & 15) == 0) {
        const auto* source = reinterpret_cast<const uint4*>(
            a + static_cast<int64_t>(global_m) * k + global_k);
        const uint4 value = *source;
        *reinterpret_cast<uint4*>(destination) = value;
      } else {
        for (int offset = 0; offset < 16; ++offset) {
          const int read_k = global_k + offset;
          const uint8_t value =
              (global_m < m && read_k < k)
                  ? a[static_cast<int64_t>(global_m) * k + read_k]
                  : 0;
          destination[offset] = fp8_from_bits(value);
        }
      }
    }

    // B is checkpoint-shaped [N,K].  The staged view is [K,N] row-major,
    // so the global assignment advances through K before the next N row.
    constexpr int kBChunks = kBlockN * (kKSlab / 16);
    for (int index = thread; index < kBChunks; index += kThreads) {
      const int n_local = index / (kKSlab / 16);
      const int chunk = index % (kKSlab / 16);
      const int k_local = chunk * 16;
      const int64_t global_n = block_n + n_local;
      const int global_k = slab_start + k_local;
      const bool aligned =
          global_n < n && global_k + 16 <= k &&
          ((static_cast<int64_t>(global_n) * k + global_k) & 15) == 0;
      if (aligned) {
        const auto* source = reinterpret_cast<const uint4*>(
            b + static_cast<int64_t>(global_n) * k + global_k);
        const uint4 value = *source;
        for (int offset = 0; offset < 16; ++offset) {
          tile_b[(k_local + offset) * kKBStride + n_local] =
              fp8_from_bits(byte_at(value, offset));
        }
      } else {
        for (int offset = 0; offset < 16; ++offset) {
          const int read_k = global_k + offset;
          const uint8_t value =
              (global_n < n && read_k < k)
                  ? b[static_cast<int64_t>(global_n) * k + read_k]
                  : 0;
          tile_b[(k_local + offset) * kKBStride + n_local] =
              fp8_from_bits(value);
        }
      }
    }
    __syncthreads();

    // Four K=16 WMMA steps reuse one K=64 staged slab.  The LDS leading
    // dimensions include padding, so the fragment reads do not use a dense
    // output-tile LDS accumulator.
    for (int k_local = 0; k_local < kKSlab; k_local += kTile) {
      FragA frag_a[kFragM];
      FragB frag_b[kFragN];
      for (int frag_m = 0; frag_m < kFragM; ++frag_m) {
        rocwmma::load_matrix_sync(
            frag_a[frag_m],
            tile_a + (wave_m_start + frag_m * kTile) * kKAStride + k_local,
            kKAStride);
      }
      for (int frag_n = 0; frag_n < kFragN; ++frag_n) {
        rocwmma::load_matrix_sync(frag_b[frag_n],
                                  tile_b + k_local * kKBStride + frag_n * kTile,
                                  kKBStride);
      }
      for (int frag_m = 0; frag_m < kFragM; ++frag_m) {
        for (int frag_n = 0; frag_n < kFragN; ++frag_n) {
          const int index = frag_m * kFragN + frag_n;
          rocwmma::mma_sync(accum[index], frag_a[frag_m], frag_b[frag_n],
                            accum[index]);
        }
      }
    }
    __syncthreads();
  }

  if constexpr (DirectBf16) {
    // rocWMMA's supported accumulator store is FP32 on this SDK.  Reuse the
    // existing wave-sized LDS epilogue, then convert each value directly into
    // the caller-owned BF16 output without a global FP32 scratch buffer.
    for (int store_wave = 0; store_wave < WaveCount; ++store_wave) {
      if (wave == store_wave) {
        for (int frag_m = 0; frag_m < kFragM; ++frag_m) {
          for (int frag_n = 0; frag_n < kFragN; ++frag_n) {
            const int index = frag_m * kFragN + frag_n;
            rocwmma::store_matrix_sync(
                tail_output + frag_m * kTile * kBlockN + frag_n * kTile,
                accum[index], kBlockN);
          }
        }
      }
      __syncthreads();
      for (int index = thread; index < kTailElements; index += kThreads) {
        const int row = index / kBlockN;
        const int col = index % kBlockN;
        const int64_t global_m = block_m + store_wave * kWaveTileM + row;
        const int64_t global_n = block_n + col;
        if (global_m < m && global_n < n) {
          bf16_output[global_m * output_stride + global_n] =
              __float2bfloat16(tail_output[index]);
        }
      }
      __syncthreads();
    }
  } else if constexpr (!TailStore) {
    // The host selects this path only for a complete M/N tile.  Every store
    // is therefore within the caller-owned output and uses its true stride.
    for (int frag_m = 0; frag_m < kFragM; ++frag_m) {
      for (int frag_n = 0; frag_n < kFragN; ++frag_n) {
        const int index = frag_m * kFragN + frag_n;
        float* destination =
            output + (block_m + wave_m_start + frag_m * kTile) * output_stride +
            block_n + frag_n * kTile;
        rocwmma::store_matrix_sync(destination, accum[index], output_stride);
      }
    }
  } else {
    // Partial tiles use a small reusable wave tile rather than an output-size
    // LDS array.  Waves store one at a time, then all lanes write valid cells.
    for (int store_wave = 0; store_wave < WaveCount; ++store_wave) {
      if (wave == store_wave) {
        for (int frag_m = 0; frag_m < kFragM; ++frag_m) {
          for (int frag_n = 0; frag_n < kFragN; ++frag_n) {
            const int index = frag_m * kFragN + frag_n;
            rocwmma::store_matrix_sync(
                tail_output + frag_m * kTile * kBlockN + frag_n * kTile,
                accum[index], kBlockN);
          }
        }
      }
      __syncthreads();
      for (int index = thread; index < kTailElements; index += kThreads) {
        const int row = index / kBlockN;
        const int col = index % kBlockN;
        const int64_t global_m = block_m + store_wave * kWaveTileM + row;
        const int64_t global_n = block_n + col;
        if (global_m < m && global_n < n) {
          output[global_m * output_stride + global_n] = tail_output[index];
        }
      }
      __syncthreads();
    }
  }
}

#else

// Host-only/fatbin fallback.  The host checks gfx1201 before launch.
template <int BlockM, int WaveCount, bool TailStore, bool DirectBf16>
__global__ void raw_fp8_prefill_kernel(const uint8_t*, const uint8_t*, float*,
                                       __hip_bfloat16*, int, int, int, int) {}

#endif  // __HIP_DEVICE_COMPILE__ && gfx1200/gfx1201

void check_raw_inputs(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(),
              "R1 raw FP8 tensors must be CUDA/HIP tensors");
  TORCH_CHECK(a.device() == b.device(),
              "R1 raw FP8 inputs must share a device");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "R1 raw FP8 inputs must be 2D");
  TORCH_CHECK(
      a.scalar_type() == torch::kByte && b.scalar_type() == torch::kByte,
      "R1 inputs must contain raw FP8 bytes in uint8 tensors");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
              "R1 raw FP8 inputs must be contiguous");
  TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && b.size(0) > 0,
              "R1 dimensions must be positive");
  TORCH_CHECK(a.size(1) == b.size(1), "R1 inputs must have matching K");
}

void check_float_output(const torch::Tensor& a, const torch::Tensor& b,
                        const torch::Tensor& output) {
  TORCH_CHECK(output.is_cuda() && output.device() == a.device(),
              "R1 output must share the input device");
  TORCH_CHECK(output.dim() == 2 && output.scalar_type() == torch::kFloat,
              "R1 raw mapping output must be contiguous float32");
  TORCH_CHECK(output.is_contiguous() && output.size(0) == a.size(0) &&
                  output.size(1) == b.size(0),
              "R1 output must be contiguous and shape-matched");
}

void check_bf16_output(const torch::Tensor& a, const torch::Tensor& b,
                       const torch::Tensor& output) {
  TORCH_CHECK(output.is_cuda() && output.device() == a.device(),
              "R1 BF16 output must share the input device");
  TORCH_CHECK(output.dim() == 2 && output.scalar_type() == torch::kBFloat16,
              "R1 BF16 output must have bfloat16 dtype");
  TORCH_CHECK(output.is_contiguous() && output.size(0) == a.size(0) &&
                  output.size(1) == b.size(0),
              "R1 BF16 output must be contiguous and shape-matched");
}

void check_inputs(const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& output) {
  check_raw_inputs(a, b);
  check_float_output(a, b, output);
}

void check_gfx1201_device(const torch::Tensor& tensor) {
  const auto* properties = at::cuda::getDeviceProperties(tensor.get_device());
  TORCH_CHECK(std::string(properties->gcnArchName) == "gfx1201",
              "R1 raw FP8 mapping requires a gfx1201 device");
}

template <int BlockM, int WaveCount>
void launch_variant(torch::Tensor a, torch::Tensor b, torch::Tensor& output) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int m = static_cast<int>(a.size(0));
  const int n = static_cast<int>(b.size(0));
  const int k = static_cast<int>(a.size(1));
  const int m_tiles = (m + BlockM - 1) / BlockM;
  const int n_tiles = (n + 64 - 1) / 64;
  dim3 grid(static_cast<unsigned int>(n_tiles),
            static_cast<unsigned int>(m_tiles));
  dim3 block(WaveCount * kWaveSize);
  const bool full_tile = (m % BlockM == 0) && (n % 64 == 0);
  if (full_tile) {
    hipLaunchKernelGGL(
        (raw_fp8_prefill_kernel<BlockM, WaveCount, false, false>), grid, block,
        0, stream, a.data_ptr<uint8_t>(), b.data_ptr<uint8_t>(),
        output.data_ptr<float>(), nullptr, m, n, k, n);
  } else {
    hipLaunchKernelGGL((raw_fp8_prefill_kernel<BlockM, WaveCount, true, false>),
                       grid, block, 0, stream, a.data_ptr<uint8_t>(),
                       b.data_ptr<uint8_t>(), output.data_ptr<float>(), nullptr,
                       m, n, k, n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int BlockM, int WaveCount>
void launch_variant_bf16(torch::Tensor a, torch::Tensor b,
                         torch::Tensor& output) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int m = static_cast<int>(a.size(0));
  const int n = static_cast<int>(b.size(0));
  const int k = static_cast<int>(a.size(1));
  const int m_tiles = (m + BlockM - 1) / BlockM;
  const int n_tiles = (n + 64 - 1) / 64;
  dim3 grid(static_cast<unsigned int>(n_tiles),
            static_cast<unsigned int>(m_tiles));
  dim3 block(WaveCount * kWaveSize);
  const bool full_tile = (m % BlockM == 0) && (n % 64 == 0);
  auto* output_ptr =
      reinterpret_cast<__hip_bfloat16*>(output.data_ptr<at::BFloat16>());
  if (full_tile) {
    hipLaunchKernelGGL((raw_fp8_prefill_kernel<BlockM, WaveCount, false, true>),
                       grid, block, 0, stream, a.data_ptr<uint8_t>(),
                       b.data_ptr<uint8_t>(), nullptr, output_ptr, m, n, k, n);
  } else {
    hipLaunchKernelGGL((raw_fp8_prefill_kernel<BlockM, WaveCount, true, true>),
                       grid, block, 0, stream, a.data_ptr<uint8_t>(),
                       b.data_ptr<uint8_t>(), nullptr, output_ptr, m, n, k, n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx1200__) || defined(__gfx1201__))

__global__ void cast_float_to_bf16_kernel(const float* input,
                                          __hip_bfloat16* output, int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    output[index] = __float2bfloat16(input[index]);
  }
}

#else

__global__ void cast_float_to_bf16_kernel(const float*, __hip_bfloat16*, int) {}

#endif

void launch_cast_to_bf16(torch::Tensor& input, torch::Tensor& output) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda(),
              "R1 cast tensors must be CUDA/HIP tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat &&
                  output.scalar_type() == torch::kBFloat16,
              "R1 cast expects float32 scratch and bfloat16 output");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous() &&
                  input.sizes() == output.sizes(),
              "R1 cast tensors must be contiguous and shape-matched");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int size = static_cast<int>(input.numel());
  const int threads = 256;
  const int blocks = (size + threads - 1) / threads;
  hipLaunchKernelGGL(
      cast_float_to_bf16_kernel, dim3(blocks), dim3(threads), 0, stream,
      input.data_ptr<float>(),
      reinterpret_cast<__hip_bfloat16*>(output.data_ptr<at::BFloat16>()), size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_h1(torch::Tensor a, torch::Tensor b, torch::Tensor& output) {
  check_inputs(a, b, output);
  check_gfx1201_device(a);
  launch_variant<128, 4>(a, b, output);
}

void launch_h2(torch::Tensor a, torch::Tensor b, torch::Tensor& output) {
  check_inputs(a, b, output);
  check_gfx1201_device(a);
  launch_variant<256, 8>(a, b, output);
}

void launch_h1_bf16(torch::Tensor a, torch::Tensor b, torch::Tensor& scratch,
                    torch::Tensor& output) {
  check_inputs(a, b, scratch);
  check_gfx1201_device(a);
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kBFloat16 &&
                  output.is_contiguous() && output.sizes() == scratch.sizes(),
              "R1 H1 BF16 output must be contiguous and shape-matched");
  launch_variant<128, 4>(a, b, scratch);
  launch_cast_to_bf16(scratch, output);
}

void launch_h2_bf16(torch::Tensor a, torch::Tensor b, torch::Tensor& scratch,
                    torch::Tensor& output) {
  check_inputs(a, b, scratch);
  check_gfx1201_device(a);
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kBFloat16 &&
                  output.is_contiguous() && output.sizes() == scratch.sizes(),
              "R1 H2 BF16 output must be contiguous and shape-matched");
  launch_variant<256, 8>(a, b, scratch);
  launch_cast_to_bf16(scratch, output);
}

void launch_h2_bf16_direct(torch::Tensor a, torch::Tensor b,
                           torch::Tensor& output) {
  check_raw_inputs(a, b);
  check_bf16_output(a, b, output);
  check_gfx1201_device(a);
  launch_variant_bf16<256, 8>(a, b, output);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("raw_fp8_h1", &launch_h1, "R1 H1: 128x64 K64 four-wave raw FP8 GEMM");
  m.def("raw_fp8_h2", &launch_h2,
        "R1 H2: 256x64 K64 eight-wave raw FP8 GEMM -> FP32");
  m.def("raw_fp8_h1_bf16", &launch_h1_bf16,
        "R1 H1 raw FP8 GEMM plus BF16 output cast");
  m.def("raw_fp8_h2_bf16", &launch_h2_bf16,
        "R1 H2 raw FP8 GEMM plus BF16 output cast");
  m.def("raw_fp8_h2_bf16_direct", &launch_h2_bf16_direct,
        "R1 H2 raw FP8 GEMM with direct BF16 epilogue");
}
