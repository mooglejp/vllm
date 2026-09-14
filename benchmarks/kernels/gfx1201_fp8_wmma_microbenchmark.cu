// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Keep this source benchmark-only.  It deliberately omits MXFP4 decoding,
// E8M0 scales, and production registration so that it isolates the raw FP8
// WMMA mapping used by the P3 candidate.
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

constexpr int kTile = 16;
constexpr int kThreads = 32;

#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx1200__) || defined(__gfx1201__))

using Fp8 = rocwmma::float8_t;

__device__ __forceinline__ Fp8 fp8_from_bits(uint8_t bits) {
  Fp8 value;
  value.__x = bits;
  return value;
}

__global__ void fp8_wmma_gemm_kernel(const uint8_t* a, const uint8_t* b,
                                     float* output, int m, int n, int k) {
  const int lane = threadIdx.x;
  if (lane >= kThreads) {
    return;
  }

  const int tile_m = blockIdx.y * kTile;
  const int tile_n = blockIdx.x * kTile;

  __shared__ alignas(16) Fp8 tile_a[kTile * kTile];
  __shared__ alignas(16) Fp8 tile_b[kTile * kTile];
  __shared__ alignas(16) float tile_output[kTile * kTile];

  using FragA = rocwmma::fragment<rocwmma::matrix_a, kTile, kTile, kTile, Fp8,
                                  rocwmma::row_major>;
  using FragB = rocwmma::fragment<rocwmma::matrix_b, kTile, kTile, kTile, Fp8,
                                  rocwmma::row_major>;
  using FragC = rocwmma::fragment<rocwmma::accumulator, kTile, kTile, kTile,
                                  float, rocwmma::row_major>;

  FragC frag_c;
  rocwmma::fill_fragment(frag_c, 0.0f);

  for (int k_start = 0; k_start < k; k_start += kTile) {
    for (int index = lane; index < kTile * kTile; index += kThreads) {
      const int row = index / kTile;
      const int col = index % kTile;
      const int global_m = tile_m + row;
      const int global_n = tile_n + col;
      const int global_k_a = k_start + col;
      const int global_k_b = k_start + row;

      if (global_m < m && global_k_a < k) {
        tile_a[index] = fp8_from_bits(a[global_m * k + global_k_a]);
      } else {
        tile_a[index] = fp8_from_bits(0);
      }

      // B is stored as [N, K], while the matrix-B fragment consumes the
      // transposed [K, N] tile required by A[M, K] x B[N, K]^T.
      if (global_n < n && global_k_b < k) {
        tile_b[index] = fp8_from_bits(b[global_n * k + global_k_b]);
      } else {
        tile_b[index] = fp8_from_bits(0);
      }
    }
    __syncthreads();

    FragA frag_a;
    FragB frag_b;
    rocwmma::load_matrix_sync(frag_a, tile_a, kTile);
    rocwmma::load_matrix_sync(frag_b, tile_b, kTile);
    rocwmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    __syncthreads();
  }

  // Keep the accumulator in the WMMA fragment for the whole K loop.  The
  // single shared-memory store is only for tail-safe scalar output writes.
  rocwmma::store_matrix_sync(tile_output, frag_c, kTile);
  __syncthreads();
  for (int index = lane; index < kTile * kTile; index += kThreads) {
    const int row = index / kTile;
    const int col = index % kTile;
    const int global_m = tile_m + row;
    const int global_n = tile_n + col;
    if (global_m < m && global_n < n) {
      output[global_m * n + global_n] = tile_output[index];
    }
  }
}

// The v2a kernels keep the same raw-byte contract as the one-wave reference,
// but reuse one staged K tile across multiple output fragments and waves.  The
// template parameters are benchmark configurations, not production tuning
// constants.  Wave32 is intentional: the rejected P3 mapping and these
// candidates are compared with the same 32-thread rocWMMA wave contract.
template <int BlockM, int BlockN, int WaveGridM, int WaveGridN>
__global__ void fp8_wmma_large_kernel(const uint8_t* a, const uint8_t* b,
                                      float* output, int m, int n, int k) {
  static_assert(WaveGridM * WaveGridN == 2 || WaveGridM * WaveGridN == 4);
  static_assert(BlockM % (WaveGridM * kTile) == 0);
  static_assert(BlockN % (WaveGridN * kTile) == 0);

  constexpr int kWaves = WaveGridM * WaveGridN;
  constexpr int kWaveTileM = BlockM / WaveGridM;
  constexpr int kWaveTileN = BlockN / WaveGridN;
  constexpr int kFragM = kWaveTileM / kTile;
  constexpr int kFragN = kWaveTileN / kTile;
  constexpr int kFragCount = kFragM * kFragN;

  const int thread = threadIdx.x;
  const int wave = thread / kThreads;
  if (thread >= kWaves * kThreads) {
    return;
  }

  const int64_t block_m = static_cast<int64_t>(blockIdx.y) * BlockM;
  const int64_t block_n = static_cast<int64_t>(blockIdx.x) * BlockN;
  const int wave_m = wave / WaveGridN;
  const int wave_n = wave % WaveGridN;
  const int wave_m_start = wave_m * kWaveTileM;
  const int wave_n_start = wave_n * kWaveTileN;

  __shared__ alignas(16) Fp8 tile_a[BlockM * kTile];
  __shared__ alignas(16) Fp8 tile_b[kTile * BlockN];
  __shared__ alignas(16) float tile_output[BlockM * BlockN];

  using FragA = rocwmma::fragment<rocwmma::matrix_a, kTile, kTile, kTile, Fp8,
                                  rocwmma::row_major>;
  using FragB = rocwmma::fragment<rocwmma::matrix_b, kTile, kTile, kTile, Fp8,
                                  rocwmma::row_major>;
  using FragC = rocwmma::fragment<rocwmma::accumulator, kTile, kTile, kTile,
                                  float, rocwmma::row_major>;

  FragC fragments[kFragCount];
  for (int index = 0; index < kFragCount; ++index) {
    rocwmma::fill_fragment(fragments[index], 0.0f);
  }

  for (int k_start = 0; k_start < k; k_start += kTile) {
    for (int index = thread; index < BlockM * kTile;
         index += kWaves * kThreads) {
      const int row = index / kTile;
      const int col = index % kTile;
      const int64_t global_m = block_m + row;
      const int global_k = k_start + col;
      if (global_m < m && global_k < k) {
        tile_a[index] = fp8_from_bits(a[global_m * k + global_k]);
      } else {
        tile_a[index] = fp8_from_bits(0);
      }
    }
    for (int index = thread; index < kTile * BlockN;
         index += kWaves * kThreads) {
      const int row = index / BlockN;
      const int col = index % BlockN;
      const int global_k = k_start + row;
      const int64_t global_n = block_n + col;
      if (global_n < n && global_k < k) {
        // B is stored as [N, K]; tile_b is the transposed [K, N] view
        // consumed by the row-major matrix-B fragment.
        tile_b[index] = fp8_from_bits(b[global_n * k + global_k]);
      } else {
        tile_b[index] = fp8_from_bits(0);
      }
    }
    __syncthreads();

    FragA frag_a[kFragM];
    FragB frag_b[kFragN];
    for (int frag_m = 0; frag_m < kFragM; ++frag_m) {
      rocwmma::load_matrix_sync(
          frag_a[frag_m], tile_a + (wave_m_start + frag_m * kTile) * kTile,
          kTile);
    }
    for (int frag_n = 0; frag_n < kFragN; ++frag_n) {
      rocwmma::load_matrix_sync(frag_b[frag_n], tile_b + frag_n * kTile,
                                BlockN);
    }
    for (int frag_m = 0; frag_m < kFragM; ++frag_m) {
      for (int frag_n = 0; frag_n < kFragN; ++frag_n) {
        const int index = frag_m * kFragN + frag_n;
        rocwmma::mma_sync(fragments[index], frag_a[frag_m], frag_b[frag_n],
                          fragments[index]);
      }
    }
    __syncthreads();
  }

  for (int frag_m = 0; frag_m < kFragM; ++frag_m) {
    for (int frag_n = 0; frag_n < kFragN; ++frag_n) {
      const int index = frag_m * kFragN + frag_n;
      rocwmma::store_matrix_sync(tile_output +
                                     (wave_m_start + frag_m * kTile) * BlockN +
                                     wave_n_start + frag_n * kTile,
                                 fragments[index], BlockN);
    }
  }
  __syncthreads();

  for (int index = thread; index < BlockM * BlockN;
       index += kWaves * kThreads) {
    const int row = index / BlockN;
    const int col = index % BlockN;
    const int64_t global_m = block_m + row;
    const int64_t global_n = block_n + col;
    if (global_m < m && global_n < n) {
      output[global_m * n + global_n] = tile_output[index];
    }
  }
}

#else

// Keep host-only compilation and non-gfx1201 fatbin variants linkable.  The
// host entry point rejects those devices before this stub can be launched.
__global__ void fp8_wmma_gemm_kernel(const uint8_t*, const uint8_t*, float*,
                                     int, int, int) {}

template <int BlockM, int BlockN, int WaveGridM, int WaveGridN>
__global__ void fp8_wmma_large_kernel(const uint8_t*, const uint8_t*, float*,
                                      int, int, int) {}

#endif  // __HIP_DEVICE_COMPILE__ && gfx1200/gfx1201

void check_inputs(const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& output) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && output.is_cuda(),
              "gfx1201 FP8 WMMA tensors must be CUDA/HIP tensors");
  TORCH_CHECK(a.device() == b.device() && a.device() == output.device(),
              "gfx1201 FP8 WMMA tensors must be on the same device");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && output.dim() == 2,
              "FP8 WMMA tensors must be 2D");
  TORCH_CHECK(
      a.scalar_type() == torch::kByte && b.scalar_type() == torch::kByte,
      "FP8 WMMA inputs must contain raw FP8 bytes in uint8 tensors");
  TORCH_CHECK(output.scalar_type() == torch::kFloat,
              "FP8 WMMA output must be float32");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && output.is_contiguous(),
              "gfx1201 FP8 WMMA tensors must be contiguous");
  TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && b.size(0) > 0,
              "FP8 WMMA dimensions must be positive");
  TORCH_CHECK(a.size(1) == b.size(1),
              "FP8 WMMA inputs must have matching K dimensions");
  TORCH_CHECK(output.size(0) == a.size(0) && output.size(1) == b.size(0),
              "FP8 WMMA output shape must be [M, N]");
}

void check_gfx1201_device(const torch::Tensor& tensor) {
  const auto* properties = at::cuda::getDeviceProperties(tensor.get_device());
  TORCH_CHECK(std::string(properties->gcnArchName) == "gfx1201",
              "FP8 WMMA microbenchmark requires a gfx1201 device");
}

}  // namespace

void fp8_wmma_gemm(torch::Tensor a, torch::Tensor b, torch::Tensor& output) {
  check_inputs(a, b, output);
  check_gfx1201_device(a);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((static_cast<unsigned int>(b.size(0)) + kTile - 1) / kTile,
            (static_cast<unsigned int>(a.size(0)) + kTile - 1) / kTile);
  dim3 block(kThreads);
  hipLaunchKernelGGL(fp8_wmma_gemm_kernel, grid, block, 0, stream,
                     a.data_ptr<uint8_t>(), b.data_ptr<uint8_t>(),
                     output.data_ptr<float>(), static_cast<int>(a.size(0)),
                     static_cast<int>(b.size(0)), static_cast<int>(a.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int BlockM, int BlockN, int WaveGridM, int WaveGridN>
void launch_large(torch::Tensor a, torch::Tensor b, torch::Tensor& output) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((static_cast<unsigned int>(b.size(0)) + BlockN - 1) / BlockN,
            (static_cast<unsigned int>(a.size(0)) + BlockM - 1) / BlockM);
  dim3 block(WaveGridM * WaveGridN * kThreads);
  hipLaunchKernelGGL(
      (fp8_wmma_large_kernel<BlockM, BlockN, WaveGridM, WaveGridN>), grid,
      block, 0, stream, a.data_ptr<uint8_t>(), b.data_ptr<uint8_t>(),
      output.data_ptr<float>(), static_cast<int>(a.size(0)),
      static_cast<int>(b.size(0)), static_cast<int>(a.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_wmma_gemm_2wave_64x64(torch::Tensor a, torch::Tensor b,
                               torch::Tensor& output) {
  check_inputs(a, b, output);
  check_gfx1201_device(a);
  launch_large<64, 64, 2, 1>(a, b, output);
}

void fp8_wmma_gemm_4wave_64x64(torch::Tensor a, torch::Tensor b,
                               torch::Tensor& output) {
  check_inputs(a, b, output);
  check_gfx1201_device(a);
  launch_large<64, 64, 2, 2>(a, b, output);
}

void fp8_wmma_gemm_4wave_64x128(torch::Tensor a, torch::Tensor b,
                                torch::Tensor& output) {
  check_inputs(a, b, output);
  check_gfx1201_device(a);
  launch_large<64, 128, 2, 2>(a, b, output);
}

void fp8_wmma_gemm_4wave_128x64(torch::Tensor a, torch::Tensor b,
                                torch::Tensor& output) {
  check_inputs(a, b, output);
  check_gfx1201_device(a);
  launch_large<128, 64, 2, 2>(a, b, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_wmma_gemm", &fp8_wmma_gemm,
        "Raw FP8 x FP8 -> FP32 gfx1201 WMMA diagnostic GEMM");
  m.def("fp8_wmma_gemm_2wave_64x64", &fp8_wmma_gemm_2wave_64x64,
        "Benchmark-only 2-wave 64x64 raw FP8 WMMA GEMM");
  m.def("fp8_wmma_gemm_4wave_64x64", &fp8_wmma_gemm_4wave_64x64,
        "Benchmark-only 4-wave 64x64 raw FP8 WMMA GEMM");
  m.def("fp8_wmma_gemm_4wave_64x128", &fp8_wmma_gemm_4wave_64x128,
        "Benchmark-only 4-wave 64x128 raw FP8 WMMA GEMM");
  m.def("fp8_wmma_gemm_4wave_128x64", &fp8_wmma_gemm_4wave_128x64,
        "Benchmark-only 4-wave 128x64 raw FP8 WMMA GEMM");
}
