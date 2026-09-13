// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// PyTorch's HIP extension flags disable half conversions globally.  rocWMMA
// registers half vector helpers even for FP8 fragments, so restore the HIP
// half constructors before any PyTorch header includes HIP half types.
#ifdef __HIP_NO_HALF_OPERATORS__
  #undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
  #undef __HIP_NO_HALF_CONVERSIONS__
#endif

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPException.h>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdint>
#include <string>

#include "rocm/ops.h"

namespace {

constexpr int kTile = 16;
constexpr int kThreads = 32;
constexpr int kGroupSize = 32;
constexpr float kFp8Max = 448.0f;

#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx1200__) || defined(__gfx1201__))

using Fp8 = rocwmma::float8_t;

__device__ __forceinline__ Fp8 fp8_from_bits(uint8_t bits) {
  Fp8 value;
  value.__x = bits;
  return value;
}

__device__ __forceinline__ float e2m1_value(uint8_t code) {
  constexpr float kMagnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                   2.0f, 3.0f, 4.0f, 6.0f};
  float value = kMagnitude[code & 0x7];
  return (code & 0x8) != 0 ? -value : value;
}

__device__ __forceinline__ float e8m0_value(uint8_t raw) {
  if (raw == 0) {
    // E8M0 exponent zero represents 2^-127, which is a float subnormal.
    return __int_as_float(0x00400000);
  }
  if (raw == 255) {
    return __int_as_float(0x7fc00000);
  }
  return __int_as_float(static_cast<int>(raw) << 23);
}

__device__ __forceinline__ float reduce_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down(value, offset, 32);
    value = (isnan(value) || isnan(other)) ? NAN : fmaxf(value, other);
  }
  return value;
}

__global__ void quantize_fp8_rows_kernel(const __hip_bfloat16* x,
                                         uint8_t* quantized, float* row_scale,
                                         int m, int k) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  if (row >= m) {
    return;
  }

  float local_amax = 0.0f;
  for (int col = lane; col < k; col += blockDim.x) {
    const float value = static_cast<float>(x[row * k + col]);
    if (isnan(value)) {
      local_amax = NAN;
    } else if (!isnan(local_amax)) {
      local_amax = fmaxf(local_amax, fabsf(value));
    }
  }

  local_amax = reduce_max(local_amax);
  __shared__ float warp_amax[kThreads / 32];
  if ((lane & 31) == 0) {
    warp_amax[lane >> 5] = local_amax;
  }
  __syncthreads();

  float amax = 0.0f;
  if (lane == 0) {
    for (int warp = 0; warp < kThreads / 32; ++warp) {
      const float value = warp_amax[warp];
      amax = (isnan(amax) || isnan(value)) ? NAN : fmaxf(amax, value);
    }
    row_scale[row] = amax == 0.0f ? 1.0f : amax / kFp8Max;
  }
  __syncthreads();

  const float scale = row_scale[row];
  for (int col = lane; col < k; col += blockDim.x) {
    float normalized = static_cast<float>(x[row * k + col]) / scale;
    normalized = fminf(fmaxf(normalized, -kFp8Max), kFp8Max);
    const Fp8 value(normalized);
    quantized[row * k + col] = value.__x;
  }
}

__global__ void w4a8_wmma_kernel(const uint8_t* quantized,
                                 const float* row_scale,
                                 const uint8_t* packed_weight,
                                 const uint8_t* weight_scale,
                                 __hip_bfloat16* output, int m, int n, int k) {
  const int lane = threadIdx.x;
  if (lane >= kThreads) {
    return;
  }

  const int tile_m = blockIdx.y * kTile;
  const int tile_n = blockIdx.x * kTile;
  const int packed_k = k / 2;
  const int scale_k = k / kGroupSize;

  __shared__ alignas(16) Fp8 tile_a[kTile * kTile];
  __shared__ alignas(16) Fp8 tile_b[kTile * kTile];
  __shared__ alignas(16) float tile_tmp[kTile * kTile];
  __shared__ alignas(16) float tile_acc[kTile * kTile];

  for (int index = lane; index < kTile * kTile; index += kThreads) {
    tile_acc[index] = 0.0f;
  }
  __syncthreads();

  using FragA = rocwmma::fragment<rocwmma::matrix_a, kTile, kTile, kTile, Fp8,
                                  rocwmma::row_major>;
  using FragB = rocwmma::fragment<rocwmma::matrix_b, kTile, kTile, kTile, Fp8,
                                  rocwmma::row_major>;
  using FragC = rocwmma::fragment<rocwmma::accumulator, kTile, kTile, kTile,
                                  float, rocwmma::row_major>;

  for (int k_start = 0; k_start < k; k_start += kTile) {
    for (int index = lane; index < kTile * kTile; index += kThreads) {
      const int row = index / kTile;
      const int col = index % kTile;
      const int global_m = tile_m + row;
      const int global_n = tile_n + col;
      const int global_k_a = k_start + col;
      const int global_k_b = k_start + row;

      if (global_m < m && global_k_a < k) {
        tile_a[index] = fp8_from_bits(quantized[global_m * k + global_k_a]);
      } else {
        tile_a[index] = fp8_from_bits(0);
      }

      if (global_n < n && global_k_b < k) {
        const uint8_t packed =
            packed_weight[global_n * packed_k + (global_k_b >> 1)];
        const uint8_t code = (global_k_b & 1) == 0 ? packed & 0xf : packed >> 4;
        tile_b[index] = Fp8(e2m1_value(code));
      } else {
        tile_b[index] = fp8_from_bits(0);
      }
    }
    __syncthreads();

    FragA frag_a;
    FragB frag_b;
    FragC frag_c;
    rocwmma::load_matrix_sync(frag_a, tile_a, kTile);
    rocwmma::load_matrix_sync(frag_b, tile_b, kTile);
    rocwmma::fill_fragment(frag_c, 0.0f);
    rocwmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    rocwmma::store_matrix_sync(tile_tmp, frag_c, kTile);
    __syncthreads();

    const int group = k_start / kGroupSize;
    for (int index = lane; index < kTile * kTile; index += kThreads) {
      const int col = index % kTile;
      const int global_n = tile_n + col;
      const float scale =
          global_n < n ? e8m0_value(weight_scale[global_n * scale_k + group])
                       : 0.0f;
      tile_acc[index] += tile_tmp[index] * scale;
    }
    __syncthreads();
  }

  for (int index = lane; index < kTile * kTile; index += kThreads) {
    const int row = index / kTile;
    const int col = index % kTile;
    const int global_m = tile_m + row;
    const int global_n = tile_n + col;
    if (global_m < m && global_n < n) {
      const float value = tile_acc[index] * row_scale[global_m];
      output[global_m * n + global_n] = __hip_bfloat16(value);
    }
  }
}

#else

// Keep host compilation and non-gfx1201 fatbin variants linkable. The host
// entry points reject those devices before these stubs can be launched.
__global__ void quantize_fp8_rows_kernel(const __hip_bfloat16*, uint8_t*,
                                         float*, int, int) {}
__global__ void w4a8_wmma_kernel(const uint8_t*, const float*, const uint8_t*,
                                 const uint8_t*, __hip_bfloat16*, int, int,
                                 int) {}

#endif  // __HIP_DEVICE_COMPILE__ && gfx1201

void check_quantize_inputs(const torch::Tensor& x,
                           const torch::Tensor& quantized,
                           const torch::Tensor& row_scale) {
  TORCH_CHECK(x.is_cuda() && quantized.is_cuda() && row_scale.is_cuda(),
              "gfx1201 W4A8 tensors must be CUDA/HIP tensors");
  TORCH_CHECK(
      x.device() == quantized.device() && x.device() == row_scale.device(),
      "gfx1201 W4A8 tensors must be on the same device");
  TORCH_CHECK(x.dim() == 2, "x must be a 2D [M, K] tensor");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16,
              "x must have bfloat16 dtype");
  TORCH_CHECK(quantized.scalar_type() == torch::kByte,
              "quantized activations must have uint8 dtype");
  TORCH_CHECK(row_scale.scalar_type() == torch::kFloat,
              "row scales must have float32 dtype");
  TORCH_CHECK(x.is_contiguous() && quantized.is_contiguous() &&
                  row_scale.is_contiguous(),
              "gfx1201 W4A8 tensors must be contiguous");
  TORCH_CHECK(quantized.sizes() == x.sizes(),
              "quantized activations must have the same shape as x");
  TORCH_CHECK(row_scale.dim() == 1 && row_scale.size(0) == x.size(0),
              "row scales must have shape [M]");
  TORCH_CHECK(x.size(0) >= 128,
              "gfx1201 W4A8 prefill is limited to large-M inputs (M >= 128)");
  TORCH_CHECK(x.size(1) > 0 && x.size(1) % kGroupSize == 0,
              "K must be positive and divisible by MXFP4 group size 32");
}

void check_gemm_inputs(const torch::Tensor& quantized,
                       const torch::Tensor& row_scale,
                       const torch::Tensor& packed_weight,
                       const torch::Tensor& weight_scale,
                       const torch::Tensor& output) {
  TORCH_CHECK(quantized.is_cuda() && row_scale.is_cuda() &&
                  packed_weight.is_cuda() && weight_scale.is_cuda() &&
                  output.is_cuda(),
              "gfx1201 W4A8 tensors must be CUDA/HIP tensors");
  TORCH_CHECK(quantized.device() == row_scale.device() &&
                  quantized.device() == packed_weight.device() &&
                  quantized.device() == weight_scale.device() &&
                  quantized.device() == output.device(),
              "gfx1201 W4A8 tensors must be on the same device");
  TORCH_CHECK(quantized.dim() == 2 && row_scale.dim() == 1 &&
                  packed_weight.dim() == 2 && weight_scale.dim() == 2 &&
                  output.dim() == 2,
              "invalid gfx1201 W4A8 tensor ranks");
  TORCH_CHECK(quantized.scalar_type() == torch::kByte,
              "quantized activations must have uint8 dtype");
  TORCH_CHECK(row_scale.scalar_type() == torch::kFloat,
              "row scales must have float32 dtype");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kByte &&
                  weight_scale.scalar_type() == torch::kByte,
              "MXFP4 weights and scales must have uint8 dtype");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
              "output must have bfloat16 dtype");
  TORCH_CHECK(quantized.is_contiguous() && row_scale.is_contiguous() &&
                  packed_weight.is_contiguous() &&
                  weight_scale.is_contiguous() && output.is_contiguous(),
              "gfx1201 W4A8 tensors must be contiguous");

  const auto m = quantized.size(0);
  const auto k = quantized.size(1);
  const auto n = packed_weight.size(0);
  TORCH_CHECK(m >= 128, "gfx1201 W4A8 prefill requires M >= 128");
  TORCH_CHECK(n > 0, "gfx1201 W4A8 prefill requires N > 0");
  TORCH_CHECK(k > 0 && k % kGroupSize == 0,
              "K must be positive and divisible by MXFP4 group size 32");
  TORCH_CHECK(packed_weight.size(1) * 2 == k,
              "packed weights must have shape [N, K / 2]");
  TORCH_CHECK(
      weight_scale.size(0) == n && weight_scale.size(1) == k / kGroupSize,
      "weight scales must have shape [N, K / 32]");
  TORCH_CHECK(
      row_scale.size(0) == m && output.size(0) == m && output.size(1) == n,
      "row scales and output shape do not match activations/weights");
}

void check_gfx1201_device(const torch::Tensor& tensor) {
  TORCH_CHECK(tensor.device().is_cuda(),
              "gfx1201 W4A8 prefill tensors must be CUDA/HIP tensors");
  const auto* properties = at::cuda::getDeviceProperties(tensor.get_device());
  TORCH_CHECK(std::string(properties->gcnArchName) == "gfx1201",
              "gfx1201 W4A8 prefill requires a gfx1201 device");
}

}  // namespace

void gfx1201_w4a8_quantize(torch::Tensor x, torch::Tensor& quantized,
                           torch::Tensor& row_scale) {
  check_quantize_inputs(x, quantized, row_scale);
  check_gfx1201_device(x);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned int>(x.size(0)));
  dim3 block(kThreads);
  hipLaunchKernelGGL(quantize_fp8_rows_kernel, grid, block, 0, stream,
                     reinterpret_cast<const __hip_bfloat16*>(x.data_ptr()),
                     quantized.data_ptr<uint8_t>(), row_scale.data_ptr<float>(),
                     static_cast<int>(x.size(0)), static_cast<int>(x.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gfx1201_w4a8_gemm(torch::Tensor quantized, torch::Tensor row_scale,
                       torch::Tensor packed_weight, torch::Tensor weight_scale,
                       torch::Tensor& output) {
  check_gemm_inputs(quantized, row_scale, packed_weight, weight_scale, output);
  check_gfx1201_device(quantized);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(quantized));
  const auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(
      (static_cast<unsigned int>(packed_weight.size(0)) + kTile - 1) / kTile,
      (static_cast<unsigned int>(quantized.size(0)) + kTile - 1) / kTile);
  dim3 block(kThreads);
  hipLaunchKernelGGL(w4a8_wmma_kernel, grid, block, 0, stream,
                     quantized.data_ptr<uint8_t>(), row_scale.data_ptr<float>(),
                     packed_weight.data_ptr<uint8_t>(),
                     weight_scale.data_ptr<uint8_t>(),
                     reinterpret_cast<__hip_bfloat16*>(output.data_ptr()),
                     static_cast<int>(quantized.size(0)),
                     static_cast<int>(packed_weight.size(0)),
                     static_cast<int>(quantized.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
