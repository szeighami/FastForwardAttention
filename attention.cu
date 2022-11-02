
#include "decoder_masked_multihead_attention.h"
#include "decoder_masked_multihead_attention_utils.h"
//#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include <sys/time.h> // for clock_gettime()
#include <vector> // for clock_gettime()

#include <assert.h>
#include <float.h>
#include <type_traits>
#include <sstream>
#include <fstream>
#include <cmath>
#include <limits>



#define tpb64
#define tpb128
#define tpb256
#define tpb512
#define tpk1
#define tpk2
#define tpk4
#define tpv1
#define tpv2
#define tpv4
#define tpv8


float get_alph_c(int batch_size, int tpb, int tpv, int tpk, int prompt_len, int head_size)
{
	std::vector<float> features  = {(float)batch_size, (float)tpb, (float)tpv, (float)tpk, (float)prompt_len, (float)head_size, 1.0f/(float)batch_size, 1.0f/(float)tpb, 1.0f/(float)tpv, 1.0f/(float)tpk, 1.0f/(float)prompt_len, 1.0f/(float)head_size};

	//parameters equation found by fitting polynomial
	std::vector<float> coefs = {1.98477462e+00, 1.40485848e+00, 9.44788188e+01, 4.97399678e-01,
							  1.38083739e+01, 1.04151458e+00, 7.80703368e+01, 2.47025323e+03,
							  7.96088089e-01, 1.17568866e+02, 3.81078416e-02, 5.98652327e-02,
							  1.73769284e-04, 1.89844292e-02, 2.43233896e-01, 1.84552851e+01,
							  9.63335902e-03, 3.43894758e-01, 2.35635243e+02, 5.69168616e+02,
							  6.42865539e-01, 2.40778083e-02};

	std::vector<std::string> terms = {"6", "8", "11", "1 10", "3 7", "3 11", "6 11", "7 11", "8 9", "8 11", "0 1 10", "1 2 10", "1 3 6", "1 6 11", "1 8 11", "1 10 11", "2 3 6", "2 3 11", "2 10 11", "3 7 11", "5 7 8", "5 8 9"};

	float alpha_c = 0;
	for (int i = 0; i < coefs.size(); i++){
		size_t pos = 0;
		std::string token;
		std::string s = terms[i];
		float prod = 1;	
		while ((pos = s.find(" ")) != std::string::npos) {
			token = s.substr(0, pos);
			prod = prod*features[std::stoi(token)];
			s.erase(0, pos + 1);
		}
		alpha_c += prod*coefs[i];
	}
    
	return alpha_c;
}

int get_max_warps(int MyRegCount, int tpb, int mySharedMemAllocationSize, int max_sharedmemory_per_block, int run_time_smem, int head_size, int tpv, int precision, int prompt_len)
{
    int MyWarpsPerBlock = tpb/32;
    int limitRegsPerBlock = 65536;
    int limitThreadsPerWarp = 32;
    int myAllocationSize = 256 ;
    int myWarpAllocationGranularity = 4;

    int smem_per_block = std::max((tpb/(head_size/tpv))*head_size*(precision/8)/2.0, std::ceil((prompt_len+1.0)/ 4.0) * 16+std::ceil((prompt_len+1.0)/ 4.0) * 4.0 * (precision/8))+run_time_smem;

    int reg_limit_sm = std::floor((limitRegsPerBlock/(std::ceil(MyRegCount*limitThreadsPerWarp/(float)myAllocationSize)*myAllocationSize))/(float)myWarpAllocationGranularity)*myWarpAllocationGranularity;
    int max_blocks_per_sm = reg_limit_sm/MyWarpsPerBlock;
    int registers_max_warps = max_blocks_per_sm*MyWarpsPerBlock;

    int smem_used = std::ceil(smem_per_block/(float)mySharedMemAllocationSize)*mySharedMemAllocationSize;
    int smem_max_warps = (max_sharedmemory_per_block/smem_used)*MyWarpsPerBlock;

    int max_warps_per_sm = std::min(registers_max_warps, smem_max_warps);

    return max_warps_per_sm;
}

int get_k_size(int tpk)
{

    if (tpk == 1)
        return 8;
    if (tpk == 2)
        return 4;
    if (tpk == 4)
        return 2;
    printf("tpk not found");
    return -1;
}

float get_time_for_param(int precision, int batch_size, int head_num, int head_size, int prompt_len, int no_sms, int tpb, int tpv, int tpk, float C, float gpu_transfer_rate, float gpu_clock_rate, int mySharedMemAllocationSize, int max_sharedmemory_per_block, int run_time_smem)
{
	float mem_trasfer_bytes_calc = (precision/8) * batch_size * 4 * head_num*head_size+2*(precision/8)*batch_size*head_num*head_size*(prompt_len);

	int no_blocks = batch_size*head_num;
	int max_blocks_per_smp = std::ceil(no_blocks/(float)no_sms);
	int max_launched_warps = max_blocks_per_smp*(tpb/32);

	int MyRegCount_est;

	//parameters equation found by fitting polynomial
	if (precision == 16)
		MyRegCount_est = 93.61111111 + 0.44444444*head_size -18.61111111*tpk;
	else
		MyRegCount_est = 135.83333333 +0.96527778*head_size -38.54761905*tpk;

	int max_warps_per_sm_est = get_max_warps(MyRegCount_est, tpb, mySharedMemAllocationSize, max_sharedmemory_per_block, run_time_smem,  head_size, tpv, precision, prompt_len);

	float overload_factor_est = 1;
	if (max_launched_warps>max_warps_per_sm_est)
		overload_factor_est = max_launched_warps/max_warps_per_sm_est;


	float alpha_m = 1;	
	float alpha_c = get_alph_c(batch_size, tpb, tpv, tpk, prompt_len, head_size);
   
	float time_mem_est = alpha_m*mem_trasfer_bytes_calc/(gpu_transfer_rate);

	int actual_tpv = head_size/tpv;
	//parameters equation found by fitting polynomial
    float no_inst_est;
	if (precision == 16)
		no_inst_est = 399.87009804 + 13.44339289*prompt_len*actual_tpv/tpb + -158.63911938*prompt_len*tpk/tpb +1.19653041*prompt_len*head_size/tpb+30.93940003*prompt_len*head_size/(tpb*get_k_size(tpk))+7.25426324*prompt_len/tpb;
	else
		no_inst_est = 450.37282135l + 1.30064347*prompt_len*actual_tpv/tpb + -64.04635652*prompt_len*tpk/tpb +2.34498193*prompt_len*head_size/tpb+49.51406228*prompt_len*head_size/(tpb*get_k_size(tpk))+60.20351018*prompt_len/tpb;

	float est_time = time_mem_est+(alpha_c*no_inst_est)/(gpu_clock_rate*std::pow(10,9))*overload_factor_est+C;

	return est_time;
}


void get_launch_params(int precision, int batch_size, int head_num, int head_size, int prompt_len, int no_sms, float C, float gpu_transfer_rate, float gpu_clock_rate, int mySharedMemAllocationSize, int max_sharedmemory_per_block, int run_time_smem, int& tpb, int& tpv, int& tpk)
{
	std::vector<int> tpks = {1, 2, 4};
	std::vector<int> tpbs = {64, 128, 256};
	std::vector<int> tpvs = {2, 4};
	if (precision == 16)
        tpvs.push_back(8);
    else
        tpvs.push_back(1);

    float min_time = std::numeric_limits<float>::max();
    for (auto curr_tpv: tpvs){
        for (auto curr_tpk: tpks){
            for (auto curr_tpb: tpbs){
                float time = get_time_for_param(precision, batch_size, head_num, head_size, prompt_len, no_sms, curr_tpb, curr_tpv, curr_tpk, C, gpu_transfer_rate, gpu_clock_rate, mySharedMemAllocationSize, max_sharedmemory_per_block, run_time_smem);
                if (time < min_time){
                    time =  min_time;
                    tpb = curr_tpb;
                    tpk = curr_tpk;
                    tpv = curr_tpv;
                }
            }
        }
    }
}


float half_to_float(uint16_t float16_value)
{
  // MSB -> LSB
  // float16=1bit: sign, 5bit: exponent, 10bit: fraction
  // float32=1bit: sign, 8bit: exponent, 23bit: fraction
  // for normal exponent(1 to 0x1e): value=2**(exponent-15)*(1.fraction)
  // for denormalized exponent(0): value=2**-14*(0.fraction)
  uint32_t sign = float16_value >> 15;
  uint32_t exponent = (float16_value >> 10) & 0x1F;
  uint32_t fraction = (float16_value & 0x3FF);
  uint32_t float32_value;
  if (exponent == 0)
  {
    if (fraction == 0)
    {
      // zero
      float32_value = (sign << 31);
    }
    else
    {
      // can be represented as ordinary value in float32
      // 2 ** -14 * 0.0101
      // => 2 ** -16 * 1.0100
      // int int_exponent = -14;
      exponent = 127 - 14;
      while ((fraction & (1 << 10)) == 0)
      {
        //int_exponent--;
        exponent--;
        fraction <<= 1;
      }
      fraction &= 0x3FF;
      // int_exponent += 127;
      float32_value = (sign << 31) | (exponent << 23) | (fraction << 13);  
    }    
  }
  else if (exponent == 0x1F)
  {
    /* Inf or NaN */
    float32_value = (sign << 31) | (0xFF << 23) | (fraction << 13);
  }
  else
  {
    /* ordinary number */
    float32_value = (sign << 31) | ((exponent + (127-15)) << 23) | (fraction << 13);
  }
  
  return *((float*)&float32_value);
}


//#define MMHA_USE_FP32_ACUM_FOR_OUT

namespace mmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.
//
// The different kernels assign a threadblock for B x H pair. The grid has size (1, B, H). We use
// 64, 128 and 256 threads per block.
//
// Each threadblock loads Dh values from Q and its associated bias. The kernels run a loop to
// compute Q * K^T where K is loaded from a cache buffer -- except for the current timestep. The
// cache buffer helps with memory accesses and contains keys with bias.
//
// The layout of the cache buffer for the keys is [B, H, Dh/x, L, x] where x == 8 for FP16 and
// x == 4 for FP32 where the fastest moving dimension (contiguous data) is the rightmost one. The
// values for x are chosen to create chunks of 16 bytes.
//
// The different kernels use 1, 2 or 4 threads per key (THREADS_PER_KEY). The size of the LDGs
// depends on the number of threads per key. Each thread sums Dh / THREADS_PER_KEY elements. At
// the end of each iteration of the Q * K^T loop, we perform a reduction between lanes using an
// HMMA instruction (Tensor Core). Each Q * K^T valuey is stored in shared memory in FP32.
//
// After that loop, a parallel softmax is computed accross the different Q * K^T values stored in
// shared memory.
//
// The kernel ends with a loop over the values in V. We use THREADS_PER_VALUE to control how many
// timesteps are computed by loop iteration. As with the keys, the values are read from a cache
// except for the current timestep. The layout of the cache buffer for the values is much simpler
// as it is [B, H, L, Dh].
//

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Dh>
struct Qk_vec_ {};

template<>
struct Qk_vec_<float, 32> {
    using Type = float;
};
template<>
struct Qk_vec_<float, 64> {
    using Type = float2;
};
template<>
struct Qk_vec_<float, 128> {
    using Type = float4;
};
template<>
struct Qk_vec_<float, 256> {
    using Type = float4;
};
template<>
struct Qk_vec_<uint16_t, 32> {
    using Type = uint32_t;
};
template<>
struct Qk_vec_<uint16_t, 64> {
    using Type = uint32_t;
};
template<>
struct Qk_vec_<uint16_t, 128> {
    using Type = uint2;
};
template<>
struct Qk_vec_<uint16_t, 256> {
    using Type = uint4;
};
#ifdef ENABLE_BF16
template<>
struct Qk_vec_<__nv_bfloat16, 32> {
    using Type = __nv_bfloat162;
};
template<>
struct Qk_vec_<__nv_bfloat16, 64> {
    using Type = __nv_bfloat162;
};
template<>
struct Qk_vec_<__nv_bfloat16, 128> {
    using Type = bf16_4_t;
};
template<>
struct Qk_vec_<__nv_bfloat16, 256> {
    using Type = bf16_8_t;
};
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int THREADS_PER_KEY>
struct K_vec_ {};

template<>
struct K_vec_<float, 4> {
    using Type = float;
};
template<>
struct K_vec_<float, 2> {
    using Type = float2;
};
template<>
struct K_vec_<float, 1> {
    using Type = float4;
};
template<>
struct K_vec_<uint16_t, 4> {
    using Type = uint32_t;
};
template<>
struct K_vec_<uint16_t, 2> {
    using Type = uint2;
};
template<>
struct K_vec_<uint16_t, 1> {
    using Type = uint4;
};
#ifdef ENABLE_BF16
template<>
struct K_vec_<__nv_bfloat16, 4> {
    using Type = __nv_bfloat162;
};
template<>
struct K_vec_<__nv_bfloat16, 2> {
    using Type = bf16_4_t;
};
template<>
struct K_vec_<__nv_bfloat16, 1> {
    using Type = bf16_8_t;
};
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int V_VEC_SIZE>
struct V_vec_ {};

template<>
struct V_vec_<float, 1> {
    using Type = float;
};
template<>
struct V_vec_<float, 2> {
    using Type = float2;
};
template<>
struct V_vec_<float, 4> {
    using Type = float4;
};
template<>
struct V_vec_<uint16_t, 2> {
    using Type = uint32_t;
};
template<>
struct V_vec_<uint16_t, 4> {
    using Type = uint2;
};
template<>
struct V_vec_<uint16_t, 8> {
    using Type = uint4;
};
#ifdef ENABLE_BF16
template<>
struct V_vec_<__nv_bfloat16, 2> {
    using Type = __nv_bfloat162;
};
template<>
struct V_vec_<__nv_bfloat16, 4> {
    using Type = bf16_4_t;
};
template<>
struct V_vec_<__nv_bfloat16, 8> {
    using Type = bf16_8_t;
};
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
template<typename T>
struct Qk_vec_acum_fp32_ {};

template<>
struct Qk_vec_acum_fp32_<float> {
    using Type = float;
};
template<>
struct Qk_vec_acum_fp32_<float2> {
    using Type = float2;
};
template<>
struct Qk_vec_acum_fp32_<float4> {
    using Type = float4;
};
// template<> struct Qk_vec_acum_fp32_<uint16_t> { using Type = float;        };
template<>
struct Qk_vec_acum_fp32_<uint32_t> {
    using Type = float2;
};
template<>
struct Qk_vec_acum_fp32_<uint2> {
    using Type = Float4_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct K_vec_acum_fp32_ {};

template<>
struct K_vec_acum_fp32_<float> {
    using Type = float;
};
template<>
struct K_vec_acum_fp32_<float2> {
    using Type = float2;
};
template<>
struct K_vec_acum_fp32_<float4> {
    using Type = float4;
};
template<>
struct K_vec_acum_fp32_<uint32_t> {
    using Type = float2;
};
template<>
struct K_vec_acum_fp32_<uint2> {
    using Type = Float4_;
};
template<>
struct K_vec_acum_fp32_<uint4> {
    using Type = Float8_;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
template<typename T>
struct V_vec_acum_fp32_ {};

template<>
struct V_vec_acum_fp32_<float> {
    using Type = float;
};
template<>
struct V_vec_acum_fp32_<float2> {
    using Type = float2;
};
template<>
struct V_vec_acum_fp32_<float4> {
    using Type = float4;
};
template<>
struct V_vec_acum_fp32_<uint32_t> {
    using Type = float2;
};
template<>
struct V_vec_acum_fp32_<uint2> {
    using Type = Float4_;
};
template<>
struct V_vec_acum_fp32_<uint4> {
    using Type = Float8_;
};
#ifdef ENABLE_BF16
template<>
struct V_vec_acum_fp32_<__nv_bfloat162> {
    using Type = float2;
};
template<>
struct V_vec_acum_fp32_<bf16_4_t> {
    using Type = Float4_;
};
template<>
struct V_vec_acum_fp32_<bf16_8_t> {
    using Type = Float8_;
};
#endif  // ENABLE_BF16
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS_PER_KEY, typename K_vec, int N>
inline __device__ float qk_dot_(const K_vec (&q)[N], const K_vec (&k)[N])
{
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    using K_vec_acum = typename K_vec_acum_fp32_<K_vec>::Type;
#else
    using K_vec_acum = K_vec;
#endif
    // Compute the parallel products for Q*K^T (treat vector lanes separately).
    K_vec_acum qk_vec = mul<K_vec_acum, K_vec, K_vec>(q[0], k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii) {
        qk_vec = fma(q[ii], k[ii], qk_vec);
    }

    // Finalize the reduction across lanes.
    float qk = sum(qk_vec);
#pragma unroll
    for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int THREADS_PER_KEY>
struct Qk_dot {
    template<typename K_vec, int N>
    static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N])
    {
        return qk_dot_<THREADS_PER_KEY>(q, k);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 hmma_fp32(const uint2& a, uint32_t b)
{
    float4 c;
    float zero = 0.f;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
                 "    {%0, %1, %2, %3}, \n"
                 "    {%4, %5}, \n"
                 "    {%6}, \n"
                 "    {%7, %7, %7, %7}; \n"

                 : "=f"(c.x), "=f"(c.y), "=f"(c.z), "=f"(c.w)
                 : "r"(a.x) "r"(a.y), "r"(b), "f"(zero));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int N>
inline __device__ float qk_hmma_dot_(const uint32_t (&q)[N], const uint32_t (&k)[N])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    using K_vec_acum = typename K_vec_acum_fp32_<uint32_t>::Type;
#else
    using K_vec_acum = uint32_t;
#endif
    K_vec_acum qk_vec = mul<K_vec_acum, uint32_t, uint32_t>(q[0], k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii) {
        qk_vec = fma(q[ii], k[ii], qk_vec);
    }
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    uint32_t qk_vec_ = float2_to_half2(qk_vec);
    return hmma_fp32(make_uint2(qk_vec_, 0u), 0x3c003c00u).x;
#else
    return hmma_fp32(make_uint2(qk_vec, 0u), 0x3c003c00u).x;
#endif
#else
    return 0.f;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Qk_dot<uint16_t, 4> {
    template<int N>
    static inline __device__ float dot(const uint32_t (&q)[N], const uint32_t (&k)[N])
    {
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA_FOR_REDUCTION)
        return qk_hmma_dot_(q, k);
#else
        return qk_dot_<4>(q, k);
#endif  // defined MMHA_USE_HMMA_FOR_REDUCTION
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int WARPS_PER_BLOCK, int WARP_SIZE = 32>
inline __device__ float block_sum(float* red_smem, float sum)
{

    // Decompose the thread index into warp / lane.
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

// Compute the sum per warp.
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
    }

    // Warp leaders store the data to shared memory.
    if (lane == 0) {
        red_smem[warp] = sum;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The warps compute the final sums.
    if (lane < WARPS_PER_BLOCK) {
        sum = red_smem[lane];
    }

// Parallel reduction inside the warp.
#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
    }

    // Broadcast to other threads.
    return __shfl_sync(uint32_t(-1), sum, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float& dst, float src)
{
    dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint16_t& dst, float src)
{
    dst = float_to_half(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint32_t& dst, float2 src)
{
    dst = float2_to_half2(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ENABLE_BF16
inline __device__ void convert_from_float(__nv_bfloat16& dst, float src)
{
    dst = __float2bfloat16(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(__nv_bfloat162& dst, float2 src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst = __float22bfloat162_rn(src);
#else
    dst = __floats2bfloat162_rn(src.x, src.y);
#endif
}
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint2& dst, Float4_ src)
{
    dst.x = float2_to_half2(src.x);
    dst.y = float2_to_half2(src.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint4& dst, Float8_ src)
{
    dst.x = float2_to_half2(src.x);
    dst.y = float2_to_half2(src.y);
    dst.z = float2_to_half2(src.z);
    dst.w = float2_to_half2(src.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
inline __device__ void convert_from_float(bf16_4_t& dst, Float4_ src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst.x = __float22bfloat162_rn(src.x);
    dst.y = __float22bfloat162_rn(src.y);
#else
    dst.x = __floats2bfloat162_rn(src.x.x, src.x.y);
    dst.y = __floats2bfloat162_rn(src.y.x, src.y.y);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(bf16_8_t& dst, Float8_ src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst.x = __float22bfloat162_rn(src.x);
    dst.y = __float22bfloat162_rn(src.y);
    dst.z = __float22bfloat162_rn(src.z);
    dst.w = __float22bfloat162_rn(src.w);
#else
    dst.x = __floats2bfloat162_rn(src.x.x, src.x.y);
    dst.y = __floats2bfloat162_rn(src.y.x, src.y.y);
    dst.z = __floats2bfloat162_rn(src.z.x, src.z.y);
    dst.w = __floats2bfloat162_rn(src.w.x, src.w.y);
#endif
}
#endif  // ENABLE_BF16

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float2& dst, float2 src)
{
    dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float4& dst, float4 src)
{
    dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float convert_to_float(float4 u)
{
    return u.x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float convert_to_float(uint4 u)
{
    float2 tmp = half2_to_float2(u.x);
    return tmp.x;
}

#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float cast_to_float(float u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(float2 u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 cast_to_float(float4 u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(Float4_ u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(Float8_ u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(uint32_t u)
{
    return half2_to_float2(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(uint2 u)
{
    Float4_ tmp;
    tmp.x = half2_to_float2(u.x);
    tmp.y = half2_to_float2(u.y);
    return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(uint4 u)
{
    Float8_ tmp;
    tmp.x = half2_to_float2(u.x);
    tmp.y = half2_to_float2(u.y);
    tmp.z = half2_to_float2(u.z);
    tmp.w = half2_to_float2(u.w);
    return tmp;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ __host__ T div_up(T m, T n)
{
    return (m + n - 1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, bool DO_CROSS_ATTENTION>
inline size_t smem_size_in_bytes(const Multihead_attention_params<T, DO_CROSS_ATTENTION>& params,
                                 int threads_per_value,
                                 int threads_per_block)
{
    // The amount of shared memory needed to store the Q*K^T values in float.
    // TODO
    size_t qk_sz = (DO_CROSS_ATTENTION) ? div_up(params.seq_length + 1, 4) * 16 : div_up(params.timestep + 1, 4) * 16;

    // The extra memory needed if we are not using floats for the final logits.
    size_t logits_sz = 0;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (sizeof(T) != 4) {
        // TDOD
        logits_sz = div_up(params.seq_length, 4) * 4 * sizeof(T);
    }
#endif

    // The total size needed during softmax.
    size_t softmax_sz = qk_sz + logits_sz;

    // The number of partial rows to reduce in the final reduction.
    int rows_per_red = threads_per_block / threads_per_value;
    // The amount of storage needed to finalize the outputs.
    size_t red_sz = rows_per_red * params.hidden_size_per_head * sizeof(T) / 2;

    // The max.
    return max(softmax_sz, red_sz);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ constexpr uint32_t shfl_mask(int threads)
{
    return threads == 32 ? uint32_t(-1) : (1u << threads) - 1u;
}


template<
    // The type of the inputs. Supported types: float and half.
    typename T,
    // The hidden dimension per head.
    int Dh,
    int Dh_MAX,
    // The number of threads per key.
    int THREADS_PER_KEY,
    // The number of threads per value.
    int THREADS_PER_VALUE,
    // The number of threads in a threadblock.
    int THREADS_PER_BLOCK,
    bool DO_CROSS_ATTENTION
    >
__global__ void masked_multihead_attention_kernel_optimized(Multihead_attention_params<T, DO_CROSS_ATTENTION> params)
{
    // Make sure the hidden dimension per head is a multiple of the number of threads per key.
    static_assert(Dh_MAX % THREADS_PER_KEY == 0, "");
    // Make sure the hidden dimension per head is a multiple of the number of threads per value.
    static_assert(Dh_MAX % THREADS_PER_VALUE == 0, "");

    // The size of a warp.
    constexpr int WARP_SIZE = 32;
    // The number of warps in a threadblock.
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

    // Use smem_size_in_bytes (above) to determine the amount of shared memory.
    extern __shared__ char smem_[];

    // The shared memory for the Q*K^T values and partial logits in softmax.
    float* qk_smem = reinterpret_cast<float*>(smem_);

    // The shared memory for the logits. For FP32, that's the same buffer as qk_smem.
    char* logits_smem_ = smem_;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (sizeof(T) != 4) {
        // TODO - cahnge to tlength
        logits_smem_ +=
            (DO_CROSS_ATTENTION) ? div_up(params.seq_length + 1, 4) * 16 : div_up(params.timestep + 1, 4) * 16;
    }
    T* logits_smem = reinterpret_cast<T*>(logits_smem_);
#else
    float* logits_smem = reinterpret_cast<float*>(logits_smem_);
#endif

    // The shared memory to do the final reduction for the output values. Reuse qk_smem.
    T* out_smem = reinterpret_cast<T*>(smem_);

    // The shared memory buffers for the block-wide reductions. One for max, one for sum.
    __shared__ float red_smem[WARPS_PER_BLOCK * 2];

    // A vector of Q or K elements for the current timestep.
    using Qk_vec = typename Qk_vec_<T, Dh_MAX>::Type;

    // Use alignment for safely casting the shared buffers as Qk_vec.
    // Shared memory to store Q inputs.
    __shared__ __align__(sizeof(Qk_vec)) T q_smem[Dh_MAX];

    // This is one of the reasons we should have a separate kernel for cross attention
    __shared__ __align__(sizeof(Qk_vec)) T bias_smem[DO_CROSS_ATTENTION ? Dh_MAX : 1];

    // A vector of Q or K elements for the current timestep.
    using Qk_vec = typename Qk_vec_<T, Dh_MAX>::Type;
    // The number of elements per vector.
    constexpr int QK_VEC_SIZE = sizeof(Qk_vec) / sizeof(T);
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % QK_VEC_SIZE == 0, "");
    // We will use block wide reduction if needed
    // static_assert(Dh_MAX / QK_VEC_SIZE <= WARP_SIZE, "");
    // The number of vectors per warp.
    constexpr int QK_VECS_PER_WARP = Dh_MAX / QK_VEC_SIZE;

    // The layout of the cache is [B, H, Dh/x, L, x] with x == 4/8 for FP32/FP16. Since each thread
    // owns x elements, we have to decompose the linear index into chunks of x values and the posi-
    // tion of the thread in that chunk.

    // The number of elements in a chunk of 16B (that's the x in the above formula).
    constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);
    // The number of K vectors in 16B.
    constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec);

    // The batch/beam idx
    const int bi = blockIdx.y;
    if (params.finished != nullptr && params.finished[bi] == true) {
        return;
    }
    // The beam idx
    const int beami = bi % params.beam_width;
    // The "beam-aware" batch idx
    const int bbi = bi / params.beam_width;
    // The head.
    const int hi = blockIdx.x;
    // Combine the batch and the head indices.
    const int bhi = bi * params.num_heads + hi;
    // Combine the "beam-aware" batch idx and the head indices.
    const int bbhi = bbi * params.beam_width * params.num_heads + hi;
    // The thread in the block.
    const int tidx = threadIdx.x;

    // While doing the product Q*K^T for the different keys we track the max.
    float qk_max = -FLT_MAX;

    float qk = 0.0F;

    int qkv_base_offset = (params.stride == 0) ? bhi * Dh : bi * params.stride + hi * Dh;

    // int tlength = (DO_CROSS_ATTENTION)? params.memory_length_per_sample[bi] - 1 : params.timestep;
    int tlength = (DO_CROSS_ATTENTION)                  ? params.memory_length_per_sample[bi] - 1 :
                  (params.length_per_sample == nullptr) ? params.timestep :
                                                          params.length_per_sample[bi];
    if ((!DO_CROSS_ATTENTION) && (params.length_per_sample != nullptr) && (tidx == 0))
        params.length_per_sample[bi]+=1;

    // First QK_VECS_PER_WARP load Q and K + the bias values for the current timestep.
    if (tidx < QK_VECS_PER_WARP) {

        // The offset in the Q and K buffer also accounts for the batch.
        int qk_offset = qkv_base_offset + tidx * QK_VEC_SIZE;
        // The offset in the bias buffer.
        int qk_bias_offset = hi * Dh + tidx * QK_VEC_SIZE;

        // Trigger the loads from the Q and K buffers.
        Qk_vec q;

        zero(q);
        q = (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ? *reinterpret_cast<const Qk_vec*>(&params.q[qk_offset]) : q;

        Qk_vec k;
        zero(k);
        if (DO_CROSS_ATTENTION) {
            // The 16B chunk written by the thread.
            int co = tidx / QK_VECS_IN_16B;
            // The position of the thread in that 16B chunk.
            int ci = tidx % QK_VECS_IN_16B * QK_VEC_SIZE;

            // Two chunks are separated by L * x elements. A thread write QK_VEC_SIZE elements.
            int offset = bhi * params.seq_length * Dh + co * params.seq_length * QK_ELTS_IN_16B +
                         // params.timestep*QK_ELTS_IN_16B +
                         tlength * QK_ELTS_IN_16B + ci;
            k = (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ? *reinterpret_cast<const Qk_vec*>(&params.k_cache[offset]) :
                                                            k;
        }
        else {
            k = (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) ? *reinterpret_cast<const Qk_vec*>(&params.k[qk_offset]) : k;
        }

        // Trigger the loads from the Q and K bias buffers.
        Qk_vec q_bias;
        zero(q_bias);
        q_bias = (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) && params.q_bias != nullptr ?
                     *reinterpret_cast<const Qk_vec*>(&params.q_bias[qk_bias_offset]) :
                     q_bias;
        Qk_vec k_bias;
        zero(k_bias);

        if (!DO_CROSS_ATTENTION || (DO_CROSS_ATTENTION && params.timestep == 0)) {
            k_bias = (Dh == Dh_MAX || tidx * QK_VEC_SIZE < Dh) && params.k_bias != nullptr ?
                         *reinterpret_cast<const Qk_vec*>(&params.k_bias[qk_bias_offset]) :
                         k_bias;
        }

        // Computes the Q/K values with bias.
        q = add(q, q_bias);
        if (!DO_CROSS_ATTENTION || (DO_CROSS_ATTENTION && params.timestep == 0)) {
            k = add(k, k_bias);
            if (params.rotary_embedding_dim > 0) {
                apply_rotary_embedding(q, k, tidx, params.rotary_embedding_dim, params.timestep);
            }
        }
        else {
            if (params.rotary_embedding_dim > 0) {
                apply_rotary_embedding(q, tidx, params.rotary_embedding_dim, params.timestep);
            }
        }

        // Store the Q values to shared memory.
        *reinterpret_cast<Qk_vec*>(&q_smem[tidx * QK_VEC_SIZE]) = q;

        // Store Dh values of k_bias into smem, since will need to add later
        // if params.timestep == 0
        if (DO_CROSS_ATTENTION && params.timestep == 0) {
            *reinterpret_cast<Qk_vec*>(&bias_smem[tidx * QK_VEC_SIZE]) = k_bias;
        }

        // Write the K values to the global memory cache.
        //
        // NOTE: The stores are uncoalesced as we have multiple chunks of 16B spread across the memory
        // system. We designed it this way as it allows much better memory loads (and there are many
        // more loads) + the stores are really "write and forget" since we won't need the ack before
        // the end of the kernel. There's plenty of time for the transactions to complete.

        // The 16B chunk written by the thread.
        int co = tidx / QK_VECS_IN_16B;
        // The position of the thread in that 16B chunk.
        int ci = tidx % QK_VECS_IN_16B * QK_VEC_SIZE;

        // Two chunks are separated by L * x elements. A thread write QK_VEC_SIZE elements.
        int offset = bhi * params.seq_length * Dh + co * params.seq_length * QK_ELTS_IN_16B +
                     // params.timestep*QK_ELTS_IN_16B +
                     tlength * QK_ELTS_IN_16B + ci;

        if (!DO_CROSS_ATTENTION || (DO_CROSS_ATTENTION && params.timestep == 0)) {
            // Trigger the stores to global memory.
            if (Dh == Dh_MAX || co < Dh / QK_ELTS_IN_16B) {
                *reinterpret_cast<Qk_vec*>(&params.k_cache[offset]) = k;
            }
        }

        // Compute \sum_i Q[i] * K^T[i] for the current timestep.
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
        using Qk_vec_acum = typename Qk_vec_acum_fp32_<Qk_vec>::Type;
#else
        using Qk_vec_acum = Qk_vec;
#endif
        qk = dot<Qk_vec_acum, Qk_vec>(q, k);
        if (QK_VECS_PER_WARP <= WARP_SIZE) {
#pragma unroll
            for (int mask = QK_VECS_PER_WARP / 2; mask >= 1; mask /= 2) {
                qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_WARP), qk, mask);
            }
        }
    }

    if (QK_VECS_PER_WARP > WARP_SIZE) {
        constexpr int WARPS_PER_RED = (QK_VECS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;
        qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
    }

    // Store that value in shared memory. Keep the Q*K^T value in register for softmax.
    if (tidx == 0) {
        // Normalize qk.
        qk *= params.inv_sqrt_dh;

        if (params.relative_attention_bias_float != nullptr) {
            qk = qk
                 + params.relative_attention_bias_float[hi * params.relative_attention_bias_stride
                                                            * params.relative_attention_bias_stride
                                                        + tlength * params.relative_attention_bias_stride + tlength];
        }
        else if (params.relative_attention_bias_half != nullptr) {
            qk = qk
                 + (float)
                       params.relative_attention_bias_half[hi * params.relative_attention_bias_stride
                                                               * params.relative_attention_bias_stride
                                                           + tlength * params.relative_attention_bias_stride + tlength];
        }
        qk_max = qk;
        qk_smem[tlength] = qk;
        // qk_smem[params.timestep] = qk;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The type of queries and keys for the math in the Q*K^T product.
    using K_vec = typename K_vec_<T, THREADS_PER_KEY>::Type;
    // The number of elements per vector.
    constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(T);
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % K_VEC_SIZE == 0, "");
    // The number of elements per thread.
    constexpr int K_ELTS_PER_THREAD = Dh_MAX / THREADS_PER_KEY;
    // The number of vectors per thread.
    constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;

    // The position the first key loaded by each thread from the cache buffer (for this B * H).
    int ko = tidx / THREADS_PER_KEY;
    // The position of the thread in the chunk of keys.
    int ki = tidx % THREADS_PER_KEY * K_VEC_SIZE;

    static_assert(Dh_MAX == THREADS_PER_KEY * K_VEC_SIZE * K_VECS_PER_THREAD);

    // Load the Q values from shared memory. The values are reused during the loop on K.
    K_vec q[K_VECS_PER_THREAD];
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
        q[ii] = *reinterpret_cast<const K_vec*>(&q_smem[ki + ii * THREADS_PER_KEY * K_VEC_SIZE]);
    }

    K_vec k_bias[DO_CROSS_ATTENTION ? K_VECS_PER_THREAD : 1];
    if (DO_CROSS_ATTENTION && params.timestep == 0) {
#pragma unroll
        for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
            k_bias[ii] = *reinterpret_cast<const K_vec*>(&bias_smem[ki + ii * THREADS_PER_KEY * K_VEC_SIZE]);
        }
    }

    // The number of timesteps loaded per iteration.
    constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
    // The number of keys per warp.
    constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

    // The base pointer for the key in the cache buffer.
    T* k_cache = &params.k_cache[bhi * params.seq_length * Dh + ki];
    // Base pointer for the beam's batch, before offsetting with indirection buffer
    T* k_cache_batch = &params.k_cache[bbhi * params.seq_length * Dh + ki];

    // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
    // int ti_end = div_up(params.timestep, K_PER_WARP) * K_PER_WARP;
    int ti_end = div_up(tlength, K_PER_WARP) * K_PER_WARP;
    


    // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
    for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {

        // The keys loaded from the key cache.
        K_vec k[K_VECS_PER_THREAD];
        K_vec k_vec_zero;
        zero(k_vec_zero);
#pragma unroll
        for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
            int jj = ii * params.seq_length + ti;
            // if( ti < params.timestep ) {
            if (ti < tlength) {
                    if (params.cache_indir != nullptr){
                    const int beam_src =
                        (params.cache_indir != nullptr) ?
                            params.cache_indir[(bbi * params.beam_width + beami) * params.seq_length + ti] :
                            0;
                    const int beam_offset = beam_src * params.num_heads * params.seq_length * Dh;
                    k[ii] = (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.seq_length) ?
                                *reinterpret_cast<const K_vec*>(&k_cache_batch[beam_offset + jj * QK_ELTS_IN_16B]) :
                                k_vec_zero;
                }
                else{
                    k[ii] = (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.seq_length) ?
                                *reinterpret_cast<const K_vec*>(&k_cache_batch[0 + jj * QK_ELTS_IN_16B]) :
                                k_vec_zero;
                }
                // add bias and update k_cache

                if (DO_CROSS_ATTENTION && params.timestep == 0) {
                    k[ii] = add(k[ii], k_bias[ii]);
                    if (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.seq_length) {
                        *reinterpret_cast<K_vec*>(&k_cache[jj * QK_ELTS_IN_16B]) = k[ii];
                    }
                }
            }
        }

        // Perform the dot product and normalize qk.
        //
        // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
        float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q, k) * params.inv_sqrt_dh;
        bool is_mask = (params.input_lengths != nullptr && ti >= params.input_lengths[bi] && ti < params.max_input_len);

        // Store the product to shared memory. There's one qk value per timestep. Update the max.
        // if( ti < params.timestep && tidx % THREADS_PER_KEY == 0 ) {
        if (ti < tlength && tidx % THREADS_PER_KEY == 0) {
            if (params.relative_attention_bias_float != nullptr) {
                qk = qk
                     + params.relative_attention_bias_float[hi * params.relative_attention_bias_stride
                                                                * params.relative_attention_bias_stride
                                                            + tlength * params.relative_attention_bias_stride + ti];
            }
            else if (params.relative_attention_bias_half != nullptr) {
                qk = qk
                     + (float)
                           params.relative_attention_bias_half[hi * params.relative_attention_bias_stride
                                                                   * params.relative_attention_bias_stride
                                                               + tlength * params.relative_attention_bias_stride + ti];
            }
            qk_max = is_mask ? qk_max : fmaxf(qk_max, qk);
            qk_smem[ti] = qk;
        }
    }

// Perform the final reduction to compute the max inside each warp.
//
// NOTE: In a group of THREADS_PER_KEY threads, the leader already has the max value for the
// group so it's not needed to run the reduction inside the group (again).
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }

    // Decompose the thread index into warp and lane.
    const int warp = tidx / WARP_SIZE;
    const int lane = tidx % WARP_SIZE;

    // The warp leader writes the max to shared memory.
    if (lane == 0) {
        red_smem[warp] = qk_max;
    }

    // Make sure the products are in shared memory.
    __syncthreads();

    // The warps finalize the reduction.
    qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }

    // Broadcast to all the threads in the warp.
    qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

    // Compute the logits and start the sum.
    float sum = 0.f;
    // for( int ti = tidx; ti <= params.timestep; ti += THREADS_PER_BLOCK ) {
    for (int ti = tidx; ti <= tlength; ti += THREADS_PER_BLOCK) {
        bool is_mask = (params.input_lengths != nullptr && ti >= params.input_lengths[bi] && ti < params.max_input_len);
        float logit = is_mask ? 0.f : __expf(qk_smem[ti] - qk_max);
        sum += logit;
        qk_smem[ti] = logit;
    }

    // Compute the sum.
    sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

    // Normalize the logits.
    float inv_sum = __fdividef(1.f, sum + 1.e-6f);
    // for( int ti = tidx; ti <= params.timestep; ti += THREADS_PER_BLOCK ) {
    for (int ti = tidx; ti <= tlength; ti += THREADS_PER_BLOCK) {
        convert_from_float(logits_smem[ti], qk_smem[ti] * inv_sum);
    }

    // Put Values part below so we leverage __syncthreads
    // from the previous step

    // The number of elements per vector.
    constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
    // A vector of V elements for the current timestep.
    using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;

    // The value computed by this thread.
    int vo = tidx / THREADS_PER_VALUE;
    // The hidden dimensions computed by this particular thread.
    int vi = tidx % THREADS_PER_VALUE * V_VEC_SIZE;

    // The base pointer for the value in the cache buffer.
    T* v_cache = &params.v_cache[bhi * params.seq_length * Dh + vi];
    // Base pointer for the beam's batch, before offsetting with indirection buffer
    T* v_cache_batch = &params.v_cache[bbhi * params.seq_length * Dh + vi];

    // The number of values processed per iteration of the loop.
    constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;

    // One group of threads computes the product(s) for the current timestep.
    V_vec v_bias;
    zero(v_bias);
    // if( vo == params.timestep % V_PER_ITER ) {
    if (Dh == Dh_MAX || vi < Dh) {
        if (!DO_CROSS_ATTENTION || (DO_CROSS_ATTENTION && params.timestep == 0)) {
            if (vo == tlength % V_PER_ITER) {
                // Trigger the loads from the V bias buffer.
                if (params.v_bias != nullptr) {
                    v_bias = *reinterpret_cast<const V_vec*>(&params.v_bias[hi * Dh + vi]);
                }
                if (DO_CROSS_ATTENTION) {
                    *reinterpret_cast<V_vec*>(&bias_smem[vi]) = v_bias;
                }
            }
        }
    }

    // From previous, before values, step
    // Also make sure the logits are in shared memory.
    __syncthreads();

    // Values continued
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
    using V_vec_acum = typename V_vec_acum_fp32_<V_vec>::Type;
#else
    using V_vec_acum = V_vec;
#endif
    // The partial outputs computed by each thread.
    V_vec_acum out;
    zero(out);


    // Loop over the timesteps to compute the partial outputs.
    // for( int ti = vo; ti < params.timestep; ti += V_PER_ITER ) {
    if (Dh == Dh_MAX || vi < Dh) {
        for (int ti = vo; ti < tlength; ti += V_PER_ITER) {

            // Fetch offset based on cache_indir when beam sampling
            V_vec v;
            if (params.cache_indir != nullptr){
                const int beam_src = (params.cache_indir != nullptr) ?
                                         params.cache_indir[(bbi * params.beam_width + beami) * params.seq_length + ti] :
                                         0;
                const int beam_offset = beam_src * params.num_heads * params.seq_length * Dh;
                // Load the values from the cache.
                v = *reinterpret_cast<const V_vec*>(&v_cache_batch[beam_offset + ti * Dh]);
            }
            else{
                v = *reinterpret_cast<const V_vec*>(&v_cache_batch[0 + ti * Dh]);
            }


            if (DO_CROSS_ATTENTION && params.timestep == 0) {
                v = add(v, *reinterpret_cast<V_vec*>(&bias_smem[vi]));
                *reinterpret_cast<V_vec*>(&v_cache[ti * Dh]) = v;
            }
            // Load the logits from shared memory.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
            float logit = logits_smem[ti];
            out = fma(logit, cast_to_float(v), out);
#else
            T logit = logits_smem[ti];

            // Update the partial sums.
            out = fma(logit, v, out);
#endif
        }
    }

    // One group of threads computes the product(s) for the current timestep.
    // if( vo == params.timestep % V_PER_ITER ) {
    if (vo == tlength % V_PER_ITER && (Dh == Dh_MAX || vi < Dh)) {

        V_vec v;
        if (DO_CROSS_ATTENTION) {
            v = *reinterpret_cast<const V_vec*>(&v_cache[tlength * Dh]);
        }
        else {
            // Trigger the loads from the V buffer.
            v = *reinterpret_cast<const V_vec*>(&params.v[qkv_base_offset + vi]);
            // Trigger the loads from the V bias buffer.
            // V_vec v_bias = *reinterpret_cast<const V_vec*>(&params.v_bias[hi*Dh + vi]);
        }

        // Compute the V values with bias.
        if (!DO_CROSS_ATTENTION || (DO_CROSS_ATTENTION && params.timestep == 0)) {
            v = add(v, v_bias);

            // Store the values with bias back to global memory in the cache for V.
            //*reinterpret_cast<V_vec*>(&v_cache[params.timestep*Dh]) = v;
            *reinterpret_cast<V_vec*>(&v_cache[tlength * Dh]) = v;
        }

        // Initialize the output value with the current timestep.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
        // out = fma(logits_smem[params.timestep], cast_to_float(v), out);
        out = fma(logits_smem[tlength], cast_to_float(v), out);
#else
        // out = fma(logits_smem[params.timestep], v, out);
        out = fma(logits_smem[tlength], v, out);
#endif
    }

    // Make sure we can start writing to shared memory.
    __syncthreads();

    // Run the final reduction amongst the different groups computing different partial outputs.
    if (Dh == Dh_MAX || vi < Dh)
#pragma unroll
        for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2) {

            // The midpoint in the number of active groups.
            int midpoint = active_groups / 2;

            // The upper part of active threads store to shared memory.
            if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
                convert_from_float(*reinterpret_cast<V_vec*>(&out_smem[(vo - midpoint) * Dh + vi]), out);
#else
                *reinterpret_cast<V_vec*>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
            }
            __syncthreads();

            // The bottom warps update their values.
            if (vo < midpoint && (Dh == Dh_MAX || vi < Dh)) {
                out = add(*reinterpret_cast<const V_vec*>(&out_smem[vo * Dh + vi]), out);
            }
            __syncthreads();
        }

    // Output the final values.
    if (vo == 0 && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
        convert_from_float(*reinterpret_cast<V_vec*>(&params.out[bhi * Dh + vi]), out);
#else
        *reinterpret_cast<V_vec*>(&params.out[bhi * Dh + vi]) = out;
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace mmha

#define MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, THDS_PER_KEY, THDS_PER_VALUE, THDS_PER_BLOCK, DO_CROSS_ATTENTION, stream)    \
    size_t smem_sz = mmha::smem_size_in_bytes<T, DO_CROSS_ATTENTION>(params, THDS_PER_VALUE, THDS_PER_BLOCK);          \
    dim3 grid(params.num_heads, params.batch_size);                                                                    \
    mmha::masked_multihead_attention_kernel_optimized<T,                                                                         \
                                            Dh,                                                                        \
                                            Dh_MAX,                                                                    \
                                            THDS_PER_KEY,                                                              \
                                            THDS_PER_VALUE,                                                            \
                                            THDS_PER_BLOCK,                                                            \
                                            DO_CROSS_ATTENTION><<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(params)


template<typename T, int Dh, int Dh_MAX, int TPK, int TPB, typename KERNEL_PARAMS_TYPE, int T_size>
struct mmha_lahnch_TPV{
    static void call(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream, int v_vec_size)
    {
        throw std::exception(); 
    }
};

template<typename T, int Dh, int Dh_MAX, int TPK, int TPB, typename KERNEL_PARAMS_TYPE>
struct mmha_lahnch_TPV<T, Dh, Dh_MAX, TPK, TPB, KERNEL_PARAMS_TYPE, 4>{
    static void call(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream, int v_vec_size)
    {
        const bool DO_CROSS_ATTENTION = false;

        switch (v_vec_size){
            case 1:
                {
#ifdef tpv1
                    constexpr int THREADS_PER_VALUE = std::min(TPB, Dh_MAX);
                    MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, TPK, THREADS_PER_VALUE, TPB, DO_CROSS_ATTENTION, stream);
#endif
                    break;
                }
            case 2:
                {
#ifdef tpv2
                    constexpr int THREADS_PER_VALUE = Dh_MAX / 2;
                    MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, TPK, THREADS_PER_VALUE, TPB, DO_CROSS_ATTENTION, stream);
#endif
                    break;
                }
            case 4:
                {
#ifdef tpv4
                    constexpr int THREADS_PER_VALUE = Dh_MAX / 4;
                    MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, TPK, THREADS_PER_VALUE, TPB, DO_CROSS_ATTENTION, stream);
#endif
                    break;
                }
            default:
                assert(false);
        } 
    }
};

template<typename T, int Dh, int Dh_MAX, int TPK, int TPB, typename KERNEL_PARAMS_TYPE>
struct mmha_lahnch_TPV<T, Dh, Dh_MAX, TPK, TPB, KERNEL_PARAMS_TYPE, 2>{
    static void call(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream, int v_vec_size) 
    {
        const bool DO_CROSS_ATTENTION = false;

        switch (v_vec_size){
            case 2:
                {
#ifdef tpv2
                    constexpr int THREADS_PER_VALUE = std::min(TPB, Dh_MAX / 2);
                    MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, TPK, THREADS_PER_VALUE, TPB, DO_CROSS_ATTENTION, stream);
#endif  
                    break;
                }
            case 4:
                {
#ifdef tpv4
                    constexpr int THREADS_PER_VALUE = Dh_MAX / 4;
                    MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, TPK, THREADS_PER_VALUE, TPB, DO_CROSS_ATTENTION, stream);
#endif  
                    break;
                }
            case 8:
                {
#ifdef tpv8
                    constexpr int THREADS_PER_VALUE = Dh_MAX / 8;
                    MMHA_LAUNCH_KERNEL_OPTIMIZED(T, Dh, Dh_MAX, TPK, THREADS_PER_VALUE, TPB, DO_CROSS_ATTENTION, stream);
#endif  
                    break;
                }
            default:
                assert(false);
        } 
    }
};


template<typename T, int Dh, int Dh_MAX, int TPB, typename KERNEL_PARAMS_TYPE>
void mmha_launch_TPK(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream, int v_vec_size, int k_vec_size)
{
    switch (k_vec_size){
        case 1:
#ifdef tpk1
            mmha_lahnch_TPV<T, Dh, Dh_MAX, 1, TPB, KERNEL_PARAMS_TYPE, sizeof(T)>::call(params, stream, v_vec_size);
#endif  
            break;
        case 2:
#ifdef tpk2
            mmha_lahnch_TPV<T, Dh, Dh_MAX, 2, TPB, KERNEL_PARAMS_TYPE, sizeof(T)>::call(params, stream, v_vec_size);
#endif  
            break;
        case 4:
#ifdef tpk4
            mmha_lahnch_TPV<T, Dh, Dh_MAX, 4, TPB, KERNEL_PARAMS_TYPE, sizeof(T)>::call(params, stream, v_vec_size);
#endif  
            break;
        default:
            assert(false);
    }
}

template<typename T, int Dh, int Dh_MAX, typename KERNEL_PARAMS_TYPE>
void mmha_launch(const KERNEL_PARAMS_TYPE& params, const cudaStream_t& stream, int v_vec_size, int k_vec_size, int threads_per_block)
{
    switch (threads_per_block){
        case 64:
#ifdef tpb64
            mmha_launch_TPK<T, Dh, Dh_MAX, 64, KERNEL_PARAMS_TYPE>(params, stream, v_vec_size, k_vec_size);
#endif  
            break;
        case 128:
#ifdef tpb128
            mmha_launch_TPK<T, Dh, Dh_MAX, 128, KERNEL_PARAMS_TYPE>(params, stream, v_vec_size, k_vec_size);
#endif  
            break;
        case 256:
#ifdef tpb256
            mmha_launch_TPK<T, Dh, Dh_MAX, 256, KERNEL_PARAMS_TYPE>(params, stream, v_vec_size, k_vec_size);
#endif  
            break;
        case 512:
#ifdef tpb512
            mmha_launch_TPK<T, Dh, Dh_MAX, 512, KERNEL_PARAMS_TYPE>(params, stream, v_vec_size, k_vec_size);
#endif  
            break;
        default:
            assert(false);
    }
}

inline bool cuda_check_error(std::string message){
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA %s error: %s\n", message.c_str(), cudaGetErrorString(error));
        return false;
    }
    return true;
}


template<typename T, int Dh, int Dh_MAX>
void call_attention_headdim(int max_seq_length, int batch_size,int head_num, T* q, T* k, T* v, T* k_cache, T* v_cache, T* out, float softmax_scale, int curr_timestep, int* seq_lengths)
{
    Masked_multihead_attention_params<T> params;
    
    memset(&params, 0, sizeof(params));
    
    int v_vec_size=8; int k_vec_size=2; int threads_per_block=128;
    //int no_sms = 108;  float C = 3;   float gpu_clock_rate = 1215e6; int mySharedMemAllocationSize=128; int max_sharedmemory_per_block=102400; int run_time_smem=1024;
    //float gpu_transfer_rate =1550*std::pow(2,30) ;
    int no_sms = 82;  float C = 3;   float gpu_clock_rate = 1395e6; int mySharedMemAllocationSize=128; int max_sharedmemory_per_block=102400; int run_time_smem=1024;
    float gpu_transfer_rate =936*std::pow(2,30) ;
    int precision = sizeof(T) == 4?32:16;

    //int hidden_units = head_num * Dh;
    params.num_heads = head_num;
    params.hidden_size_per_head = Dh;

    params.q_bias = nullptr;
    params.k_bias = nullptr;
    params.v_bias = nullptr;
    params.stride = 0; //3 * hidden_units;
    params.finished = nullptr;
    params.cache_indir = nullptr;
    params.input_lengths = nullptr;
    params.beam_width = 1;
    params.rotary_embedding_dim = 0;
    params.inv_sqrt_dh = softmax_scale ;
    params.relative_attention_bias_stride = 0;

    params.max_input_len = 0;
    params.length_per_sample = seq_lengths;
    params.batch_size = batch_size;
    params.seq_length = max_seq_length;
    params.timestep = curr_timestep;

    params.q = q;
    params.k = k;
    params.v = v;
    params.k_cache = k_cache;
    params.v_cache = v_cache;
    params.out = out;

    get_launch_params(precision, params.batch_size, params.num_heads, Dh_MAX, curr_timestep, no_sms, C, gpu_transfer_rate, gpu_clock_rate, mySharedMemAllocationSize, max_sharedmemory_per_block, run_time_smem, threads_per_block, v_vec_size, k_vec_size);

    //printf("v_vec_size=%d, k_vec_size=%d, threads_per_block=%d", v_vec_size, k_vec_size, threads_per_block);
    mmha_launch<T, Dh, Dh_MAX, Masked_multihead_attention_params<T>>(params, 0, v_vec_size, k_vec_size, threads_per_block);

    cuda_check_error("Attention kernel erro launch");
}

template<typename T>
void call_attention(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, T* q, T* k, T* v, T* k_cache, T* v_cache, T* out, int* seq_lengths);

template<>
void call_attention<float>(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, float* q, float* k, float* v, float* k_cache, float* v_cache, float* out, int* seq_lengths)
{
    switch (size_per_head){
        case 64:
            call_attention_headdim<float, 64, 64>(seq_length, batch_size,head_num, q, k, v, k_cache, v_cache, out, softmax_scale, curr_timestep, seq_lengths);
            break;
        case 80:
            call_attention_headdim<float, 80, 128>(seq_length,batch_size,head_num, q, k, v, k_cache, v_cache, out, softmax_scale, curr_timestep, seq_lengths);
            break;
        case 128:
            call_attention_headdim<float, 128, 128>(seq_length, batch_size,head_num, q, k, v, k_cache, v_cache, out, softmax_scale, curr_timestep, seq_lengths);
            break;
        case 256:
            call_attention_headdim<float, 256, 256>(seq_length, batch_size,head_num, q, k, v, k_cache, v_cache, out, softmax_scale, curr_timestep, seq_lengths);
            break;
        default:
            printf("Not support head size\n");
            break;
    }
}

template<>
void call_attention<uint16_t>(int batch_size,int head_num, int size_per_head, int seq_length, float softmax_scale, int curr_timestep, uint16_t* q, uint16_t* k, uint16_t* v, uint16_t* k_cache, uint16_t* v_cache, uint16_t* out, int* seq_lengths)
{
    switch (size_per_head){
        case 64:
            call_attention_headdim<uint16_t, 64, 64>(seq_length, batch_size,head_num, q, k, v, k_cache, v_cache, out, softmax_scale, curr_timestep, seq_lengths);
            break;
        case 80:
            call_attention_headdim<uint16_t, 80, 128>(seq_length,batch_size,head_num, q, k, v, k_cache, v_cache, out, softmax_scale, curr_timestep, seq_lengths);
            break;
        case 128:
            call_attention_headdim<uint16_t, 128, 128>(seq_length, batch_size,head_num, q, k, v, k_cache, v_cache, out, softmax_scale, curr_timestep, seq_lengths);
            break;
        case 256:
            call_attention_headdim<uint16_t, 256, 256>(seq_length, batch_size,head_num, q, k, v, k_cache, v_cache, out, softmax_scale, curr_timestep, seq_lengths);
            break;
        default:
            printf("Not support head size\n");
            break;
    }
}

template<typename T>
__global__ void transpose_4d_batch_major_k_cache(
    T* k_dst, const T* k_src, const int head_num, const int size_per_head, const int seq_len, const int max_seq_len)
{
    const int     batch_id = blockIdx.y;
    const int     head_id  = blockIdx.z;
    constexpr int X_ELEMS  = (sizeof(T) == 4) ? 4 : 8;

    auto key_src = reinterpret_cast<const uint4*>(k_src + batch_id * head_num * size_per_head * seq_len
                                                  + head_id * size_per_head * seq_len);
    auto key_dst = reinterpret_cast<uint4*>(k_dst + batch_id * head_num * size_per_head * max_seq_len
                                            + head_id * size_per_head * max_seq_len);

    const int out_idx             = blockIdx.x * blockDim.x + threadIdx.x;
    int       size_per_head_div_x = size_per_head / X_ELEMS;
    if (out_idx >= size_per_head_div_x * max_seq_len) {
        return;
    }

    int       idx            = out_idx;
    const int k_seq_len_id   = idx % max_seq_len;
    idx                      = (idx - k_seq_len_id) / max_seq_len;
    const int k_head_size_id = idx % size_per_head_div_x;

    if (k_seq_len_id < seq_len) {
        key_dst[out_idx] = key_src[k_seq_len_id * size_per_head_div_x + k_head_size_id];
    }
}

template<typename T>
__global__ void transpose_4d_batch_major_v_cache(
    T* v_dst, const T* v_src, const int head_num, const int size_per_head, const int seq_len, const int max_seq_len)
{
    const int batch_id = blockIdx.y;
    const int head_id  = blockIdx.z;

    // 16 byte loads will handle "x" dimension
    auto val_src = reinterpret_cast<const uint4*>(v_src + batch_id * head_num * size_per_head * seq_len
                                                  + head_id * size_per_head * seq_len);
    auto val_dst = reinterpret_cast<uint4*>(v_dst + batch_id * head_num * size_per_head * max_seq_len
                                            + head_id * size_per_head * max_seq_len);

    // idx is over output dimension L * size_per_head / x for values
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int X_ELEMS             = (sizeof(T) == 4) ? 4 : 8;
    const int     size_per_head_div_x = size_per_head / X_ELEMS;

    if (idx >= size_per_head_div_x * seq_len) {
        return;
    }

    val_dst[idx] = val_src[idx];
}

template<typename T>
void invokeTranspose4dBatchMajor_T(T*           k_dst,
                                 T*           v_dst,
                                 const T*     k_src,
                                 const T*     v_src,
                                 const int    local_batch_size,
                                 const int    seq_len,
                                 const int    max_seq_len,
                                 const int    size_per_head,
                                 const int    local_head_num
                                 )
{
    constexpr int block_sz = 128;
    constexpr int x        = (sizeof(T) == 4) ? 4 : 8;
    int           size     = max_seq_len * size_per_head / x;
    dim3          grid((size + block_sz - 1) / block_sz, local_batch_size, local_head_num);
    dim3          grid_v((seq_len * size_per_head / x + block_sz - 1) / block_sz, local_batch_size, local_head_num);

    transpose_4d_batch_major_k_cache<<<grid, block_sz>>>(
        k_dst, k_src, local_head_num, size_per_head, seq_len, max_seq_len);

    transpose_4d_batch_major_v_cache<<<grid_v, block_sz>>>(
        v_dst, v_src, local_head_num, size_per_head, seq_len, max_seq_len);
}

template<typename T>
void invokeTranspose4dBatchMajor(T*           k_dst,
                                 T*           v_dst,
                                 const T*     k_src,
                                 const T*     v_src,
                                 const int    local_batch_size,
                                 const int    seq_len,
                                 const int    max_seq_len,
                                 const int    size_per_head,
                                 const int    local_head_num
                                 );


template<>
void invokeTranspose4dBatchMajor<float>(float*           k_dst,
                                         float*           v_dst,
                                         const float*     k_src,
                                         const float*     v_src,
                                         const int    batch_size,
                                         const int    seq_len,
                                         const int    max_seq_len,
                                         const int    size_per_head,
                                         const int    local_head_num
                                         )
{
        invokeTranspose4dBatchMajor_T<float>( k_dst, v_dst, k_src, v_src,
                                     batch_size,
                                     seq_len,
                                     max_seq_len,
                                     size_per_head,
                                     local_head_num
                                     );
}

template<>
void invokeTranspose4dBatchMajor<uint16_t>(uint16_t*           k_dst,
                                         uint16_t*           v_dst,
                                         const uint16_t*     k_src,
                                         const uint16_t*     v_src,
                                         const int    local_batch_size,
                                         const int    seq_len,
                                         const int    max_seq_len,
                                         const int    size_per_head,
                                         const int    local_head_num
                                     )
{
        invokeTranspose4dBatchMajor_T<uint16_t>( k_dst, v_dst, k_src, v_src,
                                     local_batch_size,
                                     seq_len,
                                     max_seq_len,
                                     size_per_head,
                                     local_head_num
                                     );
}


