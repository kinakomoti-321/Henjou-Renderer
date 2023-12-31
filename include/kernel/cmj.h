#pragma once 
#include <cuda/helpers.h>
#include <sutil/vec_math.h>

//corrected multi-jittered sampling
//https://blog.teastat.uk/post/2022/08/use-cmj-in-montecarlo-raytracing/
//https://github.com/yumcyaWiz/fredholm/blob/main/fredholm/modules/cmj.cu

#define CMJ_M 4
#define CMJ_N 4
#define uint unsigned int

// https://www.shadertoy.com/view/XlGcRh
static __forceinline__ __device__ __host__ uint xxhash32(uint p)
{
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint h32 = p + PRIME32_5;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

static __forceinline__ __device__ __host__ uint xxhash32(const uint3& p)
{
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint h32 = p.z + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

static __forceinline__ __device__ __host__ uint xxhash32(const uint4& p)
{
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint h32 = p.w + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.z * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

struct CMJState {
    unsigned long long n_spp = 0;
    unsigned int scramble = 0;
    unsigned int depth = 0;
    unsigned int image_idx = 0;
};

__forceinline__ __device__ unsigned int cmj_permute(unsigned int i,
    unsigned int l,
    unsigned int p)
{
    unsigned int w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;
    do {
        i ^= p;
        i *= 0xe170893d;
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;
        i *= 0x0929eb3f;
        i ^= p >> 23;
        i ^= (i & w) >> 1;
        i *= 1 | p >> 27;
        i *= 0x6935fa69;
        i ^= (i & w) >> 11;
        i *= 0x74dcb303;
        i ^= (i & w) >> 2;
        i *= 0x9e501cc3;
        i ^= (i & w) >> 2;
        i *= 0xc860a3df;
        i &= w;
        i ^= i >> 5;
    } while (i >= l);
    return (i + p) % l;
}

__forceinline__ __device__ float cmj_randfloat(unsigned int i, unsigned int p)
{
    i ^= p;
    i ^= i >> 17;
    i ^= i >> 10;
    i *= 0xb36534e5;
    i ^= i >> 12;
    i ^= i >> 21;
    i *= 0x93fc4795;
    i ^= 0xdf6e307f;
    i ^= i >> 17;
    i *= 1 | p >> 18;
    return i * (1.0f / 4294967808.0f);
}

__forceinline__ __device__ float2 cmj(unsigned int index, unsigned int scramble)
{
    index = cmj_permute(index, CMJ_M * CMJ_N, scramble * 0x51633e2d);
    unsigned int sx = cmj_permute(index % CMJ_M, CMJ_M, scramble * 0xa511e9b3);
    unsigned int sy = cmj_permute(index / CMJ_M, CMJ_N, scramble * 0x63d83595);
    float jx = cmj_randfloat(index, scramble * 0xa399d265);
    float jy = cmj_randfloat(index, scramble * 0x711ad6a5);
    return make_float2((index % CMJ_M + (sy + jx) / CMJ_N) / CMJ_M,
        (index / CMJ_M + (sx + jy) / CMJ_M) / CMJ_N);
}

static __forceinline__ __device__ float2 cmj_2d(CMJState& state)
{
    const unsigned int index = state.n_spp % (CMJ_M * CMJ_N);
    const unsigned int scramble =
        xxhash32(make_uint4(state.n_spp / (CMJ_M * CMJ_N), state.image_idx,
            state.depth, state.scramble));
    const float2 result = cmj(index, scramble);
    state.depth++;
    return result;
}

static __forceinline__ __device__ float cmj_1d(CMJState& state)
{
    return cmj_2d(state).x;
}

static __forceinline__ __device__ float3 cmj_3d(CMJState& state)
{
    return make_float3(cmj_2d(state), cmj_1d(state));
}

static __forceinline__ __device__ float4 cmj_4d(CMJState& state)
{
    return make_float4(cmj_2d(state), cmj_2d(state));
}

//static __forceinline__ __device__ unsigned int cmj_permute(unsigned int i, unsigned int l,
//    unsigned int p) {
//    unsigned int w = l - 1;
//    w |= w >> 1;
//    w |= w >> 2;
//    w |= w >> 4;
//    w |= w >> 8;
//    w |= w >> 16;
//    do {
//        i ^= p;
//        i *= 0xe170893d;
//        i ^= p >> 16;
//        i ^= (i & w) >> 4;
//        i ^= p >> 8;
//        i *= 0x0929eb3f;
//        i ^= p >> 23;
//        i ^= (i & w) >> 1;
//        i *= 1 | p >> 27;
//        i *= 0x6935fa69;
//        i ^= (i & w) >> 11;
//        i *= 0x74dcb303;
//        i ^= (i & w) >> 2;
//        i *= 0x9e501cc3;
//        i ^= (i & w) >> 2;
//        i *= 0xc860a3df;
//        i &= w;
//        i ^= i >> 5;
//    } while (i >= l);
//    return (i + p) % l;
//}
//
//static __forceinline__ __device__ float cmj_randfloat(unsigned int i, unsigned int p) {
//    i ^= p;
//    i ^= i >> 17;
//    i ^= i >> 10;
//    i *= 0xb36534e5;
//    i ^= i >> 12;
//    i ^= i >> 21;
//    i *= 0x93fc4795;
//    i ^= 0xdf6e307f;
//    i ^= i >> 17;
//    i *= 1 | p >> 18;
//    return i * (1.0f / (1ULL << 32));
//}
//
//static __forceinline__ __device__ float2 cmj(unsigned int s, unsigned int p) {
//    const unsigned int CMJ_M = 16, CMJ_N = 16;
//    s = cmj_permute(s, CMJ_M * CMJ_N, p * 0x51633e2d); 
//    unsigned int sx = cmj_permute(s % CMJ_M, CMJ_M, p * 0xa511e9b3);
//    unsigned int sy = cmj_permute(s / CMJ_M, CMJ_N, p * 0x63d83595);
//    float jx = cmj_randfloat(s, p * 0xa399d265);
//    float jy = cmj_randfloat(s, p * 0x711ad6a5);
//    return make_float2(
//        (s % CMJ_M + (sy + jx) / CMJ_N) / CMJ_M,
//        (s / CMJ_M + (sx + jy) / CMJ_M) / CMJ_N);
//}
//
//static __forceinline__ __device__  unsigned int xxhash32(const uint4& p)
//{
//    const unsigned int PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
//    const unsigned int PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
//    unsigned int h32 = p.w + PRIME32_5 + p.x * PRIME32_3;
//    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
//    h32 += p.y * PRIME32_3;
//    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
//    h32 += p.z * PRIME32_3;
//    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
//    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
//    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
//    return h32 ^ (h32 >> 16);
//}
//
//__device__ struct CMJstate {
//    unsigned int n_spp;
//    unsigned int pixel_index;
//    unsigned int dimension;
//    unsigned int seed;
//};
//
//static __forceinline__ __device__ float2 cmj_2d(CMJstate& state) {
//    const unsigned int CMJ_M = 16, CMJ_N = 16;
//    unsigned int s = state.n_spp % (CMJ_M * CMJ_N);
//    unsigned int p = xxhash32(make_uint4(state.n_spp / (CMJ_M * CMJ_N),state.pixel_index, state.dimension, state.seed));
//
//    float2 result = cmj(s, p);
//    state.seed++;
//
//    return result;
//}
//
//static __forceinline__ __device__  float cmj_1d(CMJstate& state){
//    return cmj_2d(state).x;
//}
