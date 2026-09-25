/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "x64_simd_inl.h" // for IntelliSense

#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512

namespace mochi {

/***********************************************************************************************
  Simd<float, 16>
*/
template <>
class Simd<float, 16> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(float, 16, __m512);
  Simd(
      float a,
      float b,
      float c = 0.0f,
      float d = 0.0f,
      float e = 0.0f,
      float f = 0.0f,
      float g = 0.0f,
      float h = 0.0f,
      float i = 0.0f,
      float j = 0.0f,
      float k = 0.0f,
      float l = 0.0f,
      float m = 0.0f,
      float n = 0.0f,
      float o = 0.0f,
      float p = 0.0f)
      : raw(_mm512_set_ps(p, o, n, m, l, k, j, i, h, g, f, e, d, c, b, a)) {} // AVX512F

  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  Simd(U a) : raw(_mm512_set1_ps(a)) {} // AVX512F

  Simd(Simd<float, 8> const& low, Simd<float, 8> const& high)
      : raw(_mm512_insertf32x8(_mm512_castps256_ps512(low.raw), high.raw, 1)) {} // AVX512DQ

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE float Get(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return v[i];
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Scalar operator[](int i) const {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    return raw.m512_f32[i];
#else
    return raw[i];
#endif
  }

  template <int iHalf>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd<float, 8> GetHalf(Simd a) {
    static_assert(iHalf == 0 || iHalf == 1);
    if constexpr (iHalf == 0) {
      return _mm512_castps512_ps256(a.raw); // AVX512F
    } else {
      return _mm512_extractf32x8_ps(a.raw, 1); // AVX512DQ
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, int i, Scalar value) {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
    auto const mask = static_cast<__mmask16>(1u << i);
    return _mm512_mask_broadcastss_ps(v.raw, mask, _mm_set_ss(value));
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Set(Simd v, Scalar value) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    constexpr auto kMask = static_cast<__mmask16>(1u << i);
    return _mm512_mask_broadcastss_ps(v.raw, kMask, _mm_set_ss(value));
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    auto const mask = ToMask(v);
    if constexpr (N == kSize) {
      return _kortestc_mask16_u8(mask, mask) != 0;
    } else {
      constexpr auto kLanes = LaneMask<N>();
      return (mask & kLanes) == kLanes;
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AnyTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    auto const mask = ToMask(v);
    if constexpr (N == kSize) {
      return _kortestz_mask16_u8(mask, mask) == 0;
    } else {
      return (mask & LaneMask<N>()) != 0;
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Scalar const* p) {
    return _mm512_set1_ps(*p); // AVX512F
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    if constexpr (i == 0) {
      return _mm512_broadcastss_ps(_mm512_castps512_ps128(v.raw));
    } else {
      constexpr int kLane = i % 4;
      constexpr int kGroup = i / 4;
      auto const group =
          _mm512_shuffle_f32x4(v.raw, v.raw, _MM_SHUFFLE(kGroup, kGroup, kGroup, kGroup));
      return _mm512_permute_ps(group, _MM_SHUFFLE(kLane, kLane, kLane, kLane));
    }
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Scalar const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Zero();
    } else if constexpr (N == 1) {
      return _mm512_zextps128_ps512(_mm_load_ss(ptr));
    } else if constexpr (N == 2) {
      auto const low = _mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<__m128i const*>(ptr)));
      return _mm512_zextps128_ps512(low);
    } else if constexpr (N == 3) {
      return _mm512_zextps128_ps512(
          _mm_maskz_loadu_ps(static_cast<__mmask8>((uint32_t{1} << N) - 1), ptr));
    } else if constexpr (N == 4) {
      return _mm512_zextps128_ps512(_mm_loadu_ps(ptr));
    } else if constexpr (N < 8) {
      return _mm512_zextps256_ps512(
          _mm256_maskz_loadu_ps(static_cast<__mmask8>((uint32_t{1} << N) - 1), ptr));
    } else if constexpr (N == 8) {
      return _mm512_zextps256_ps512(_mm256_loadu_ps(ptr));
    } else if constexpr (N < kSize) {
      return _mm512_maskz_loadu_ps(LaneMask<N>(), ptr); // AVX512F
    } else {
      return _mm512_loadu_ps(ptr); // AVX512F
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    return _mm512_maskz_loadu_ps(LaneMask(n), ptr); // AVX512F
  }

  [[nodiscard]] static Simd LoadIndexed(Scalar const* ptr, Simd<int, 16> const& indices) {
    return _mm512_i32gather_ps(indices.raw, ptr, sizeof(float)); // AVX512F
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void
  LoadTransposed(Scalar const* ptr, Simd& out0, Simd& out1, Simd& out2) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    if constexpr (kTupleCount == 1) {
      out0 = Load<1>(ptr);
      out1 = Load<1>(ptr + 1);
      out2 = Load<1>(ptr + 2);
      return;
    }
    constexpr int kTotalCount = kTupleCount * 3;
    auto const x0 = Load<Clamp(kTotalCount, 0, kSize)>(ptr).raw;
    if constexpr (kTupleCount <= 10) {
      auto const index0 = LoadTransposeIndices<kTupleCount, 0>();
      auto const index1 = LoadTransposeIndices<kTupleCount, 1>();
      auto const index2 = LoadTransposeIndices<kTupleCount, 2>();
      if constexpr (kTupleCount <= 5) {
        out0.raw = _mm512_permutexvar_ps(index0, x0);
        out1.raw = _mm512_permutexvar_ps(index1, x0);
        out2.raw = _mm512_permutexvar_ps(index2, x0);
      } else {
        auto const x1 = Load<kTotalCount - kSize>(ptr + kSize).raw;
        out0.raw = _mm512_permutex2var_ps(x0, index0, x1);
        out1.raw = _mm512_permutex2var_ps(x0, index1, x1);
        out2.raw = _mm512_permutex2var_ps(x0, index2, x1);
      }
    } else {
      // clang-format off
      auto const x1 = Load<kSize>(ptr + kSize).raw;
      constexpr int kCount2 = kTotalCount - 2 * kSize;
      auto const x2 = Load<kCount2>(ptr + 2 * kSize).raw;
      auto const index0 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
      auto const index1 = _mm512_setr_epi32(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0);
      auto const index2 = _mm512_setr_epi32(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0);
      if constexpr (kTupleCount == 11) {
        out0.raw = _mm512_maskz_permutex2var_ps(LaneMask<kTupleCount>(), x0, index0, x1);
        out1.raw = _mm512_maskz_permutex2var_ps(LaneMask<kTupleCount>(), x0, index1, x1);
        constexpr int kZeroIndex = kSize + kCount2;
        auto const partial2 = _mm512_permutex2var_ps(x0, index2, x1);
        auto const finalIndex2 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, kZeroIndex, kZeroIndex, kZeroIndex, kZeroIndex, kZeroIndex);
        out2.raw = _mm512_permutex2var_ps(partial2, finalIndex2, x2);
      } else {
        constexpr int kZeroIndex = kSize + kCount2;
        auto const partial0 = _mm512_permutex2var_ps(x0, index0, x1);
        auto const finalIndex0 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, kTupleCount > 11 ? 17 : kZeroIndex, kTupleCount > 12 ? 20 : kZeroIndex, kTupleCount > 13 ? 23 : kZeroIndex, kTupleCount > 14 ? 26 : kZeroIndex, kTupleCount > 15 ? 29 : kZeroIndex);
        out0.raw = _mm512_permutex2var_ps(partial0, finalIndex0, x2);
        auto const partial1 = _mm512_permutex2var_ps(x0, index1, x1);
        auto const finalIndex1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, kTupleCount > 11 ? 18 : kZeroIndex, kTupleCount > 12 ? 21 : kZeroIndex, kTupleCount > 13 ? 24 : kZeroIndex, kTupleCount > 14 ? 27 : kZeroIndex, kTupleCount > 15 ? 30 : kZeroIndex);
        out1.raw = _mm512_permutex2var_ps(partial1, finalIndex1, x2);
        auto const partial2 = _mm512_permutex2var_ps(x0, index2, x1);
        auto const finalIndex2 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, kTupleCount > 11 ? 19 : kZeroIndex, kTupleCount > 12 ? 22 : kZeroIndex, kTupleCount > 13 ? 25 : kZeroIndex, kTupleCount > 14 ? 28 : kZeroIndex, kTupleCount > 15 ? 31 : kZeroIndex);
        out2.raw = _mm512_permutex2var_ps(partial2, finalIndex2, x2);
      }
      // clang-format on
    }
  }

  template <int N = kSize>
  static MOCHI_FORCE_INLINE void Store([[maybe_unused]] Scalar* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N == 1) {
      _mm_store_ss(ptr, _mm512_castps512_ps128(v.raw));
    } else if constexpr (N == 2) {
      _mm_storel_epi64(
          reinterpret_cast<__m128i*>(ptr), _mm_castps_si128(_mm512_castps512_ps128(v.raw)));
    } else if constexpr (N == 3) {
      _mm_mask_storeu_ps(
          ptr, static_cast<__mmask8>((uint32_t{1} << N) - 1), _mm512_castps512_ps128(v.raw));
    } else if constexpr (N == 4) {
      _mm_storeu_ps(ptr, _mm512_castps512_ps128(v.raw));
    } else if constexpr (N < 8) {
      _mm256_mask_storeu_ps(
          ptr, static_cast<__mmask8>((uint32_t{1} << N) - 1), _mm512_castps512_ps256(v.raw));
    } else if constexpr (N == 8) {
      _mm256_storeu_ps(ptr, _mm512_castps512_ps256(v.raw));
    } else if constexpr (N < kSize) {
      _mm512_mask_storeu_ps(ptr, LaneMask<N>(), v.raw); // AVX512F
    } else {
      _mm512_storeu_ps(ptr, v.raw); // AVX512F
    }
  }

  static MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    _mm512_mask_storeu_ps(ptr, LaneMask(n), v.raw); // AVX512F
  }

  MOCHI_FORCE_INLINE static int StoreSelected(Scalar* ptr, Simd condition, Simd values) {
    __mmask16 const mask = ToMask(condition);
    _mm512_mask_compressstoreu_ps(ptr, mask, values.raw); // AVX512F
    return _mm_popcnt_u32(mask);
  }

  template <int kTupleCount = kSize>
  MOCHI_FORCE_INLINE static void StoreTransposed(Scalar* ptr, Simd a, Simd b, Simd c) {
    static_assert(kTupleCount >= 1 && kTupleCount <= kSize, "Invalid kTupleCount");
    if constexpr (kTupleCount == 1) {
      Store<1>(ptr, a);
      Store<1>(ptr + 1, b);
      Store<1>(ptr + 2, c);
      return;
    }
    constexpr int kTotalCount = kTupleCount * 3;
    auto const ab0 = _mm512_permutex2var_ps(
        a.raw, _mm512_setr_epi32(0, 16, 0, 1, 17, 0, 2, 18, 0, 3, 19, 0, 4, 20, 0, 5), b.raw);
    auto const x0 = _mm512_permutex2var_ps(
        ab0, _mm512_setr_epi32(0, 1, 16, 3, 4, 17, 6, 7, 18, 9, 10, 19, 12, 13, 20, 15), c.raw);
    Store<Clamp(kTotalCount, 0, kSize)>(ptr, x0);
    if constexpr (kTotalCount > kSize) {
      auto const ab1 = _mm512_permutex2var_ps(
          a.raw, _mm512_setr_epi32(21, 0, 6, 22, 0, 7, 23, 0, 8, 24, 0, 9, 25, 0, 10, 26), b.raw);
      auto const x1 = _mm512_permutex2var_ps(
          ab1, _mm512_setr_epi32(0, 21, 2, 3, 22, 5, 6, 23, 8, 9, 24, 11, 12, 25, 14, 15), c.raw);
      Store<Clamp(kTotalCount - kSize, 0, kSize)>(ptr + kSize, x1);
    }
    if constexpr (kTotalCount > 2 * kSize) {
      auto const ab2 = _mm512_permutex2var_ps(
          a.raw,
          _mm512_setr_epi32(0, 11, 27, 0, 12, 28, 0, 13, 29, 0, 14, 30, 0, 15, 31, 0),
          b.raw);
      auto const x2 = _mm512_permutex2var_ps(
          ab2, _mm512_setr_epi32(26, 1, 2, 27, 4, 5, 28, 7, 8, 29, 10, 11, 30, 13, 14, 31), c.raw);
      Store<kTotalCount - 2 * kSize>(ptr + 2 * kSize, x2);
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Select(Simd mask, Simd a, Simd b) {
    return _mm512_mask_blend_ps(ToMask(mask), b.raw, a.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Sqrt(Simd v) {
    return _mm512_sqrt_ps(v.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd RcpApprox(Simd v) {
    return _mm512_rcp14_ps(v.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd RcpSqrtApprox(Simd v) {
    return _mm512_rsqrt14_ps(v.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd SignBitMask() {
    return _mm512_castsi512_ps(_mm512_set1_epi32(static_cast<int>(0x80000000u))); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Abs(Simd v) {
    return _mm512_abs_ps(v.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Min(Simd a, Simd b) {
    return _mm512_min_ps(a.raw, b.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Max(Simd a, Simd b) {
    return _mm512_max_ps(a.raw, b.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Floor(Simd a) {
    return _mm512_floor_ps(a.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd FastRound(Simd v) {
    return _mm512_roundscale_ps(v.raw, _MM_FROUND_TO_NEAREST_INT); // AVX512F
  }

#if MOCHI_ARCH_X64_SVML
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Cos(Simd a) {
    return _mm512_cos_ps(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Sin(Simd a) {
    return _mm512_sin_ps(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Tan(Simd a) {
    return _mm512_tan_ps(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ACos(Simd a) {
    return _mm512_acos_ps(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ASin(Simd a) {
    return _mm512_asin_ps(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd ATan(Simd a) {
    return _mm512_atan_ps(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Exp(Simd a) {
    return _mm512_exp_ps(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Ln(Simd a) {
    return _mm512_log_ps(a.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Tanh(Simd a) {
    return _mm512_tanh_ps(a.raw);
  }
#endif // MOCHI_ARCH_X64_SVML

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd MulAdd(Simd a, Simd b, Simd c) {
    return _mm512_fmadd_ps(a.raw, b.raw, c.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd MulSub(Simd a, Simd b, Simd c) {
    return _mm512_fmsub_ps(a.raw, b.raw, c.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NegMulAdd(Simd a, Simd b, Simd c) {
    return _mm512_fnmadd_ps(a.raw, b.raw, c.raw); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NegMulSub(Simd a, Simd b, Simd c) {
    return _mm512_fnmsub_ps(a.raw, b.raw, c.raw); // AVX512F
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMin(Simd a) {
    static_assert(N >= 2 && N <= kSize, "Unsupported N");
    using HalfT = Simd<Scalar, 8>;
    auto const lo = GetHalf<0>(a);
    if constexpr (N <= 8) {
      return HalfT::template HMin<N>(lo);
    } else {
      auto const hi = GetHalf<1>(a);
      if constexpr (N == 9) {
        return mochi::Min(HalfT::template HMin<8>(lo), HalfT::template Get<0>(hi));
      } else if constexpr (N == kSize) {
        return HalfT::template HMin<8>(HalfT::Min(lo, hi));
      } else {
        return _mm512_mask_reduce_min_ps(LaneMask<N>(), a.raw);
      }
    }
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMax(Simd a) {
    static_assert(N >= 2 && N <= kSize, "Unsupported N");
    using HalfT = Simd<Scalar, 8>;
    auto const lo = GetHalf<0>(a);
    if constexpr (N <= 8) {
      return HalfT::template HMax<N>(lo);
    } else {
      auto const hi = GetHalf<1>(a);
      if constexpr (N == 9) {
        return mochi::Max(HalfT::template HMax<8>(lo), HalfT::template Get<0>(hi));
      } else if constexpr (N == kSize) {
        return HalfT::template HMax<8>(HalfT::Max(lo, hi));
      } else {
        return _mm512_mask_reduce_max_ps(LaneMask<N>(), a.raw);
      }
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HSum(Simd a) {
    static_assert(N >= 2 && N <= kSize, "Unsupported N");
    using HalfT = Simd<Scalar, 8>;
    auto const lo = GetHalf<0>(a);
    if constexpr (N <= 8) {
      return HalfT::template HSum<N>(lo);
    } else {
      auto const hi = GetHalf<1>(a);
      if constexpr (N == 9) {
        return HalfT::template HSum<8>(lo) + HalfT::template Get<0>(hi);
      } else if constexpr (N == kSize) {
        return HalfT::template HSum<8>(lo + hi);
      } else {
        return _mm512_mask_reduce_add_ps(LaneMask<N>(), a.raw);
      }
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Dot(Simd a, Simd b) {
    static_assert(N >= 2 && N <= kSize, "Unsupported N");
    return Simd{HSum<N>(a * b)};
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return FromMask(_mm512_cmp_ps_mask(raw, rhs.raw, _CMP_LT_OQ)); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return FromMask(_mm512_cmp_ps_mask(raw, rhs.raw, _CMP_GT_OQ)); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return FromMask(_mm512_cmp_ps_mask(raw, rhs.raw, _CMP_LE_OQ)); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return FromMask(_mm512_cmp_ps_mask(raw, rhs.raw, _CMP_GE_OQ)); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    return FromMask(_mm512_cmp_ps_mask(a.raw, b.raw, _CMP_EQ_OQ)); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return FromMask(_mm512_cmp_ps_mask(a.raw, b.raw, _CMP_NEQ_UQ)); // AVX512F
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm512_setzero_ps(); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    return _mm512_cmp_ps_mask(raw, rhs.raw, _CMP_EQ_OQ) == static_cast<__mmask16>(0xFFFFu);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return _mm512_cmp_ps_mask(raw, rhs.raw, _CMP_NEQ_UQ) != 0;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    return _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_castps_si512(raw), _mm512_set1_epi32(-1))); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return _mm512_xor_ps(raw, SignBitMask().raw); // AVX512DQ
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return _mm512_add_ps(raw, rhs.raw); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return _mm512_sub_ps(raw, rhs.raw); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    return _mm512_mul_ps(raw, rhs.raw); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
    return _mm512_div_ps(raw, rhs.raw); // AVX512F
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return _mm512_and_ps(raw, rhs.raw); // AVX512DQ
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return _mm512_or_ps(raw, rhs.raw); // AVX512DQ
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return _mm512_xor_ps(raw, rhs.raw); // AVX512DQ
  }

 private:
  template <int kTupleCount, int kComponent>
  [[nodiscard]] static MOCHI_FORCE_INLINE __m512i LoadTransposeIndices() {
    constexpr int kZeroIndex = kTupleCount * 3;
    return _mm512_setr_epi32(
        kComponent,
        kTupleCount > 1 ? 3 + kComponent : kZeroIndex,
        kTupleCount > 2 ? 6 + kComponent : kZeroIndex,
        kTupleCount > 3 ? 9 + kComponent : kZeroIndex,
        kTupleCount > 4 ? 12 + kComponent : kZeroIndex,
        kTupleCount > 5 ? 15 + kComponent : kZeroIndex,
        kTupleCount > 6 ? 18 + kComponent : kZeroIndex,
        kTupleCount > 7 ? 21 + kComponent : kZeroIndex,
        kTupleCount > 8 ? 24 + kComponent : kZeroIndex,
        kTupleCount > 9 ? 27 + kComponent : kZeroIndex,
        kTupleCount > 10 ? 30 + kComponent : kZeroIndex,
        kTupleCount > 11 ? 33 + kComponent : kZeroIndex,
        kTupleCount > 12 ? 36 + kComponent : kZeroIndex,
        kTupleCount > 13 ? 39 + kComponent : kZeroIndex,
        kTupleCount > 14 ? 42 + kComponent : kZeroIndex,
        kTupleCount > 15 ? 45 + kComponent : kZeroIndex);
  }

  // Returns a mask selecting the lowest N lanes.
  template <int N>
  [[nodiscard]] static constexpr __mmask16 LaneMask() {
    static_assert(N >= 0 && N <= kSize);
    return static_cast<__mmask16>((uint32_t{1} << N) - 1);
  }

  // Returns a mask selecting the lowest n lanes.
  [[nodiscard]] static constexpr __mmask16 LaneMask(int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid lane count");
    return static_cast<__mmask16>((uint32_t{1} << n) - 1);
  }

  // Converts a canonical logical vector (all-zero or all-one lanes) to a mask.
  [[nodiscard]] static MOCHI_FORCE_INLINE __mmask16 ToMask(Simd a) {
    auto const bits = _mm512_castps_si512(a.raw);
    auto const mask = _mm512_movepi32_mask(bits); // AVX512DQ
    MOCHI_ASSERT_VERBOSE(
        _mm512_cmpeq_epi32_mask(bits, _mm512_movm_epi32(mask)) == LaneMask<kSize>(),
        "Expected a canonical logical mask");
    return mask;
  }

  // Expands a mask into a canonical logical vector (all-zero or all-one lanes).
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd FromMask(__mmask16 mask) {
    return _mm512_castsi512_ps(_mm512_movm_epi32(mask)); // AVX512DQ
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512
