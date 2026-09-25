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
  Simd<int64_t, 8>
*/
template <>
class Simd<int64_t, 8> {
 public:
  static_assert(sizeof(int64_t) == sizeof(long long));

  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(int64_t, 8, __m512i);

  Simd(
      int64_t a,
      int64_t b,
      int64_t c = 0,
      int64_t d = 0,
      int64_t e = 0,
      int64_t f = 0,
      int64_t g = 0,
      int64_t h = 0)
      : raw(_mm512_set_epi64(h, g, f, e, d, c, b, a)) {}

  template <class U, MOCHI_REQUIRES_NON_BOOL_SCALAR(U, Scalar)>
  Simd(U a) : raw(_mm512_set1_epi64(static_cast<long long>(a))) {}

  Simd(Simd<int64_t, 4> const& low, Simd<int64_t, 4> const& high)
      : raw(_mm512_inserti64x4(_mm512_castsi256_si512(low.raw), high.raw, 1)) {}

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar Get(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    return v[i];
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Scalar operator[](int i) const {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range");
#if MOCHI_COMPILER_MSVC
    return raw.m512i_i64[i];
#else
    return static_cast<Scalar>(raw[i]);
#endif
  }

  template <int iHalf>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd<int64_t, 4> GetHalf(Simd a) {
    static_assert(iHalf == 0 || iHalf == 1);
    if constexpr (iHalf == 0) {
      return _mm512_castsi512_si256(a.raw);
    } else {
      return _mm512_extracti64x4_epi64(a.raw, 1);
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    auto const mask = ToMask(v);
    if constexpr (N == kSize) {
      return _kortestc_mask8_u8(mask, mask) != 0;
    } else {
      constexpr auto kLanes = LaneMask<N>();
      return (mask & kLanes) == kLanes;
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Scalar const* p) {
    return Simd{*p};
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Broadcast(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    if constexpr (i == 0) {
      return _mm512_broadcastq_epi64(_mm512_castsi512_si128(v.raw));
    } else {
      constexpr int kLane = i % 2;
      constexpr int kGroup = i / 2;
      auto const group =
          _mm512_shuffle_i64x2(v.raw, v.raw, _MM_SHUFFLE(kGroup, kGroup, kGroup, kGroup));
      return _mm512_shuffle_epi32(
          group,
          static_cast<_MM_PERM_ENUM>(
              kLane == 0 ? _MM_SHUFFLE(1, 0, 1, 0) : _MM_SHUFFLE(3, 2, 3, 2)));
    }
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMin(Simd a) {
    static_assert(N >= 2 && N <= kSize, "Unsupported N");
    using HalfT = Simd<Scalar, 4>;
    auto const lo = GetHalf<0>(a);
    if constexpr (N <= 4) {
      return HalfT::template HMin<N>(lo);
    } else {
      auto const hi = GetHalf<1>(a);
      if constexpr (N == 5) {
        return mochi::Min(HalfT::template HMin<4>(lo), HalfT::template Get<0>(hi));
      } else if constexpr (N == kSize) {
        return HalfT::template HMin<4>(HalfT::Min(lo, hi));
      } else {
        return _mm512_mask_reduce_min_epi64(LaneMask<N>(), a.raw);
      }
    }
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HMax(Simd a) {
    static_assert(N >= 2 && N <= kSize, "Unsupported N");
    using HalfT = Simd<Scalar, 4>;
    auto const lo = GetHalf<0>(a);
    if constexpr (N <= 4) {
      return HalfT::template HMax<N>(lo);
    } else {
      auto const hi = GetHalf<1>(a);
      if constexpr (N == 5) {
        return mochi::Max(HalfT::template HMax<4>(lo), HalfT::template Get<0>(hi));
      } else if constexpr (N == kSize) {
        return HalfT::template HMax<4>(HalfT::Max(lo, hi));
      } else {
        return _mm512_mask_reduce_max_epi64(LaneMask<N>(), a.raw);
      }
    }
  }

  template <int N>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar HSum(Simd a) {
    static_assert(N >= 2 && N <= kSize, "Unsupported N");
    using HalfT = Simd<Scalar, 4>;
    auto const lo = GetHalf<0>(a);
    if constexpr (N <= 4) {
      return HalfT::template HSum<N>(lo);
    } else {
      auto const hi = GetHalf<1>(a);
      if constexpr (N == 5) {
        return HalfT::template HSum<4>(lo) + HalfT::template Get<0>(hi);
      } else if constexpr (N == kSize) {
        return HalfT::template HSum<4>(lo + hi);
      } else {
        return _mm512_mask_reduce_add_epi64(LaneMask<N>(), a.raw);
      }
    }
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Scalar const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Zero();
    } else if constexpr (N == 1) {
      return _mm512_zextsi128_si512(_mm_loadl_epi64(reinterpret_cast<__m128i const*>(ptr)));
    } else if constexpr (N == 2) {
      return _mm512_zextsi128_si512(_mm_loadu_si128(reinterpret_cast<__m128i const*>(ptr)));
    } else if constexpr (N < 4) {
      return _mm512_zextsi256_si512(
          _mm256_maskz_loadu_epi64(static_cast<__mmask8>((uint32_t{1} << N) - 1), ptr));
    } else if constexpr (N == 4) {
      return _mm512_zextsi256_si512(_mm256_loadu_si256(reinterpret_cast<__m256i const*>(ptr)));
    } else if constexpr (N < kSize) {
      return _mm512_maskz_loadu_epi64(LaneMask<N>(), ptr);
    } else {
      return _mm512_loadu_si512(ptr);
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Scalar const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    auto const mask = static_cast<__mmask8>((uint32_t{1} << n) - 1);
    return _mm512_maskz_loadu_epi64(mask, ptr);
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
    if constexpr (kTupleCount <= 5) {
      auto const index0 = LoadTransposeIndices<kTupleCount, 0>();
      auto const index1 = LoadTransposeIndices<kTupleCount, 1>();
      auto const index2 = LoadTransposeIndices<kTupleCount, 2>();
      if constexpr (kTupleCount <= 2) {
        out0.raw = _mm512_permutexvar_epi64(index0, x0);
        out1.raw = _mm512_permutexvar_epi64(index1, x0);
        out2.raw = _mm512_permutexvar_epi64(index2, x0);
      } else {
        auto const x1 = Load<kTotalCount - kSize>(ptr + kSize).raw;
        out0.raw = _mm512_permutex2var_epi64(x0, index0, x1);
        out1.raw = _mm512_permutex2var_epi64(x0, index1, x1);
        out2.raw = _mm512_permutex2var_epi64(x0, index2, x1);
      }
    } else {
      auto const x1 = Load<kSize>(ptr + kSize).raw;
      constexpr int kCount2 = kTotalCount - 2 * kSize;
      auto const x2 = Load<kCount2>(ptr + 2 * kSize).raw;
      auto const index0 = _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 0, 0);
      auto const index1 = _mm512_setr_epi64(1, 4, 7, 10, 13, 0, 0, 0);
      auto const index2 = _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0);
      if constexpr (kTupleCount == 6) {
        out0.raw = _mm512_maskz_permutex2var_epi64(LaneMask<kTupleCount>(), x0, index0, x1);
        constexpr int kZeroIndex = kSize + kCount2;
        auto const partial1 = _mm512_permutex2var_epi64(x0, index1, x1);
        auto const finalIndex1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 8, kZeroIndex, kZeroIndex);
        out1.raw = _mm512_permutex2var_epi64(partial1, finalIndex1, x2);
        auto const partial2 = _mm512_permutex2var_epi64(x0, index2, x1);
        auto const finalIndex2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 9, kZeroIndex, kZeroIndex);
        out2.raw = _mm512_permutex2var_epi64(partial2, finalIndex2, x2);
      } else {
        constexpr int kZeroIndex = kSize + kCount2;
        auto const partial0 = _mm512_permutex2var_epi64(x0, index0, x1);
        auto const finalIndex0 = _mm512_setr_epi64(
            0, 1, 2, 3, 4, 5, kTupleCount > 6 ? 10 : kZeroIndex, kTupleCount > 7 ? 13 : kZeroIndex);
        out0.raw = _mm512_permutex2var_epi64(partial0, finalIndex0, x2);
        auto const partial1 = _mm512_permutex2var_epi64(x0, index1, x1);
        auto const finalIndex1 = _mm512_setr_epi64(
            0, 1, 2, 3, 4, 8, kTupleCount > 6 ? 11 : kZeroIndex, kTupleCount > 7 ? 14 : kZeroIndex);
        out1.raw = _mm512_permutex2var_epi64(partial1, finalIndex1, x2);
        auto const partial2 = _mm512_permutex2var_epi64(x0, index2, x1);
        auto const finalIndex2 = _mm512_setr_epi64(
            0, 1, 2, 3, 4, 9, kTupleCount > 6 ? 12 : kZeroIndex, kTupleCount > 7 ? 15 : kZeroIndex);
        out2.raw = _mm512_permutex2var_epi64(partial2, finalIndex2, x2);
      }
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Min(Simd a, Simd b) {
    return _mm512_min_epi64(a.raw, b.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Max(Simd a, Simd b) {
    return _mm512_max_epi64(a.raw, b.raw);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Select(Simd mask, Simd a, Simd b) {
    return _mm512_mask_blend_epi64(ToMask(mask), b.raw, a.raw);
  }

  template <int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Shuffle(Simd a) {
    static_assert(
        x0 >= 0 && x0 < kSize && x1 >= 0 && x1 < kSize && x2 >= 0 && x2 < kSize && x3 >= 0 &&
            x3 < kSize && x4 >= 0 && x4 < kSize && x5 >= 0 && x5 < kSize && x6 >= 0 && x6 < kSize &&
            x7 >= 0 && x7 < kSize,
        "Invalid index");
    if constexpr (
        x0 == 0 && x1 == 1 && x2 == 2 && x3 == 3 && x4 == 4 && x5 == 5 && x6 == 6 && x7 == 7) {
      return a;
    } else {
      auto const indices = _mm512_setr_epi64(x0, x1, x2, x3, x4, x5, x6, x7);
      return _mm512_permutexvar_epi64(indices, a.raw);
    }
  }

  template <int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Shuffle(Simd a, Simd b) {
    static_assert(
        x0 >= 0 && x0 < kSize && x1 >= 0 && x1 < kSize && x2 >= 0 && x2 < kSize && x3 >= 0 &&
            x3 < kSize && x4 >= 0 && x4 < kSize && x5 >= 0 && x5 < kSize && x6 >= 0 && x6 < kSize &&
            x7 >= 0 && x7 < kSize,
        "Invalid index");
    if constexpr (
        x0 == 0 && x1 == 1 && x2 == 2 && x3 == 3 && x4 == 0 && x5 == 1 && x6 == 2 && x7 == 3) {
      return _mm512_inserti64x4(_mm512_castsi256_si512(GetHalf<0>(a).raw), GetHalf<0>(b).raw, 1);
    } else {
      auto const indices =
          _mm512_setr_epi64(x0, x1, x2, x3, x4 + kSize, x5 + kSize, x6 + kSize, x7 + kSize);
      return _mm512_permutex2var_epi64(a.raw, indices, b.raw);
    }
  }

  template <int N = kSize>
  static MOCHI_FORCE_INLINE void Store([[maybe_unused]] Scalar* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N == 1) {
      _mm_storel_epi64(reinterpret_cast<__m128i*>(ptr), _mm512_castsi512_si128(v.raw));
    } else if constexpr (N == 2) {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), _mm512_castsi512_si128(v.raw));
    } else if constexpr (N < 4) {
      _mm256_mask_storeu_epi64(
          ptr, static_cast<__mmask8>((uint32_t{1} << N) - 1), _mm512_castsi512_si256(v.raw));
    } else if constexpr (N == 4) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), _mm512_castsi512_si256(v.raw));
    } else if constexpr (N < kSize) {
      _mm512_mask_storeu_epi64(ptr, LaneMask<N>(), v.raw);
    } else {
      _mm512_storeu_si512(ptr, v.raw);
    }
  }

  static MOCHI_FORCE_INLINE void Store(Scalar* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    auto const mask = static_cast<__mmask8>((uint32_t{1} << n) - 1);
    _mm512_mask_storeu_epi64(ptr, mask, v.raw);
  }

  MOCHI_FORCE_INLINE static int StoreSelected(Scalar* ptr, Simd condition, Simd values) {
    auto const mask = ToMask(condition);
    _mm512_mask_compressstoreu_epi64(ptr, mask, values.raw);
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
    auto const ab0 =
        _mm512_permutex2var_epi64(a.raw, _mm512_setr_epi64(0, 8, 0, 1, 9, 0, 2, 10), b.raw);
    auto const x0 =
        _mm512_permutex2var_epi64(ab0, _mm512_setr_epi64(0, 1, 8, 3, 4, 9, 6, 7), c.raw);
    Store<Clamp(kTotalCount, 0, kSize)>(ptr, x0);
    if constexpr (kTotalCount > kSize) {
      auto const ab1 =
          _mm512_permutex2var_epi64(a.raw, _mm512_setr_epi64(0, 3, 11, 0, 4, 12, 0, 5), b.raw);
      auto const x1 =
          _mm512_permutex2var_epi64(ab1, _mm512_setr_epi64(10, 1, 2, 11, 4, 5, 12, 7), c.raw);
      Store<Clamp(kTotalCount - kSize, 0, kSize)>(ptr + kSize, x1);
    }
    if constexpr (kTotalCount > 2 * kSize) {
      auto const ab2 =
          _mm512_permutex2var_epi64(a.raw, _mm512_setr_epi64(13, 0, 6, 14, 0, 7, 15, 0), b.raw);
      auto const x2 =
          _mm512_permutex2var_epi64(ab2, _mm512_setr_epi64(0, 13, 2, 3, 14, 5, 6, 15), c.raw);
      Store<kTotalCount - 2 * kSize>(ptr + 2 * kSize, x2);
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm512_setzero_si512();
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<(Simd rhs) const {
    return FromMask(_mm512_cmp_epi64_mask(raw, rhs.raw, _MM_CMPINT_LT));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>(Simd rhs) const {
    return FromMask(_mm512_cmp_epi64_mask(raw, rhs.raw, _MM_CMPINT_GT));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<=(Simd rhs) const {
    return FromMask(_mm512_cmp_epi64_mask(raw, rhs.raw, _MM_CMPINT_LE));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator>=(Simd rhs) const {
    return FromMask(_mm512_cmp_epi64_mask(raw, rhs.raw, _MM_CMPINT_GE));
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    return FromMask(_mm512_cmp_epi64_mask(a.raw, b.raw, _MM_CMPINT_EQ));
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return FromMask(_mm512_cmp_epi64_mask(a.raw, b.raw, _MM_CMPINT_NE));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    return _mm512_cmp_epi64_mask(raw, rhs.raw, _MM_CMPINT_EQ) == __mmask8{0xFF};
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    return _mm512_cmp_epi64_mask(raw, rhs.raw, _MM_CMPINT_NE) != 0;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    return _mm512_xor_si512(raw, _mm512_set1_epi64(-1));
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-() const {
    return _mm512_sub_epi64(_mm512_setzero_si512(), raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator+(Simd rhs) const {
    return _mm512_add_epi64(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator-(Simd rhs) const {
    return _mm512_sub_epi64(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator*(Simd rhs) const {
    return _mm512_mullo_epi64(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator/(Simd rhs) const {
#if MOCHI_ARCH_X64_SVML
    return _mm512_div_epi64(raw, rhs.raw);
#else
    return Simd{
        Get<0>(*this) / Get<0>(rhs),
        Get<1>(*this) / Get<1>(rhs),
        Get<2>(*this) / Get<2>(rhs),
        Get<3>(*this) / Get<3>(rhs),
        Get<4>(*this) / Get<4>(rhs),
        Get<5>(*this) / Get<5>(rhs),
        Get<6>(*this) / Get<6>(rhs),
        Get<7>(*this) / Get<7>(rhs)};
#endif
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator&(Simd rhs) const {
    return _mm512_and_si512(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator|(Simd rhs) const {
    return _mm512_or_si512(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator^(Simd rhs) const {
    return _mm512_xor_si512(raw, rhs.raw);
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator<<(int rhs) const {
    return _mm512_sll_epi64(raw, _mm_cvtsi32_si128(rhs));
  }

  template <int kShift>
  [[nodiscard]] MOCHI_FORCE_INLINE static Simd ShiftRight(Simd a) {
    static_assert(kShift >= 0 && kShift < 64, "Shift amount out-of-range");
    if constexpr (kShift == 0) {
      return a;
    } else {
      return _mm512_srai_epi64(a.raw, kShift);
    }
  }

 private:
  template <int kTupleCount, int kComponent>
  [[nodiscard]] static MOCHI_FORCE_INLINE __m512i LoadTransposeIndices() {
    constexpr int kZeroIndex = kTupleCount * 3;
    return _mm512_setr_epi64(
        kComponent,
        kTupleCount > 1 ? 3 + kComponent : kZeroIndex,
        kTupleCount > 2 ? 6 + kComponent : kZeroIndex,
        kTupleCount > 3 ? 9 + kComponent : kZeroIndex,
        kTupleCount > 4 ? 12 + kComponent : kZeroIndex,
        kTupleCount > 5 ? 15 + kComponent : kZeroIndex,
        kTupleCount > 6 ? 18 + kComponent : kZeroIndex,
        kTupleCount > 7 ? 21 + kComponent : kZeroIndex);
  }

  // Returns a mask selecting the lowest N lanes.
  template <int N>
  [[nodiscard]] static constexpr __mmask8 LaneMask() {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == kSize) {
      return __mmask8{0xFF};
    } else {
      return static_cast<__mmask8>((uint32_t{1} << N) - 1);
    }
  }

  // Converts a canonical logical vector (all-zero or all-one lanes) to a mask.
  [[nodiscard]] static MOCHI_FORCE_INLINE __mmask8 ToMask(Simd a) {
    auto const mask = _mm512_movepi64_mask(a.raw);
    MOCHI_ASSERT_VERBOSE(
        _mm512_cmpeq_epi64_mask(a.raw, _mm512_movm_epi64(mask)) == LaneMask<kSize>(),
        "Expected a canonical logical mask");
    return mask;
  }

  // Expands a mask into a canonical logical vector (all-zero or all-one lanes).
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd FromMask(__mmask8 mask) {
    return _mm512_movm_epi64(mask);
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512
