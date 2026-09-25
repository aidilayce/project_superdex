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

#include "x64_simd_half_16_inl.h" // Must come before the definition of Simd<Half, 32>

#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512

namespace mochi {

/***********************************************************************************************
  Simd<Half, 32> — 512-bit AVX register holding 32 half-precision floating point values
*/
template <>
class Simd<Half, 32> {
 public:
  MOCHI_NATIVE_SIMD_IMPL_BOILERPLATE(Half, 32, __m512i);

  MOCHI_FORCE_INLINE explicit Simd(Scalar val)
      : raw(_mm512_set1_epi16(static_cast<short>(ReinterpretCast<uint16_t>(val)))) {}

  MOCHI_FORCE_INLINE Simd(
      Scalar a,
      Scalar b,
      Scalar c = Scalar{},
      Scalar d = Scalar{},
      Scalar e = Scalar{},
      Scalar f = Scalar{},
      Scalar g = Scalar{},
      Scalar h = Scalar{},
      Scalar i = Scalar{},
      Scalar j = Scalar{},
      Scalar k = Scalar{},
      Scalar l = Scalar{},
      Scalar m = Scalar{},
      Scalar n = Scalar{},
      Scalar o = Scalar{},
      Scalar p = Scalar{},
      Scalar q = Scalar{},
      Scalar r = Scalar{},
      Scalar s = Scalar{},
      Scalar t = Scalar{},
      Scalar u = Scalar{},
      Scalar v = Scalar{},
      Scalar w = Scalar{},
      Scalar x = Scalar{},
      Scalar y = Scalar{},
      Scalar z = Scalar{},
      Scalar aa = Scalar{},
      Scalar ab = Scalar{},
      Scalar ac = Scalar{},
      Scalar ad = Scalar{},
      Scalar ae = Scalar{},
      Scalar af = Scalar{})
      : raw(_mm512_set_epi16(
            static_cast<short>(ReinterpretCast<uint16_t>(af)),
            static_cast<short>(ReinterpretCast<uint16_t>(ae)),
            static_cast<short>(ReinterpretCast<uint16_t>(ad)),
            static_cast<short>(ReinterpretCast<uint16_t>(ac)),
            static_cast<short>(ReinterpretCast<uint16_t>(ab)),
            static_cast<short>(ReinterpretCast<uint16_t>(aa)),
            static_cast<short>(ReinterpretCast<uint16_t>(z)),
            static_cast<short>(ReinterpretCast<uint16_t>(y)),
            static_cast<short>(ReinterpretCast<uint16_t>(x)),
            static_cast<short>(ReinterpretCast<uint16_t>(w)),
            static_cast<short>(ReinterpretCast<uint16_t>(v)),
            static_cast<short>(ReinterpretCast<uint16_t>(u)),
            static_cast<short>(ReinterpretCast<uint16_t>(t)),
            static_cast<short>(ReinterpretCast<uint16_t>(s)),
            static_cast<short>(ReinterpretCast<uint16_t>(r)),
            static_cast<short>(ReinterpretCast<uint16_t>(q)),
            static_cast<short>(ReinterpretCast<uint16_t>(p)),
            static_cast<short>(ReinterpretCast<uint16_t>(o)),
            static_cast<short>(ReinterpretCast<uint16_t>(n)),
            static_cast<short>(ReinterpretCast<uint16_t>(m)),
            static_cast<short>(ReinterpretCast<uint16_t>(l)),
            static_cast<short>(ReinterpretCast<uint16_t>(k)),
            static_cast<short>(ReinterpretCast<uint16_t>(j)),
            static_cast<short>(ReinterpretCast<uint16_t>(i)),
            static_cast<short>(ReinterpretCast<uint16_t>(h)),
            static_cast<short>(ReinterpretCast<uint16_t>(g)),
            static_cast<short>(ReinterpretCast<uint16_t>(f)),
            static_cast<short>(ReinterpretCast<uint16_t>(e)),
            static_cast<short>(ReinterpretCast<uint16_t>(d)),
            static_cast<short>(ReinterpretCast<uint16_t>(c)),
            static_cast<short>(ReinterpretCast<uint16_t>(b)),
            static_cast<short>(ReinterpretCast<uint16_t>(a)))) {}

  MOCHI_FORCE_INLINE Simd(Simd<Half, 16> const& low, Simd<Half, 16> const& high)
      : raw(_mm512_inserti64x4(_mm512_castsi256_si512(low.raw), high.raw, 1)) {}

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd<Half, 16> GetHalf(Simd v) {
    static_assert(i == 0 || i == 1, "Index must be 0 or 1");
    if constexpr (i == 0) {
      return _mm512_castsi512_si256(v.raw);
    } else {
      return _mm512_extracti64x4_epi64(v.raw, 1);
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Zero() {
    return _mm512_setzero_si512();
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load([[maybe_unused]] Half const* ptr) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
      return Zero();
    } else if constexpr (N == 1) {
      return _mm512_zextsi128_si512(_mm_loadu_si16(ptr));
    } else if constexpr (N == 2) {
      return _mm512_zextsi128_si512(_mm_loadu_si32(ptr));
    } else if constexpr (N == 4) {
      return _mm512_zextsi128_si512(_mm_loadl_epi64(reinterpret_cast<__m128i const*>(ptr)));
    } else if constexpr (N < 8) {
      return _mm512_zextsi128_si512(
          _mm_maskz_loadu_epi16(static_cast<__mmask8>((uint32_t{1} << N) - 1), ptr));
    } else if constexpr (N == 8) {
      return _mm512_zextsi128_si512(_mm_loadu_si128(reinterpret_cast<__m128i const*>(ptr)));
    } else if constexpr (N < 16) {
      return _mm512_zextsi256_si512(
          _mm256_maskz_loadu_epi16(static_cast<__mmask16>((uint32_t{1} << N) - 1), ptr));
    } else if constexpr (N == 16) {
      return _mm512_zextsi256_si512(_mm256_loadu_si256(reinterpret_cast<__m256i const*>(ptr)));
    } else if constexpr (N < kSize) {
      __mmask32 constexpr kMask = static_cast<__mmask32>((uint64_t{1} << N) - 1);
      return _mm512_maskz_loadu_epi16(kMask, ptr);
    } else {
      return _mm512_loadu_si512(ptr);
    }
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Load(Half const* ptr, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    __mmask32 const mask =
        n == kSize ? static_cast<__mmask32>(-1) : static_cast<__mmask32>((uint32_t{1} << n) - 1);
    return _mm512_maskz_loadu_epi16(mask, ptr);
  }

  template <int N = kSize>
  static MOCHI_FORCE_INLINE void Store([[maybe_unused]] Half* ptr, [[maybe_unused]] Simd v) {
    static_assert(N >= 0 && N <= kSize);
    if constexpr (N == 0) {
    } else if constexpr (N == 1) {
      _mm_storeu_si16(ptr, _mm512_castsi512_si128(v.raw));
    } else if constexpr (N == 2) {
      _mm_storeu_si32(ptr, _mm512_castsi512_si128(v.raw));
    } else if constexpr (N == 4) {
      _mm_storel_epi64(reinterpret_cast<__m128i*>(ptr), _mm512_castsi512_si128(v.raw));
    } else if constexpr (N < 8) {
      _mm_mask_storeu_epi16(
          ptr, static_cast<__mmask8>((uint32_t{1} << N) - 1), _mm512_castsi512_si128(v.raw));
    } else if constexpr (N == 8) {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), _mm512_castsi512_si128(v.raw));
    } else if constexpr (N < 16) {
      _mm256_mask_storeu_epi16(
          ptr, static_cast<__mmask16>((uint32_t{1} << N) - 1), _mm512_castsi512_si256(v.raw));
    } else if constexpr (N == 16) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), _mm512_castsi512_si256(v.raw));
    } else if constexpr (N < kSize) {
      __mmask32 constexpr kMask = static_cast<__mmask32>((uint64_t{1} << N) - 1);
      _mm512_mask_storeu_epi16(ptr, kMask, v.raw);
    } else {
      _mm512_storeu_si512(ptr, v.raw);
    }
  }

  static MOCHI_FORCE_INLINE void Store(Half* ptr, Simd v, int n) {
    MOCHI_ASSERT_VERBOSE(n >= 0 && n <= kSize, "Invalid size parameter");
    __mmask32 const mask =
        n == kSize ? static_cast<__mmask32>(-1) : static_cast<__mmask32>((uint32_t{1} << n) - 1);
    _mm512_mask_storeu_epi16(ptr, mask, v.raw);
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

  [[nodiscard]] MOCHI_FORCE_INLINE Simd operator~() const {
    return _mm512_xor_si512(raw, _mm512_set1_epi32(-1));
  }

  // IEEE 754 float equality: +0 == -0, NaN != NaN (matches Simd<float> behavior)
  [[nodiscard]] MOCHI_FORCE_INLINE bool operator==(Simd rhs) const {
    auto const mask = EqualMask(*this, rhs);
    return _kortestc_mask32_u8(mask, mask) != 0;
  }

  [[nodiscard]] MOCHI_FORCE_INLINE bool operator!=(Simd rhs) const {
    auto const mask = NotEqualMask(*this, rhs);
    return _kortestz_mask32_u8(mask, mask) == 0;
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd Equal(Simd a, Simd b) {
    return FromMask(EqualMask(a, b));
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE Simd NotEqual(Simd a, Simd b) {
    return FromMask(NotEqualMask(a, b));
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AllTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    auto const mask = ToMask(v);
    if constexpr (N == kSize) {
      return _kortestc_mask32_u8(mask, mask) != 0;
    } else {
      constexpr auto kLanes = LaneMask<N>();
      return (mask & kLanes) == kLanes;
    }
  }

  template <int N = kSize>
  [[nodiscard]] static MOCHI_FORCE_INLINE bool AnyTrue(Simd v) {
    static_assert(N >= 1 && N <= kSize, "Unsupported N");
    auto const mask = ToMask(v);
    if constexpr (N == kSize) {
      return _kortestz_mask32_u8(mask, mask) == 0;
    } else {
      return (mask & LaneMask<N>()) != 0;
    }
  }

  template <int i>
  [[nodiscard]] static MOCHI_FORCE_INLINE Scalar Get(Simd v) {
    static_assert(i >= 0 && i < kSize, "Index out of range");
    constexpr int kQuarter = i / 8;
    constexpr int kLane = i % 8;
    if constexpr (kQuarter == 0) {
      return Simd<Half, 8>::template Get<kLane>(_mm512_castsi512_si128(v.raw));
    } else {
      return Simd<Half, 8>::template Get<kLane>(_mm512_extracti32x4_epi32(v.raw, kQuarter));
    }
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Scalar operator[](int i) const {
    MOCHI_ASSERT_VERBOSE(i >= 0 && i < kSize, "Index out of range.");
    auto const indices = _mm512_set1_epi16(static_cast<short>(i));
    auto const value = _mm512_permutexvar_epi16(indices, raw);
    return ReinterpretCast<Half>(
        static_cast<uint16_t>(_mm_extract_epi16(_mm512_castsi512_si128(value), 0)));
  }

 private:
  template <int kPredicate>
  [[nodiscard]] static MOCHI_FORCE_INLINE __mmask32 CompareMask(Simd a, Simd b) {
    __mmask32 const cmpLo = static_cast<__mmask32>(_mm512_cmp_ps_mask(
        _mm512_cvtph_ps(_mm512_castsi512_si256(a.raw)),
        _mm512_cvtph_ps(_mm512_castsi512_si256(b.raw)),
        kPredicate));
    __mmask32 const cmpHi = static_cast<__mmask32>(_mm512_cmp_ps_mask(
        _mm512_cvtph_ps(_mm512_extracti64x4_epi64(a.raw, 1)),
        _mm512_cvtph_ps(_mm512_extracti64x4_epi64(b.raw, 1)),
        kPredicate));
    return cmpLo | (cmpHi << 16);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE __mmask32 EqualMask(Simd a, Simd b) {
    return CompareMask<_CMP_EQ_OQ>(a, b);
  }

  [[nodiscard]] static MOCHI_FORCE_INLINE __mmask32 NotEqualMask(Simd a, Simd b) {
    return CompareMask<_CMP_NEQ_UQ>(a, b);
  }

  // Returns a mask selecting the lowest N lanes.
  template <int N>
  [[nodiscard]] static constexpr __mmask32 LaneMask() {
    static_assert(N >= 0 && N <= kSize);
    return static_cast<__mmask32>((uint64_t{1} << N) - 1);
  }

  // Converts a canonical logical vector (all-zero or all-one lanes) to a mask.
  [[nodiscard]] static MOCHI_FORCE_INLINE __mmask32 ToMask(Simd a) {
    auto const mask = _mm512_movepi16_mask(a.raw);
    MOCHI_ASSERT_VERBOSE(
        _mm512_cmpeq_epi16_mask(a.raw, _mm512_movm_epi16(mask)) == LaneMask<kSize>(),
        "Expected a canonical logical mask");
    return mask;
  }

  // Expands a mask into a canonical logical vector (all-zero or all-one lanes).
  [[nodiscard]] static MOCHI_FORCE_INLINE Simd FromMask(__mmask32 mask) {
    return _mm512_movm_epi16(mask);
  }
};

} // namespace mochi

#endif // MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512
