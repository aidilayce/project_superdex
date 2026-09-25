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

#include "simd_test.h"

#include <mochi_core/utils/half.h>

#include <cmath>
#include <iterator>
#include <limits>

using namespace mochi;
using namespace mochi::simd_test;

template <typename To, typename From, int N>
static void TestSimdStaticCastTo(Simd<From, N> const& from) {
  if constexpr (Simd<To, N>::kIsSupported) {
    auto result = StaticCast<Simd<To, N>>(from);
    for (int i = 0; i < N; ++i) {
      To const expected = static_cast<To>(from[i]);
      To const actual = result[i];
      if constexpr (std::is_floating_point_v<To> || IsHalf<To>) {
        if (std::isnan(static_cast<double>(expected))) {
          EXPECT_TRUE(std::isnan(static_cast<double>(actual)));
        } else {
          EXPECT_EQ(expected, actual);
          if (static_cast<double>(expected) == 0.0) {
            EXPECT_EQ(
                std::signbit(static_cast<double>(expected)),
                std::signbit(static_cast<double>(actual)));
          }
        }
      } else {
        EXPECT_EQ(expected, actual);
      }
    }
  }
}

template <typename From, int N>
static void TestSimdStaticCastToEach(Simd<From, N> const& from) {
  // Cast it to every other Simd type of the same size N
  TestSimdStaticCastTo<double>(from);
  TestSimdStaticCastTo<float>(from);
  TestSimdStaticCastTo<int>(from);
  TestSimdStaticCastTo<int64_t>(from);
  if constexpr (Simd<Half, N>::kIsSupported) {
    TestSimdStaticCastTo<Half>(from);
  }
}

// Like TestSimdStaticCastToEach, but skips floating-point to integer conversions where
// numeric_limits<From>::max() exceeds the target integer range (undefined behavior in C++).
template <typename From, int N>
static void TestSimdStaticCastToEachLimits(Simd<From, N> const& from) {
  TestSimdStaticCastTo<double>(from);
  TestSimdStaticCastTo<float>(from);
  if constexpr (!std::is_floating_point_v<From>) {
    TestSimdStaticCastTo<int>(from);
    TestSimdStaticCastTo<int64_t>(from);
  }
}

template <typename From, int N>
static void TestSimdStaticCastFrom() {
  From kTestValues[N] = {};
  for (int i = 0; i < N; ++i) {
    kTestValues[i] = static_cast<From>(i + 1);
  }

  auto from = Load<Simd<From, N>>(kTestValues);
  TestSimdStaticCastToEach(from);

  From signedValues[N] = {};
  for (int i = 0; i < N; ++i) {
    if constexpr (IsHalf<From>) {
      signedValues[i] = StaticCast<Half>((i % 2 == 0 ? -1.0f : 1.0f) * (i + 0.5f));
    } else if constexpr (std::is_floating_point_v<From>) {
      signedValues[i] = static_cast<From>((i % 2 == 0 ? -1.0 : 1.0) * (i + 0.5));
    } else {
      signedValues[i] = static_cast<From>(i % 3 == 0 ? 0 : i % 2 == 0 ? -i : i);
    }
  }
  TestSimdStaticCastToEach(Load<Simd<From, N>>(signedValues));

  // Repeat with numeric limits, skipping conversions that would be undefined behavior
  if constexpr (IsHalf<From>) {
    from = Simd<From, N>{kHalfMin, kHalfMax};
  } else {
    from = Simd<From, N>{std::numeric_limits<From>::min(), std::numeric_limits<From>::max()};
  }
  TestSimdStaticCastToEachLimits(from);
}

TEST(Vec2d, StaticCast) {
  TestSimdStaticCastFrom<double, 2>();
}

TEST(Vec2l, StaticCast) {
  TestSimdStaticCastFrom<int64_t, 2>();
}

TEST(Vec4d, StaticCast) {
  TestSimdStaticCastFrom<double, 4>();
}

TEST(Vec4f, StaticCast) {
  TestSimdStaticCastFrom<float, 4>();
}

TEST(Vec4i, StaticCast) {
  TestSimdStaticCastFrom<int, 4>();
}

TEST(Vec4l, StaticCast) {
  TestSimdStaticCastFrom<int64_t, 4>();
}

TEST(Vec8d, StaticCast) {
  TestSimdStaticCastFrom<double, 8>();
}

TEST(Vec8f, StaticCast) {
  TestSimdStaticCastFrom<float, 8>();
}

TEST(Vec8i, StaticCast) {
  TestSimdStaticCastFrom<int, 8>();
}

TEST(Vec8l, StaticCast) {
  TestSimdStaticCastFrom<int64_t, 8>();
}

TEST(Vec12d, StaticCast) {
  TestSimdStaticCastFrom<double, 12>();
}

TEST(Vec12f, StaticCast) {
  TestSimdStaticCastFrom<float, 12>();
}

TEST(Vec12i, StaticCast) {
  TestSimdStaticCastFrom<int, 12>();
}

TEST(Vec12l, StaticCast) {
  TestSimdStaticCastFrom<int64_t, 12>();
}

TEST(Vec16d, StaticCast) {
  TestSimdStaticCastFrom<double, 16>();
}

TEST(Vec16f, StaticCast) {
  TestSimdStaticCastFrom<float, 16>();
}

TEST(Vec16i, StaticCast) {
  TestSimdStaticCastFrom<int, 16>();
}

TEST(Vec16l, StaticCast) {
  TestSimdStaticCastFrom<int64_t, 16>();
}

#if MOCHI_HAS_SIMD_HALF

TEST(Vec8h, StaticCast) {
  TestSimdStaticCastFrom<Half, 8>();
}
TEST(Vec16h, StaticCast) {
  TestSimdStaticCastFrom<Half, 16>();
}
TEST(Vec24h, StaticCast) {
  TestSimdStaticCastFrom<Half, 24>();
}
TEST(Vec32h, StaticCast) {
  TestSimdStaticCastFrom<Half, 32>();
}
#endif // MOCHI_HAS_SIMD_HALF

TEST(Vec6d, StaticCast) {
  TestSimdStaticCastFrom<double, 6>();
}

TEST(Vec6l, StaticCast) {
  TestSimdStaticCastFrom<int64_t, 6>();
}

TEST(Vec28f, StaticCast) {
  TestSimdStaticCastFrom<float, 28>();
}

TEST(Vec28i, StaticCast) {
  TestSimdStaticCastFrom<int, 28>();
}

TEST(Vec32f, StaticCast) {
  TestSimdStaticCastFrom<float, 32>();
}

TEST(Vec32i, StaticCast) {
  TestSimdStaticCastFrom<int, 32>();
}

TEST(SimdStaticCast, FloatingSpecialValuesAndSafeIntegerBoundaries) {
  float constexpr kFloatPattern[] = {
      0.0f,
      -0.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      16777215.0f,
      -16777215.0f,
      65504.0f};
  double constexpr kDoublePattern[] = {
      0.0,
      -0.0,
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
      2147483647.0,
      -2147483648.0,
      65504.0};
  float floatValues[32];
  double doubleValues[32];
  for (int i = 0; i < 32; ++i) {
    floatValues[i] = kFloatPattern[i % std::size(kFloatPattern)];
    doubleValues[i] = kDoublePattern[i % std::size(kDoublePattern)];
  }

  auto const floats = Load<Simd<float, 32>>(floatValues);
  TestSimdStaticCastTo<float>(floats);
  TestSimdStaticCastTo<double>(floats);
#if MOCHI_HAS_SIMD_HALF
  TestSimdStaticCastTo<Half>(floats);
#endif
  auto const doubles = Load<Simd<double, 32>>(doubleValues);
  TestSimdStaticCastTo<double>(doubles);
  TestSimdStaticCastTo<float>(doubles);
#if MOCHI_HAS_SIMD_HALF
  TestSimdStaticCastTo<Half>(doubles);
#endif

  for (int i = 0; i < 32; ++i) {
    floatValues[i] = static_cast<float>((i % 2 == 0 ? 1 : -1) * (i * 65537));
    doubleValues[i] = static_cast<double>((i % 2 == 0 ? 1 : -1) * (i * 65537));
  }
  TestSimdStaticCastToEach(Load<Simd<float, 32>>(floatValues));
  TestSimdStaticCastToEach(Load<Simd<double, 32>>(doubleValues));
}
