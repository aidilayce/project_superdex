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

#include "simd_test_half.h"

#include <type_traits>

#if MOCHI_HAS_SIMD_HALF

using namespace mochi;
using namespace mochi::simd_half_test;

MOCHI_SIMD_TEST_SCALAR_CONVERSIONS_TO_HALF(Vec32h);

TEST(Vec32h, TypeProperties) {
  static_assert(Vec32h::kIsSupported);
  static_assert(Vec32h::kSize == 32);
  static_assert(Vec32h::size() == 32);
  static_assert(sizeof(Vec32h) == sizeof(Half) * Vec32h::kSize);
  static_assert(alignof(Vec32h) == alignof(typename Vec32h::NativeType));
  static_assert(std::is_same_v<Vec32h::Scalar, Half>);
  static_assert(std::is_trivially_copyable_v<Vec32h>);
  static_assert(!Vec32h::kIsEmulated);
  static_assert(Vec32h::kIsComposite == !MOCHI_ARCH_X64_AVX512);
}

MOCHI_DEFINE_SIMD_HALF_COMMON_TESTS(Vec32h)

TEST(Vec32h, HalfConstructor) {
  auto const values = GetTestValues(Vec32h::kSize);
  auto const v = Load<Vec32h>(values.data());
  EXPECT_EQ(v, (Vec32h{GetHalf<0>(v), GetHalf<1>(v)}));
}

TEST(Vec32h, ScalarLaneConstructor) {
  auto const v = GetTestValues(Vec32h::kSize);
  Vec32h const values{v[0],  v[1],  v[2],  v[3],  v[4],  v[5],  v[6],  v[7],  v[8],  v[9],  v[10],
                      v[11], v[12], v[13], v[14], v[15], v[16], v[17], v[18], v[19], v[20], v[21],
                      v[22], v[23], v[24], v[25], v[26], v[27], v[28], v[29], v[30], v[31]};
  for (int i = 0; i < Vec32h::kSize; ++i) {
    EXPECT_EQ(v[i], values[i]);
  }
}

TEST(Vec32h, HalfValues) {
  auto const values = GetTestValues(Vec32h::kSize);
  auto const vector = Load<Vec32h>(values.data());
  auto const low = GetHalf<0>(vector);
  auto const high = GetHalf<1>(vector);
  for (int i = 0; i < Vec16h::kSize; ++i) {
    EXPECT_EQ(values[i], low[i]);
    EXPECT_EQ(values[i + Vec16h::kSize], high[i]);
  }
  EXPECT_EQ(vector, (Vec32h{low, high}));
}

#else

TEST(Vec32h, DISABLED_RequiresSimdHalf) {
  GTEST_SKIP() << "Half SIMD is unavailable in this build";
}

#endif // MOCHI_HAS_SIMD_HALF
