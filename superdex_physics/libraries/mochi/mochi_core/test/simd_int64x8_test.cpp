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

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

using namespace mochi;
using namespace mochi::simd_test;

static_assert(std::is_trivially_copyable_v<Vec8l>);

MOCHI_SIMD_TEST_SCALAR_CONVERSIONS(Vec8l);

namespace {

template <int I = 0>
void ExpectCompileTimeBroadcasts(Vec8l source, std::array<int64_t, Vec8l::kSize> const& values) {
  if constexpr (I < Vec8l::kSize) {
    auto const broadcast = Broadcast<I>(source);
    for (int lane = 0; lane < Vec8l::kSize; ++lane) {
      EXPECT_EQ(values[I], broadcast[lane]);
    }
    ExpectCompileTimeBroadcasts<I + 1>(source, values);
  }
}

template <int I = 0>
void ExpectIsTrueLanes(Vec8l mask, std::array<bool, Vec8l::kSize> const& expected) {
  if constexpr (I < Vec8l::kSize) {
    EXPECT_EQ(expected[I], IsTrue<I>(mask) != 0);
    ExpectIsTrueLanes<I + 1>(mask, expected);
  }
}

template <int Shift = 0>
void ExpectAllShifts(Vec8l values) {
  auto const shiftedLeft = values << Shift;
  auto const shiftedRight = ShiftRight<Shift>(values);
  for (int i = 0; i < Vec8l::kSize; ++i) {
    auto const expectedLeft = std::bit_cast<int64_t>(static_cast<uint64_t>(values[i]) << Shift);
    EXPECT_EQ(expectedLeft, shiftedLeft[i]);
    EXPECT_EQ(values[i] >> Shift, shiftedRight[i]);
  }
  if constexpr (Shift + 1 < 8 * sizeof(Vec8l::Scalar)) {
    ExpectAllShifts<Shift + 1>(values);
  }
}

#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512
template <int Offset = 0>
void ExpectShuffleRotations(Vec8l a, Vec8l b) {
  auto const oneSource = Vec8l::Shuffle<
      (Offset + 0) % 8,
      (Offset + 1) % 8,
      (Offset + 2) % 8,
      (Offset + 3) % 8,
      (Offset + 4) % 8,
      (Offset + 5) % 8,
      (Offset + 6) % 8,
      (Offset + 7) % 8>(a);
  auto const twoSource = Vec8l::Shuffle<
      (Offset + 0) % 8,
      (Offset + 1) % 8,
      (Offset + 2) % 8,
      (Offset + 3) % 8,
      (Offset + 4) % 8,
      (Offset + 5) % 8,
      (Offset + 6) % 8,
      (Offset + 7) % 8>(a, b);
  for (int i = 0; i < Vec8l::kSize; ++i) {
    int const sourceIndex = (Offset + i) % Vec8l::kSize;
    EXPECT_EQ(a[sourceIndex], oneSource[i]);
    EXPECT_EQ(i < 4 ? a[sourceIndex] : b[sourceIndex], twoSource[i]);
  }
  if constexpr (Offset + 1 < Vec8l::kSize) {
    ExpectShuffleRotations<Offset + 1>(a, b);
  }
}
#endif

template <int I = 0>
void ExpectCompileTimeLanes(Vec8l v, std::array<int64_t, Vec8l::kSize> const& values) {
  if constexpr (I < Vec8l::kSize) {
    EXPECT_EQ(values[I], Get<I>(v));
    ExpectCompileTimeLanes<I + 1>(v, values);
  }
}

template <int N>
void TestCompileTimePartial(std::array<int64_t, Vec8l::kSize> const& values) {
  auto const v = Load<N, Vec8l>(values.data());
  for (int i = 0; i < Vec8l::kSize; ++i) {
    EXPECT_EQ(i < N ? values[i] : int64_t{0}, v[i]);
  }

  constexpr int64_t kSentinel = -911;
  std::array<int64_t, Vec8l::kSize + 2> stored{};
  stored.fill(kSentinel);
  Store<N>(stored.data() + 1, Load<Vec8l>(values.data()));
  EXPECT_EQ(kSentinel, stored.front());
  EXPECT_EQ(kSentinel, stored.back());
  for (int i = 0; i < Vec8l::kSize; ++i) {
    EXPECT_EQ(i < N ? values[i] : kSentinel, stored[i + 1]);
  }
}

template <size_t... Is>
void TestAllCompileTimePartials(
    std::array<int64_t, Vec8l::kSize> const& values,
    std::index_sequence<Is...>) {
  (TestCompileTimePartial<static_cast<int>(Is)>(values), ...);
}

template <int N = 1>
void ExpectPrefixAllTrue(Vec8l mask, std::array<bool, Vec8l::kSize> const& expected) {
  bool expectedAll = true;
  for (int i = 0; i < N; ++i) {
    expectedAll = expectedAll && expected[i];
  }
  EXPECT_EQ(expectedAll, AllTrue<N>(mask));
  if constexpr (N < Vec8l::kSize) {
    ExpectPrefixAllTrue<N + 1>(mask, expected);
  }
}

void ExpectEveryPrefixMaskBoundary() {
  std::array<int64_t, Vec8l::kSize> values{};
  std::array<bool, Vec8l::kSize> expected{};
  for (int lane = 0; lane < Vec8l::kSize; ++lane) {
    values.fill(-1);
    expected.fill(true);
    values[lane] = 0;
    expected[lane] = false;
    ExpectPrefixAllTrue(Load<Vec8l>(values.data()), expected);

    values.fill(0);
    expected.fill(false);
    values[lane] = -1;
    expected[lane] = true;
    ExpectPrefixAllTrue(Load<Vec8l>(values.data()), expected);
  }
}

template <int N = 2>
void ExpectReductions(Vec8l values) {
  int64_t expectedMin = values[0];
  int64_t expectedMax = values[0];
  int64_t expectedSum = values[0];
  for (int i = 1; i < N; ++i) {
    expectedMin = std::min(expectedMin, values[i]);
    expectedMax = std::max(expectedMax, values[i]);
    expectedSum += values[i];
  }
  EXPECT_EQ(expectedMin, HMin<N>(values));
  EXPECT_EQ(expectedMax, HMax<N>(values));
  EXPECT_EQ(expectedSum, HSum<N>(values));
  if constexpr (N < Vec8l::kSize) {
    ExpectReductions<N + 1>(values);
  }
}

template <int N = 1>
void ExpectTransposedIO(std::array<int64_t, 3 * Vec8l::kSize> const& tuples) {
  Vec8l x;
  Vec8l y;
  Vec8l z;
  LoadTransposed<N>(tuples.data(), x, y, z);
  for (int i = 0; i < Vec8l::kSize; ++i) {
    EXPECT_EQ(i < N ? tuples[3 * i] : int64_t{0}, x[i]);
    EXPECT_EQ(i < N ? tuples[3 * i + 1] : int64_t{0}, y[i]);
    EXPECT_EQ(i < N ? tuples[3 * i + 2] : int64_t{0}, z[i]);
  }

  constexpr int64_t kSentinel = -911;
  std::array<int64_t, 3 * Vec8l::kSize + 2> stored{};
  stored.fill(kSentinel);
  StoreTransposed<N>(stored.data() + 1, x, y, z);
  for (int i = 0; i < 3 * Vec8l::kSize; ++i) {
    EXPECT_EQ(i < 3 * N ? tuples[i] : kSentinel, stored[i + 1]);
  }
  EXPECT_EQ(kSentinel, stored.front());
  EXPECT_EQ(kSentinel, stored.back());
  if constexpr (N < Vec8l::kSize) {
    ExpectTransposedIO<N + 1>(tuples);
  }
}

} // namespace

TEST(Vec8l, TypePropertiesAvx512) {
  static_assert(Vec8l::kIsSupported);
  static_assert(Vec8l::kSize == 8);
  static_assert(Vec8l::size() == 8);
  static_assert(sizeof(Vec8l) == sizeof(int64_t) * Vec8l::kSize);
  static_assert(alignof(Vec8l) == alignof(typename Vec8l::NativeType));
  static_assert(std::is_same_v<Vec8l::Scalar, int64_t>);

#if MOCHI_USE_SIMD
  static_assert(Vec8l::kIsComposite == !MOCHI_ARCH_X64_AVX512);
  static_assert(!Vec8l::kIsEmulated);
#else
  static_assert(!Vec8l::kIsComposite);
  static_assert(Vec8l::kIsEmulated);
#endif
}

TEST(Vec8l, HalfConstructorAndLaneAccessAvx512) {
  std::array<int64_t, Vec8l::kSize> values{};
  for (int i = 0; i < Vec8l::kSize; ++i) {
    values[i] = i + 1;
  }
  auto const v = Load<Vec8l>(values.data());
  EXPECT_EQ(v, (Vec8l{GetHalf<0>(v), GetHalf<1>(v)}));
  EXPECT_EQ(values[0], Get0(v));
  ExpectCompileTimeLanes(v, values);
  for (int i = 0; i < Vec8l::kSize; ++i) {
    EXPECT_EQ(values[i], v[i]);
  }
}

TEST(Vec8l, PartialLoadStoreAvx512) {
  std::array<int64_t, Vec8l::kSize> values{};
  for (int i = 0; i < Vec8l::kSize; ++i) {
    values[i] = i + 1;
  }

  constexpr int64_t kSentinel = -911;
  for (int count = 0; count <= Vec8l::kSize; ++count) {
    auto const v = Load<Vec8l>(values.data(), count);
    for (int i = 0; i < Vec8l::kSize; ++i) {
      EXPECT_EQ(i < count ? values[i] : int64_t{0}, v[i]);
    }

    std::array<int64_t, Vec8l::kSize + 2> stored{};
    stored.fill(kSentinel);
    Store(stored.data() + 1, Load<Vec8l>(values.data()), count);
    EXPECT_EQ(kSentinel, stored.front());
    EXPECT_EQ(kSentinel, stored.back());
    for (int i = 0; i < Vec8l::kSize; ++i) {
      EXPECT_EQ(i < count ? values[i] : kSentinel, stored[i + 1]);
    }
  }

  TestAllCompileTimePartials(values, std::make_index_sequence<Vec8l::kSize + 1>{});
}

TEST(Vec8l, StoreSelectedAvx512) {
  std::array<int64_t, Vec8l::kSize> values{};
  for (int i = 0; i < Vec8l::kSize; ++i) {
    values[i] = i + 1;
  }
  auto const source = Load<Vec8l>(values.data());
  std::array<int64_t, Vec8l::kSize> conditions{};
  constexpr int64_t kSentinel = -911;
  std::array<int64_t, Vec8l::kSize + 2> destination{};

  for (uint32_t mask = 0; mask < (uint32_t{1} << Vec8l::kSize); ++mask) {
    destination.fill(kSentinel);
    for (int i = 0; i < Vec8l::kSize; ++i) {
      conditions[i] = (mask & (uint32_t{1} << i)) != 0 ? int64_t{-1} : int64_t{0};
    }

    int const stored =
        StoreSelected(destination.data() + 1, Load<Vec8l>(conditions.data()), source);
    EXPECT_EQ(std::popcount(mask), stored);
    int expectedIndex = 0;
    for (int i = 0; i < Vec8l::kSize; ++i) {
      if ((mask & (uint32_t{1} << i)) != 0) {
        EXPECT_EQ(values[i], destination[++expectedIndex]);
      }
    }
    EXPECT_EQ(kSentinel, destination.front());
    if constexpr (MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512) {
      for (int i = stored + 1; i < static_cast<int>(destination.size()); ++i) {
        EXPECT_EQ(kSentinel, destination[i]);
      }
    }
  }
}

TEST(Vec8l, ComparisonsMasksAndSelectAvx512) {
  std::array<int64_t, Vec8l::kSize> lhsValues{};
  std::array<int64_t, Vec8l::kSize> rhsValues{};
  for (int i = 0; i < Vec8l::kSize; ++i) {
    lhsValues[i] = 3 * i;
    rhsValues[i] = lhsValues[i] + (i % 3) - 1;
  }
  auto const lhs = Load<Vec8l>(lhsValues.data());
  auto const rhs = Load<Vec8l>(rhsValues.data());
  auto const less = lhs < rhs;
  auto const greater = lhs > rhs;
  auto const lessEqual = lhs <= rhs;
  auto const greaterEqual = lhs >= rhs;
  auto const equal = VEqual(lhs, rhs);
  auto const notEqual = VNotEqual(lhs, rhs);
  auto const selected = Select(less, lhs, rhs);
  for (int i = 0; i < Vec8l::kSize; ++i) {
    bool const expectedLess = lhsValues[i] < rhsValues[i];
    EXPECT_EQ(expectedLess ? int64_t{-1} : int64_t{0}, less[i]);
    EXPECT_EQ(lhsValues[i] > rhsValues[i] ? int64_t{-1} : int64_t{0}, greater[i]);
    EXPECT_EQ(lhsValues[i] <= rhsValues[i] ? int64_t{-1} : int64_t{0}, lessEqual[i]);
    EXPECT_EQ(lhsValues[i] >= rhsValues[i] ? int64_t{-1} : int64_t{0}, greaterEqual[i]);
    EXPECT_EQ(lhsValues[i] == rhsValues[i] ? int64_t{-1} : int64_t{0}, equal[i]);
    EXPECT_EQ(lhsValues[i] != rhsValues[i] ? int64_t{-1} : int64_t{0}, notEqual[i]);
    EXPECT_EQ(expectedLess ? lhsValues[i] : rhsValues[i], selected[i]);
  }
  EXPECT_TRUE(lhs == lhs);
  EXPECT_FALSE(lhs == rhs);
  EXPECT_FALSE(lhs != lhs);
  EXPECT_TRUE(lhs != rhs);

  std::array<int64_t, Vec8l::kSize> prefixMaskValues{};
  std::array<bool, Vec8l::kSize> expectedPrefix{};
  prefixMaskValues.fill(-1);
  expectedPrefix.fill(true);
  prefixMaskValues[4] = 0;
  expectedPrefix[4] = false;
  ExpectPrefixAllTrue(Load<Vec8l>(prefixMaskValues.data()), expectedPrefix);
  ExpectEveryPrefixMaskBoundary();
  EXPECT_TRUE(AllTrue(Vec8l{-1}));
}

TEST(Vec8l, BroadcastAvx512) {
  std::array<int64_t, Vec8l::kSize> values{};
  for (int i = 0; i < Vec8l::kSize; ++i) {
    values[i] = i + 1;
  }
  auto const source = Load<Vec8l>(values.data());
  auto const fromPointer = Broadcast<Vec8l>(values.data() + 2);
  auto const fromLowLane = Broadcast<3>(source);
  auto const fromHighLane = Broadcast<4>(source);
  auto const fromLastLane = Broadcast<7>(source);
  for (int i = 0; i < Vec8l::kSize; ++i) {
    EXPECT_EQ(values[2], fromPointer[i]);
    EXPECT_EQ(values[3], fromLowLane[i]);
    EXPECT_EQ(values[4], fromHighLane[i]);
    EXPECT_EQ(values[7], fromLastLane[i]);
  }
}

TEST(Vec8l, ReductionsAvx512) {
  std::array<int64_t, Vec8l::kSize> values{};
  for (int i = 0; i < Vec8l::kSize; ++i) {
    values[i] = i % 2 == 0 ? i + 1 : -i;
  }
  ExpectReductions(Load<Vec8l>(values.data()));
}

TEST(Vec8l, TransposedIOAvx512) {
  std::array<int64_t, 3 * Vec8l::kSize> tuples{};
  for (int i = 0; i < static_cast<int>(tuples.size()); ++i) {
    tuples[i] = i + 1;
  }
  ExpectTransposedIO(tuples);
}

TEST(Vec8l, ShiftsAvx512) {
  auto const positive = Vec8l{1, 2, 3, 4, 5, 6, 7, 8};
  auto const shiftedLeft = positive << 3;
  auto const shiftedRight = ShiftRight<1>(Vec8l{-1, -3, -7, -15, 1, 2, 4, 8});
  std::array<int64_t, Vec8l::kSize> const expectedLeft{8, 16, 24, 32, 40, 48, 56, 64};
  std::array<int64_t, Vec8l::kSize> const expectedRight{-1, -2, -4, -8, 0, 1, 2, 4};
  for (int i = 0; i < Vec8l::kSize; ++i) {
    EXPECT_EQ(expectedLeft[i], shiftedLeft[i]);
    EXPECT_EQ(expectedRight[i], shiftedRight[i]);
  }
  EXPECT_EQ(positive, ShiftRight<0>(positive));

  constexpr int kHalfWidthShift = 8 * sizeof(Vec8l::Scalar) / 2;
  constexpr int64_t kHalfWidthScale = int64_t{1} << kHalfWidthShift;
  constexpr int64_t kInt64Min = std::numeric_limits<int64_t>::min();
  auto const signFill = ShiftRight<kHalfWidthShift>(Vec8l{-1, kInt64Min, -100, 100});
  EXPECT_EQ(-1, signFill[0]);
  EXPECT_EQ(kInt64Min / kHalfWidthScale, signFill[1]);
  EXPECT_EQ(-1, signFill[2]);
  EXPECT_EQ(0, signFill[3]);
}

#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512
TEST(Vec8l, ShuffleAvx512) {
  auto const a = Vec8l{1, 2, 3, 4, 5, 6, 7, 8};
  auto const b = Vec8l{9, 10, 11, 12, 13, 14, 15, 16};
  auto const oneSource = Vec8l::Shuffle<7, 0, 6, 1, 5, 2, 4, 3>(a);
  auto const twoSource = Vec8l::Shuffle<7, 0, 6, 1, 3, 2, 1, 0>(a, b);
  std::array<int64_t, Vec8l::kSize> const expectedOneSource{8, 1, 7, 2, 6, 3, 5, 4};
  std::array<int64_t, Vec8l::kSize> const expectedTwoSource{8, 1, 7, 2, 12, 11, 10, 9};
  for (int i = 0; i < Vec8l::kSize; ++i) {
    EXPECT_EQ(expectedOneSource[i], oneSource[i]);
    EXPECT_EQ(expectedTwoSource[i], twoSource[i]);
  }
}
#endif

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec8l, Add, +);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec8l, BitwiseAND, &);
MOCHI_SIMD_TEST_IS_VALID_LOGICAL_MASK(Vec8l);
MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec8l, Div, /);
MOCHI_SIMD_TEST_BINARY_FN_EXACT(Vec8l, Max, ([](auto a, auto b) { return std::max(a, b); }));
MOCHI_SIMD_TEST_BINARY_FN_EXACT(Vec8l, Min, ([](auto a, auto b) { return std::min(a, b); }));
MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec8l, Mul, *);
MOCHI_SIMD_TEST_UNARY_OP_EXACT(Vec8l, Neg, -);
MOCHI_SIMD_TEST_UNARY_BITWISE_OP(Vec8l, BitwiseNOT, ~);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec8l, BitwiseOR, |);
MOCHI_SIMD_TEST_UNARY_FN_EXACT(Vec8l, Sqr, ([](auto a) { return a * a; }));
MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec8l, Sub, -);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec8l, BitwiseXOR, ^);

TEST(Vec8l, ConstructorsBroadcastsSequenceAndZero) {
  Vec8l const scalar{3};
  Vec8l const values{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<int64_t, Vec8l::kSize> const expected{1, 2, 3, 4, 5, 6, 7, 8};
  auto const sequence = Sequence<Vec8l>();
  for (int i = 0; i < Vec8l::kSize; ++i) {
    EXPECT_EQ(3, scalar[i]);
    EXPECT_EQ(expected[i], values[i]);
    EXPECT_EQ(i, sequence[i]);
  }
  ExpectCompileTimeBroadcasts(values, expected);
  for (int lane = 0; lane < Vec8l::kSize; ++lane) {
    auto const broadcast = Broadcast(values, lane);
    for (int i = 0; i < Vec8l::kSize; ++i) {
      EXPECT_EQ(expected[lane], broadcast[i]);
    }
  }
  EXPECT_EQ(Vec8l{}, SimdZero<Vec8l>());
}

TEST(Vec8l, ExhaustiveMasks) {
  std::array<int64_t, Vec8l::kSize> lhsValues{};
  std::array<int64_t, Vec8l::kSize> rhsValues{};
  Vec8l const selectedTrue{17};
  Vec8l const selectedFalse{-19};
  for (uint32_t bits = 0; bits < (uint32_t{1} << Vec8l::kSize); ++bits) {
    for (int i = 0; i < Vec8l::kSize; ++i) {
      rhsValues[i] = (bits & (uint32_t{1} << i)) != 0 ? 1 : -1;
    }
    auto const mask = Load<Vec8l>(lhsValues.data()) < Load<Vec8l>(rhsValues.data());
    EXPECT_EQ(bits == ((uint32_t{1} << Vec8l::kSize) - 1), AllTrue(mask));
    auto const selected = Select(mask, selectedTrue, selectedFalse);
    for (int i = 0; i < Vec8l::kSize; ++i) {
      bool const expected = (bits & (uint32_t{1} << i)) != 0;
      EXPECT_EQ(expected ? int64_t{-1} : int64_t{0}, mask[i]);
      EXPECT_EQ(expected ? int64_t{17} : int64_t{-19}, selected[i]);
    }
  }
}

TEST(Vec8l, SimdMaskAndIsTrue) {
  auto const mask = SimdMask<Vec8l>(true, false, true, false, false, true, false, true);
  std::array<bool, Vec8l::kSize> const expected = {
      true, false, true, false, false, true, false, true};
  ExpectIsTrueLanes(mask, expected);
}

TEST(Vec8l, EveryShiftCount) {
  std::array<int64_t, Vec8l::kSize> const values = {
      -1,
      std::numeric_limits<int64_t>::min(),
      1,
      std::numeric_limits<int64_t>::max(),
      -3,
      7,
      -17,
      31};
  ExpectAllShifts(Load<Vec8l>(values.data()));
}

#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512
TEST(Vec8l, ShuffleAllSourcePositionsAvx512) {
  Vec8l const a{1, 2, 3, 4, 5, 6, 7, 8};
  Vec8l const b{9, 10, 11, 12, 13, 14, 15, 16};
  ExpectShuffleRotations(a, b);

  auto const optimized = Vec8l::Shuffle<0, 1, 2, 3, 0, 1, 2, 3>(a, b);
  std::array<int64_t, Vec8l::kSize> const expected{1, 2, 3, 4, 9, 10, 11, 12};
  for (int i = 0; i < Vec8l::kSize; ++i) {
    EXPECT_EQ(expected[i], optimized[i]);
  }
}
#endif
