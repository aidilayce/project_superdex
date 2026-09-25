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

static_assert(std::is_trivially_copyable_v<Vec16i>);

MOCHI_SIMD_TEST_SCALAR_CONVERSIONS(Vec16i);

namespace {

template <int I = 0>
void ExpectCompileTimeBroadcastsAndSets(
    Vec16i source,
    std::array<int, Vec16i::kSize> const& values) {
  if constexpr (I < Vec16i::kSize) {
    auto const broadcast = Broadcast<I>(source);
    auto const changed = Set<I>(source, -values[I]);
    for (int lane = 0; lane < Vec16i::kSize; ++lane) {
      EXPECT_EQ(values[I], broadcast[lane]);
      EXPECT_EQ(lane == I ? -values[I] : values[lane], changed[lane]);
    }
    ExpectCompileTimeBroadcastsAndSets<I + 1>(source, values);
  }
}

template <int I = 0>
void ExpectIsTrueLanes(Vec16i mask, std::array<bool, Vec16i::kSize> const& expected) {
  if constexpr (I < Vec16i::kSize) {
    EXPECT_EQ(expected[I], IsTrue<I>(mask) != 0);
    ExpectIsTrueLanes<I + 1>(mask, expected);
  }
}

template <int Shift = 0>
void ExpectAllShifts(Vec16i values) {
  auto const shiftedLeft = values << Shift;
  auto const shiftedRight = ShiftRight<Shift>(values);
  for (int i = 0; i < Vec16i::kSize; ++i) {
    auto const expectedLeft = std::bit_cast<int>(static_cast<uint32_t>(values[i]) << Shift);
    EXPECT_EQ(expectedLeft, shiftedLeft[i]);
    EXPECT_EQ(values[i] >> Shift, shiftedRight[i]);
  }
  if constexpr (Shift + 1 < 8 * sizeof(Vec16i::Scalar)) {
    ExpectAllShifts<Shift + 1>(values);
  }
}

template <int I = 0>
void ExpectCompileTimeLanes(Vec16i v, std::array<int, Vec16i::kSize> const& values) {
  if constexpr (I < Vec16i::kSize) {
    EXPECT_EQ(values[I], Get<I>(v));
    ExpectCompileTimeLanes<I + 1>(v, values);
  }
}

template <int N>
void TestCompileTimePartial(std::array<int, Vec16i::kSize> const& values) {
  auto const v = Load<N, Vec16i>(values.data());
  for (int i = 0; i < Vec16i::kSize; ++i) {
    EXPECT_EQ(i < N ? values[i] : 0, v[i]);
  }

  constexpr int kSentinel = -911;
  std::array<int, Vec16i::kSize + 2> stored{};
  stored.fill(kSentinel);
  Store<N>(stored.data() + 1, Load<Vec16i>(values.data()));
  EXPECT_EQ(kSentinel, stored.front());
  EXPECT_EQ(kSentinel, stored.back());
  for (int i = 0; i < Vec16i::kSize; ++i) {
    EXPECT_EQ(i < N ? values[i] : kSentinel, stored[i + 1]);
  }
}

template <size_t... Is>
void TestAllCompileTimePartials(
    std::array<int, Vec16i::kSize> const& values,
    std::index_sequence<Is...>) {
  (TestCompileTimePartial<static_cast<int>(Is)>(values), ...);
}

template <int N = 1>
void ExpectPrefixMaskReductions(Vec16i mask, std::array<bool, Vec16i::kSize> const& expected) {
  bool expectedAll = true;
  bool expectedAny = false;
  for (int i = 0; i < N; ++i) {
    expectedAll = expectedAll && expected[i];
    expectedAny = expectedAny || expected[i];
  }
  EXPECT_EQ(expectedAll, AllTrue<N>(mask));
  EXPECT_EQ(expectedAny, AnyTrue<N>(mask));
  if constexpr (N < Vec16i::kSize) {
    ExpectPrefixMaskReductions<N + 1>(mask, expected);
  }
}

void ExpectEveryPrefixMaskBoundary() {
  std::array<int, Vec16i::kSize> values{};
  std::array<bool, Vec16i::kSize> expected{};
  for (int lane = 0; lane < Vec16i::kSize; ++lane) {
    values.fill(-1);
    expected.fill(true);
    values[lane] = 0;
    expected[lane] = false;
    ExpectPrefixMaskReductions(Load<Vec16i>(values.data()), expected);

    values.fill(0);
    expected.fill(false);
    values[lane] = -1;
    expected[lane] = true;
    ExpectPrefixMaskReductions(Load<Vec16i>(values.data()), expected);
  }
}

template <int N = 2>
void ExpectReductions(Vec16i values) {
  int expectedMin = values[0];
  int expectedMax = values[0];
  int expectedSum = values[0];
  for (int i = 1; i < N; ++i) {
    expectedMin = std::min(expectedMin, values[i]);
    expectedMax = std::max(expectedMax, values[i]);
    expectedSum += values[i];
  }
  EXPECT_EQ(expectedMin, HMin<N>(values));
  EXPECT_EQ(expectedMax, HMax<N>(values));
  EXPECT_EQ(expectedSum, HSum<N>(values));
  if constexpr (N < Vec16i::kSize) {
    ExpectReductions<N + 1>(values);
  }
}

template <int N = 1>
void ExpectTransposedIO(std::array<int, 3 * Vec16i::kSize> const& tuples) {
  Vec16i x;
  Vec16i y;
  Vec16i z;
  LoadTransposed<N>(tuples.data(), x, y, z);
  for (int i = 0; i < Vec16i::kSize; ++i) {
    EXPECT_EQ(i < N ? tuples[3 * i] : 0, x[i]);
    EXPECT_EQ(i < N ? tuples[3 * i + 1] : 0, y[i]);
    EXPECT_EQ(i < N ? tuples[3 * i + 2] : 0, z[i]);
  }

  constexpr int kSentinel = -911;
  std::array<int, 3 * Vec16i::kSize + 2> stored{};
  stored.fill(kSentinel);
  StoreTransposed<N>(stored.data() + 1, x, y, z);
  for (int i = 0; i < 3 * Vec16i::kSize; ++i) {
    EXPECT_EQ(i < 3 * N ? tuples[i] : kSentinel, stored[i + 1]);
  }
  EXPECT_EQ(kSentinel, stored.front());
  EXPECT_EQ(kSentinel, stored.back());
  if constexpr (N < Vec16i::kSize) {
    ExpectTransposedIO<N + 1>(tuples);
  }
}

} // namespace

TEST(Vec16i, TypeProperties) {
  static_assert(Vec16i::kIsSupported);
  static_assert(Vec16i::kSize == 16);
  static_assert(Vec16i::size() == 16);
  static_assert(sizeof(Vec16i) == sizeof(int) * Vec16i::kSize);
  static_assert(alignof(Vec16i) == alignof(typename Vec16i::NativeType));
  static_assert(std::is_same_v<Vec16i::Scalar, int>);

#if MOCHI_USE_SIMD
  static_assert(Vec16i::kIsComposite == !MOCHI_ARCH_X64_AVX512);
  static_assert(!Vec16i::kIsEmulated);
#else
  static_assert(!Vec16i::kIsComposite);
  static_assert(Vec16i::kIsEmulated);
#endif
}

TEST(Vec16i, HalfConstructorAndLaneAccess) {
  std::array<int, Vec16i::kSize> values{};
  for (int i = 0; i < Vec16i::kSize; ++i) {
    values[i] = i + 1;
  }
  auto const v = Load<Vec16i>(values.data());
  EXPECT_EQ(v, (Vec16i{GetHalf<0>(v), GetHalf<1>(v)}));
  EXPECT_EQ(values[0], Get0(v));
  ExpectCompileTimeLanes(v, values);
  for (int i = 0; i < Vec16i::kSize; ++i) {
    EXPECT_EQ(values[i], v[i]);
  }
}

TEST(Vec16i, PartialLoadStore) {
  std::array<int, Vec16i::kSize> values{};
  for (int i = 0; i < Vec16i::kSize; ++i) {
    values[i] = i + 1;
  }

  constexpr int kSentinel = -911;
  for (int count = 0; count <= Vec16i::kSize; ++count) {
    auto const v = Load<Vec16i>(values.data(), count);
    for (int i = 0; i < Vec16i::kSize; ++i) {
      EXPECT_EQ(i < count ? values[i] : 0, v[i]);
    }

    std::array<int, Vec16i::kSize + 2> stored{};
    stored.fill(kSentinel);
    Store(stored.data() + 1, Load<Vec16i>(values.data()), count);
    EXPECT_EQ(kSentinel, stored.front());
    EXPECT_EQ(kSentinel, stored.back());
    for (int i = 0; i < Vec16i::kSize; ++i) {
      EXPECT_EQ(i < count ? values[i] : kSentinel, stored[i + 1]);
    }
  }

  TestAllCompileTimePartials(values, std::make_index_sequence<Vec16i::kSize + 1>{});
}

TEST(Vec16i, StoreSelectedExhaustive) {
  std::array<int, Vec16i::kSize> values{};
  for (int i = 0; i < Vec16i::kSize; ++i) {
    values[i] = i + 1;
  }
  auto const source = Load<Vec16i>(values.data());
  std::array<int, Vec16i::kSize> conditions{};
  constexpr int kSentinel = -911;
  std::array<int, Vec16i::kSize + 2> destination{};

  for (uint32_t mask = 0; mask < (uint32_t{1} << Vec16i::kSize); ++mask) {
    destination.fill(kSentinel);
    for (int i = 0; i < Vec16i::kSize; ++i) {
      conditions[i] = (mask & (uint32_t{1} << i)) != 0 ? -1 : 0;
    }

    int const stored =
        StoreSelected(destination.data() + 1, Load<Vec16i>(conditions.data()), source);
    EXPECT_EQ(std::popcount(mask), stored);
    int expectedIndex = 0;
    for (int i = 0; i < Vec16i::kSize; ++i) {
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

TEST(Vec16i, ComparisonsMasksAndSelect) {
  std::array<int, Vec16i::kSize> lhsValues{};
  std::array<int, Vec16i::kSize> rhsValues{};
  std::array<bool, Vec16i::kSize> expectedLess{};
  for (int i = 0; i < Vec16i::kSize; ++i) {
    lhsValues[i] = 3 * i;
    rhsValues[i] = lhsValues[i] + (i % 3) - 1;
  }
  auto const lhs = Load<Vec16i>(lhsValues.data());
  auto const rhs = Load<Vec16i>(rhsValues.data());
  auto const less = lhs < rhs;
  auto const greater = lhs > rhs;
  auto const lessEqual = lhs <= rhs;
  auto const greaterEqual = lhs >= rhs;
  auto const equal = VEqual(lhs, rhs);
  auto const notEqual = VNotEqual(lhs, rhs);
  auto const selected = Select(less, lhs, rhs);
  for (int i = 0; i < Vec16i::kSize; ++i) {
    expectedLess[i] = lhsValues[i] < rhsValues[i];
    EXPECT_EQ(expectedLess[i] ? -1 : 0, less[i]);
    EXPECT_EQ(lhsValues[i] > rhsValues[i] ? -1 : 0, greater[i]);
    EXPECT_EQ(lhsValues[i] <= rhsValues[i] ? -1 : 0, lessEqual[i]);
    EXPECT_EQ(lhsValues[i] >= rhsValues[i] ? -1 : 0, greaterEqual[i]);
    EXPECT_EQ(lhsValues[i] == rhsValues[i] ? -1 : 0, equal[i]);
    EXPECT_EQ(lhsValues[i] != rhsValues[i] ? -1 : 0, notEqual[i]);
    EXPECT_EQ(expectedLess[i] ? lhsValues[i] : rhsValues[i], selected[i]);
  }
  EXPECT_TRUE(lhs == lhs);
  EXPECT_FALSE(lhs == rhs);
  EXPECT_FALSE(lhs != lhs);
  EXPECT_TRUE(lhs != rhs);
  ExpectPrefixMaskReductions(less, expectedLess);
  ExpectEveryPrefixMaskBoundary();

  std::array<int, Vec16i::kSize> prefixMaskValues{};
  std::array<bool, Vec16i::kSize> expectedPrefix{};
  prefixMaskValues.fill(-1);
  expectedPrefix.fill(true);
  prefixMaskValues[8] = 0;
  expectedPrefix[8] = false;
  ExpectPrefixMaskReductions(Load<Vec16i>(prefixMaskValues.data()), expectedPrefix);
  EXPECT_TRUE(AllTrue(Vec16i{-1}));
  EXPECT_FALSE(AnyTrue(Vec16i{0}));
}

#if MOCHI_USE_SIMD && MOCHI_ARCH_X64_AVX512
TEST(Vec16i, SetInt64) {
  std::array<uint64_t, 8> const patterns = {
      0x0123456789abcdefULL,
      0xfedcba9876543210ULL,
      0x8000000000000000ULL,
      0x0000000080000000ULL,
      0x7fffffff00000000ULL,
      0x00000000ffffffffULL,
      0xaaaaaaaa55555555ULL,
      0x13579bdf2468ace0ULL};
  auto const values = Vec16i::SetInt64(
      std::bit_cast<int64_t>(patterns[0]),
      std::bit_cast<int64_t>(patterns[1]),
      std::bit_cast<int64_t>(patterns[2]),
      std::bit_cast<int64_t>(patterns[3]),
      std::bit_cast<int64_t>(patterns[4]),
      std::bit_cast<int64_t>(patterns[5]),
      std::bit_cast<int64_t>(patterns[6]),
      std::bit_cast<int64_t>(patterns[7]));
  for (int i = 0; i < Vec16i::kSize; ++i) {
    auto const word = static_cast<uint32_t>(patterns[i / 2] >> (32 * (i % 2)));
    EXPECT_EQ(std::bit_cast<int>(word), values[i]);
  }
}
#endif

TEST(Vec16i, SetAndBroadcast) {
  std::array<int, Vec16i::kSize> values{};
  for (int i = 0; i < Vec16i::kSize; ++i) {
    values[i] = i + 1;
  }
  auto const source = Load<Vec16i>(values.data());
  EXPECT_EQ(-1, Set<0>(source, -1)[0]);
  EXPECT_EQ(-16, Set<15>(source, -16)[15]);

  auto changed = source;
  for (int i = 0; i < Vec16i::kSize; ++i) {
    changed = Set(changed, i, -values[i]);
  }
  for (int i = 0; i < Vec16i::kSize; ++i) {
    EXPECT_EQ(-values[i], changed[i]);
  }

  auto const fromPointer = Broadcast<Vec16i>(values.data() + 5);
  auto const fromLowLane = Broadcast<7>(source);
  auto const fromHighLane = Broadcast<8>(source);
  auto const fromLastLane = Broadcast<15>(source);
  for (int i = 0; i < Vec16i::kSize; ++i) {
    EXPECT_EQ(values[5], fromPointer[i]);
    EXPECT_EQ(values[7], fromLowLane[i]);
    EXPECT_EQ(values[8], fromHighLane[i]);
    EXPECT_EQ(values[15], fromLastLane[i]);
  }
}

TEST(Vec16i, Reductions) {
  std::array<int, Vec16i::kSize> values{};
  for (int i = 0; i < Vec16i::kSize; ++i) {
    values[i] = i % 2 == 0 ? i + 1 : -i;
  }
  ExpectReductions(Load<Vec16i>(values.data()));
}

TEST(Vec16i, TransposedIO) {
  std::array<int, 3 * Vec16i::kSize> tuples{};
  for (int i = 0; i < static_cast<int>(tuples.size()); ++i) {
    tuples[i] = i + 1;
  }
  ExpectTransposedIO(tuples);
}

TEST(Vec16i, Shifts) {
  std::array<int, Vec16i::kSize> positive{};
  std::array<int, Vec16i::kSize> mixed{};
  for (int i = 0; i < Vec16i::kSize; ++i) {
    positive[i] = i + 1;
    mixed[i] = i % 4 == 0 ? -1 : i % 4 == 1 ? -3 : i % 4 == 2 ? 4 : 8;
  }
  auto const positiveVector = Load<Vec16i>(positive.data());
  auto const mixedVector = Load<Vec16i>(mixed.data());
  auto const shiftedLeft = positiveVector << 3;
  auto const shiftedRight = ShiftRight<1>(mixedVector);
  for (int i = 0; i < Vec16i::kSize; ++i) {
    EXPECT_EQ(positive[i] * 8, shiftedLeft[i]);
    EXPECT_EQ(i % 4 == 0 ? -1 : i % 4 == 1 ? -2 : i % 4 == 2 ? 2 : 4, shiftedRight[i]);
  }
  EXPECT_EQ(positiveVector, ShiftRight<0>(positiveVector));

  constexpr int kHalfWidthShift = 8 * sizeof(Vec16i::Scalar) / 2;
  constexpr int kHalfWidthScale = 1 << kHalfWidthShift;
  constexpr int kIntMin = std::numeric_limits<int>::min();
  auto const signFill = ShiftRight<kHalfWidthShift>(Vec16i{-1, kIntMin, -100, 100});
  EXPECT_EQ(-1, signFill[0]);
  EXPECT_EQ(kIntMin / kHalfWidthScale, signFill[1]);
  EXPECT_EQ(-1, signFill[2]);
  EXPECT_EQ(0, signFill[3]);
}

MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec16i, Add, +);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec16i, BitwiseAND, &);
MOCHI_SIMD_TEST_IS_VALID_LOGICAL_MASK(Vec16i);
MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec16i, Div, /);
MOCHI_SIMD_TEST_BINARY_FN_EXACT(Vec16i, Max, ([](auto a, auto b) { return std::max(a, b); }));
MOCHI_SIMD_TEST_BINARY_FN_EXACT(Vec16i, Min, ([](auto a, auto b) { return std::min(a, b); }));
MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec16i, Mul, *);
MOCHI_SIMD_TEST_UNARY_OP_EXACT(Vec16i, Neg, -);
MOCHI_SIMD_TEST_UNARY_BITWISE_OP(Vec16i, BitwiseNOT, ~);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec16i, BitwiseOR, |);
MOCHI_SIMD_TEST_UNARY_FN_EXACT(Vec16i, Sqr, ([](auto a) { return a * a; }));
MOCHI_SIMD_TEST_BINARY_OP_EXACT(Vec16i, Sub, -);
MOCHI_SIMD_TEST_BINARY_BITWISE_OP(Vec16i, BitwiseXOR, ^);

TEST(Vec16i, ConstructorsAndAllLaneOperations) {
  Vec16i const scalar{3};
  Vec16i const values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::array<int, Vec16i::kSize> expected{};
  for (int i = 0; i < Vec16i::kSize; ++i) {
    expected[i] = i + 1;
    EXPECT_EQ(3, scalar[i]);
  }
  EXPECT_EQ(values, Load<Vec16i>(expected.data()));
  ExpectCompileTimeBroadcastsAndSets(values, expected);
  for (int lane = 0; lane < Vec16i::kSize; ++lane) {
    auto const broadcast = Broadcast(values, lane);
    for (int i = 0; i < Vec16i::kSize; ++i) {
      EXPECT_EQ(expected[lane], broadcast[i]);
    }
  }
}

TEST(Vec16i, ExhaustiveMasks) {
  std::array<int, Vec16i::kSize> lhsValues{};
  std::array<int, Vec16i::kSize> rhsValues{};
  Vec16i const selectedTrue{17};
  Vec16i const selectedFalse{-19};
  for (uint32_t bits = 0; bits < (uint32_t{1} << Vec16i::kSize); ++bits) {
    for (int i = 0; i < Vec16i::kSize; ++i) {
      rhsValues[i] = (bits & (uint32_t{1} << i)) != 0 ? 1 : -1;
    }
    auto const mask = Load<Vec16i>(lhsValues.data()) < Load<Vec16i>(rhsValues.data());
    EXPECT_EQ(bits == ((uint32_t{1} << Vec16i::kSize) - 1), AllTrue(mask));
    EXPECT_EQ(bits != 0, AnyTrue(mask));
    auto const selected = Select(mask, selectedTrue, selectedFalse);
    for (int i = 0; i < Vec16i::kSize; ++i) {
      bool const expected = (bits & (uint32_t{1} << i)) != 0;
      EXPECT_EQ(expected ? -1 : 0, mask[i]);
      EXPECT_EQ(expected ? 17 : -19, selected[i]);
    }
  }
}

TEST(Vec16i, SequenceMaskIsTrueAndZero) {
  auto const sequence = Sequence<Vec16i>();
  auto const mask = SimdMask<Vec16i>(
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true);
  std::array<bool, Vec16i::kSize> const expected = {
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true};
  ExpectIsTrueLanes(mask, expected);
  for (int i = 0; i < Vec16i::kSize; ++i) {
    EXPECT_EQ(i, sequence[i]);
  }
  EXPECT_EQ(Vec16i{}, SimdZero<Vec16i>());
}

TEST(Vec16i, EveryShiftCount) {
  std::array<int, Vec16i::kSize> values{};
  for (int i = 0; i < Vec16i::kSize; ++i) {
    values[i] = i % 4 == 0 ? -1
        : i % 4 == 1       ? std::numeric_limits<int>::min()
        : i % 4 == 2       ? 1
                           : std::numeric_limits<int>::max();
  }
  ExpectAllShifts(Load<Vec16i>(values.data()));
}
