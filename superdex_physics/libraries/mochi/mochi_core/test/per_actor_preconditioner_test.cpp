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

#include <mochi_core/linear_algebra/utils/matrix_conversions.h>
#include <mochi_core/solvers/per_actor_preconditioner.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

using namespace mochi;

namespace {

thread_local int gPhysicalWorkerId = -1;

struct ConcurrentCall {
  int workerId;
  int numWorkers;
  int rowBegin;
  int rowEnd;
  int physicalWorkerId;

  friend bool operator==(ConcurrentCall const&, ConcurrentCall const&) = default;
};

class RecordingActorPreconditioner final : public ActorPreconditioner<real> {
 public:
  RecordingActorPreconditioner(
      ActorPreconditionerParallelMode mode,
      int rowBlockSize,
      real scale,
      int numTeamBarriers = 0)
      : RecordingActorPreconditioner(mode, rowBlockSize, scale, DefaultCost(mode)) {
    _cost.numTeamBarriers = numTeamBarriers;
  }

  RecordingActorPreconditioner(
      ActorPreconditionerParallelMode mode,
      int rowBlockSize,
      real scale,
      ActorPreconditionerCost const& cost)
      : _parallelism{mode, rowBlockSize}, _cost(cost), _scale(scale) {}

  [[nodiscard]] ActorPreconditionerParallelism GetConcurrentSolveRequirements() const override {
    return _parallelism;
  }

  ActorPreconditionerCost GetConcurrentSolveCost() const override {
    return _cost;
  }

  void Solve(ColumnVectorView<real const> x, ColumnVectorView<real> Px) const override {
    {
      std::lock_guard lock(_callsMutex);
      _solvePhysicalWorkerIds.push_back(gPhysicalWorkerId);
    }
    for (int row = 0; row < x.Rows(); ++row) {
      Px(row, 0) = _scale * x(row, 0);
    }
    if (auto* signal = _solveCompletionSignal.exchange(nullptr); signal != nullptr) {
      signal->set_value();
    }
  }

  void ConcurrentSolve(
      ColumnVectorView<real const> x,
      ColumnVectorView<real> Px,
      ParallelWorkerInfo const& data) const override {
    {
      std::lock_guard lock(_callsMutex);
      _calls.push_back({data.workerId, data.numWorkers, data.rBegin, data.rEnd, gPhysicalWorkerId});
    }
    for (int wait = 0; wait < _cost.numTeamBarriers; ++wait) {
      data.BarrierWait();
    }
    for (int row = data.rBegin; row < data.rEnd; ++row) {
      Px(row, 0) = _scale * x(row, 0);
    }
  }

  void Update(ActorPseudoMatrix<real> const&) override {}

  PreconditionerType GetType() const override {
    return PreconditionerType::Jacobi;
  }

  DynamicArray<ConcurrentCall> Calls() const {
    std::lock_guard lock(_callsMutex);
    return _calls;
  }

  DynamicArray<int> SolvePhysicalWorkerIds() const {
    std::lock_guard lock(_callsMutex);
    return _solvePhysicalWorkerIds;
  }

  void SignalAfterNextSolve(std::promise<void>& signal) const {
    std::promise<void>* expected = nullptr;
    [[maybe_unused]] bool const installed =
        _solveCompletionSignal.compare_exchange_strong(expected, &signal);
    MOCHI_ASSERT(installed, "A solve completion signal is already armed.");
  }

 private:
  static ActorPreconditionerCost DefaultCost(ActorPreconditionerParallelMode mode) {
    if (mode == ActorPreconditionerParallelMode::SingleWorker) {
      return {.fixedCost = 100000.0, .parallelCost = 0.0, .maxUsefulWorkers = 1};
    }
    if (mode == ActorPreconditionerParallelMode::IndependentRows) {
      return {.fixedCost = 0.0, .parallelCost = 100000.0, .maxUsefulWorkers = 64};
    }
    return {.fixedCost = 25000.0, .parallelCost = 100000.0, .maxUsefulWorkers = 64};
  }

  ActorPreconditionerParallelism _parallelism;
  ActorPreconditionerCost _cost;
  real _scale;
  mutable std::atomic<std::promise<void>*> _solveCompletionSignal = nullptr;
  mutable std::mutex _callsMutex;
  mutable DynamicArray<ConcurrentCall> _calls;
  mutable DynamicArray<int> _solvePhysicalWorkerIds;
};

} // namespace

template <typename Fn>
static void RunWithExactWorkers(int numWorkers, Fn const& fn) {
  DynamicArray<std::thread> threads;
  threads.reserve(numWorkers);
  for (int workerId = 0; workerId < numWorkers; ++workerId) {
    threads.emplace_back([&fn, workerId] {
      gPhysicalWorkerId = workerId;
      fn(workerId);
      gPhysicalWorkerId = -1;
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

static void RunConcurrentSolve(
    PerActorPrec<real> const& prec,
    ColumnVector<real> const& x,
    ColumnVector<real>& Px,
    DynamicArray<int> const& workerRowRanges,
    int repetitions = 1) {
  int const numWorkers = static_cast<int>(workerRowRanges.size()) - 1;
  ParallelBarrier barrier(numWorkers);
  RunWithExactWorkers(numWorkers, [&](int workerId) {
    auto workerBarrier = barrier;
    if (workerId == 0) {
      prec.PrepareConcurrentSolve(MakeConstSpan(workerRowRanges));
    }
    workerBarrier.Wait();
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      prec.ConcurrentSolve(
          x,
          Px,
          {workerId,
           numWorkers,
           workerRowRanges[workerId],
           workerRowRanges[workerId + 1],
           workerBarrier});
    }
  });
}

static std::future<void> StartConcurrentSolveWorker(
    PerActorPrec<real> const& prec,
    ColumnVector<real> const& x,
    ColumnVector<real>& Px,
    DynamicArray<int> const& workerRowRanges,
    ParallelBarrier const& barrier,
    int workerId) {
  auto workerBarrier = barrier;
  return std::async(std::launch::async, [&, workerBarrier, workerId] {
    gPhysicalWorkerId = workerId;
    prec.ConcurrentSolve(
        x,
        Px,
        {workerId,
         static_cast<int>(workerRowRanges.size()) - 1,
         workerRowRanges[workerId],
         workerRowRanges[workerId + 1],
         workerBarrier});
    gPhysicalWorkerId = -1;
  });
}

static DynamicArray<ConcurrentCall> SortedCalls(RecordingActorPreconditioner const& prec) {
  auto calls = prec.Calls();
  std::sort(calls.begin(), calls.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.rowBegin < rhs.rowBegin;
  });
  return calls;
}

static DynamicArray<int> SortedUniquePhysicalWorkerIds(DynamicArray<ConcurrentCall> const& calls) {
  DynamicArray<int> result;
  result.reserve(calls.size());
  for (auto const& call : calls) {
    result.push_back(call.physicalWorkerId);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

static void ExpectCost(
    ActorPreconditionerCost const& actual,
    ActorPreconditionerCost const& expected) {
  EXPECT_DOUBLE_EQ(expected.fixedCost, actual.fixedCost);
  EXPECT_DOUBLE_EQ(expected.parallelCost, actual.parallelCost);
  EXPECT_EQ(expected.maxUsefulWorkers, actual.maxUsefulWorkers);
  EXPECT_EQ(expected.numTeamBarriers, actual.numTeamBarriers);
}

TEST(PerActorPreconditioner, ReuseMatVecRangesCoverRowsWithoutFinalSynchronization) {
  ActorPreconditionerCost const suffixCost{
      .fixedCost = 0.0, .parallelCost = 1000.0, .maxUsefulWorkers = 1};
  RecordingActorPreconditioner actor(ActorPreconditionerParallelMode::IndependentRows, 2, 2_r);
  RecordingActorPreconditioner suffix(
      ActorPreconditionerParallelMode::IndependentRows, 2, 2_r, suffixCost);
  PerActorPrec<real> prec({{0, 12, actor}, {12, 2, suffix}});
  ColumnVector<real> x(14), Px(14);
  x.SetRandom(11);
  Px.SetZero();
  DynamicArray<int> const workerRowRanges{0, 0, 4, 8, 14};
  prec.PrepareConcurrentSolve(MakeConstSpan(workerRowRanges));
  ParallelBarrier barrier(4);

  // An owner can finish before an empty worker starts because every write is local.
  StartConcurrentSolveWorker(prec, x, Px, workerRowRanges, barrier, 1).get();
  auto worker0 = StartConcurrentSolveWorker(prec, x, Px, workerRowRanges, barrier, 0);
  auto worker2 = StartConcurrentSolveWorker(prec, x, Px, workerRowRanges, barrier, 2);
  auto worker3 = StartConcurrentSolveWorker(prec, x, Px, workerRowRanges, barrier, 3);
  worker0.get();
  worker2.get();
  worker3.get();

  auto const calls = SortedCalls(actor);
  EXPECT_EQ(
      (DynamicArray<ConcurrentCall>{{1, 4, 0, 4, 1}, {2, 4, 4, 8, 2}, {3, 4, 8, 12, 3}}), calls);
  EXPECT_EQ((DynamicArray<ConcurrentCall>{{3, 4, 0, 2, 3}}), SortedCalls(suffix));
  ColumnVector<real> const expected = 2_r * x;
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
  EXPECT_TRUE(actor.SolvePhysicalWorkerIds().empty());
  EXPECT_TRUE(suffix.SolvePhysicalWorkerIds().empty());
}

TEST(PerActorPreconditioner, ReuseMatVecRangesWinsCostTie) {
  // Four times the two-worker barrier cost makes a 1:3 ownership split tie Broad's balanced
  // 2:2 split plus its final barrier.
  ActorPreconditionerCost const cost{
      .fixedCost = 0.0, .parallelCost = 15280.0, .maxUsefulWorkers = 2};
  RecordingActorPreconditioner actor(
      ActorPreconditionerParallelMode::IndependentRows, 1, 2_r, cost);
  PerActorPrec<real> prec({{0, 4, actor}});
  ColumnVector<real> x(4), Px(4);
  x.SetRandom(25);
  Px.SetZero();

  RunConcurrentSolve(prec, x, Px, {0, 1, 4});

  EXPECT_EQ((DynamicArray<ConcurrentCall>{{0, 2, 0, 1, 0}, {1, 2, 1, 4, 1}}), SortedCalls(actor));
  ColumnVector<real> const expected = 2_r * x;
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, BroadWinsWhenReuseMatVecRangesIsSlower) {
  // Just above the tie cost, Broad's balanced 2:2 split plus its final barrier beats the 1:3
  // ownership split.
  ActorPreconditionerCost const cost{
      .fixedCost = 0.0, .parallelCost = 16000.0, .maxUsefulWorkers = 2};
  RecordingActorPreconditioner actor(
      ActorPreconditionerParallelMode::IndependentRows, 1, 2_r, cost);
  PerActorPrec<real> prec({{0, 4, actor}});
  ColumnVector<real> x(4), Px(4);
  x.SetRandom(27);
  Px.SetZero();

  RunConcurrentSolve(prec, x, Px, {0, 1, 4});

  EXPECT_EQ((DynamicArray<ConcurrentCall>{{0, 2, 0, 2, 0}, {1, 2, 2, 4, 1}}), SortedCalls(actor));
  ColumnVector<real> const expected = 2_r * x;
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, IndependentRowsHonorsMaxUsefulWorkers) {
  ActorPreconditionerCost const cost{
      .fixedCost = 0.0, .parallelCost = 100000.0, .maxUsefulWorkers = 2};
  RecordingActorPreconditioner actor(
      ActorPreconditionerParallelMode::IndependentRows, 2, 2_r, cost);
  PerActorPrec<real> prec({ActorPreconditionerEntry<real>{0, 8, actor}});
  ColumnVector<real> x(8), Px(8);
  x.SetRandom(23);
  Px.SetZero();

  RunConcurrentSolve(prec, x, Px, {0, 2, 4, 6, 8});

  EXPECT_EQ((DynamicArray<ConcurrentCall>{{1, 4, 0, 4, 1}, {2, 4, 4, 8, 2}}), SortedCalls(actor));
  ColumnVector<real> const expected = 2_r * x;
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, IndependentRowsAvoidsAlreadyLoadedWorker) {
  ActorPreconditionerCost const singleCost{
      .fixedCost = 100.0, .parallelCost = 0.0, .maxUsefulWorkers = 1};
  ActorPreconditionerCost const independentCost{
      .fixedCost = 0.0, .parallelCost = 50.0, .maxUsefulWorkers = 2};
  RecordingActorPreconditioner single(
      ActorPreconditionerParallelMode::SingleWorker, 0, 2_r, singleCost);
  RecordingActorPreconditioner independent(
      ActorPreconditionerParallelMode::IndependentRows, 1, 3_r, independentCost);
  PerActorPrec<real> prec({{0, 2, single}, {2, 2, independent}});
  ColumnVector<real> x(4), Px(4), expected(4);
  x.SetRandom(12);
  Px.SetZero();
  expected.TopRows(2) = 2_r * x.TopRows(2);
  expected.BottomRows(2) = 3_r * x.BottomRows(2);

  RunConcurrentSolve(prec, x, Px, {0, 2, 4});

  EXPECT_EQ((DynamicArray<int>{0}), single.SolvePhysicalWorkerIds());
  EXPECT_EQ((DynamicArray<ConcurrentCall>{{1, 2, 0, 2, 1}}), independent.Calls());
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, IndependentRowsPlanningDoesNotScaleWithRowCount) {
  // A per-row or per-block planning loop would make this input impractical.
  int constexpr kNumRows = std::numeric_limits<int>::max() - 1;
  int constexpr kNumBlocks = kNumRows / 6;
  RecordingActorPreconditioner actor(ActorPreconditionerParallelMode::IndependentRows, 6, 1_r);
  PerActorPrec<real> prec({ActorPreconditionerEntry<real>{0, kNumRows, actor}});
  DynamicArray<int> const workerRowRanges{
      0, 6 * (kNumBlocks / 4), 6 * (kNumBlocks / 2), 6 * ((3 * kNumBlocks) / 4), kNumRows};

  prec.PrepareConcurrentSolve(MakeConstSpan(workerRowRanges));
}

TEST(PerActorPreconditioner, OneWorkerHandlesEveryExecutionMode) {
  RecordingActorPreconditioner independent(
      ActorPreconditionerParallelMode::IndependentRows, 2, 2_r);
  RecordingActorPreconditioner single(ActorPreconditionerParallelMode::SingleWorker, 1, 3_r);
  RecordingActorPreconditioner synchronized(
      ActorPreconditionerParallelMode::SynchronizedTeam, 2, 4_r, 2);
  PerActorPrec<real> prec({
      {0, 4, independent},
      {4, 2, single},
      {6, 4, synchronized},
  });
  ColumnVector<real> x(10), Px(10), expected(10);
  x.SetRandom(24);
  Px.SetZero();
  expected.TopRows(4) = 2_r * x.TopRows(4);
  expected.MiddleRows(4, 2) = 3_r * x.MiddleRows(4, 2);
  expected.BottomRows(4) = 4_r * x.BottomRows(4);

  RunConcurrentSolve(prec, x, Px, {0, 10});

  EXPECT_EQ(1, independent.Calls().size());
  EXPECT_EQ(1, single.SolvePhysicalWorkerIds().size());
  auto const synchronizedCalls = synchronized.Calls();
  ASSERT_EQ(1, synchronizedCalls.size());
  EXPECT_EQ(1, synchronizedCalls.front().numWorkers);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, SingleWorkerUsesLocalityToBreakLoadTies) {
  RecordingActorPreconditioner actor(ActorPreconditionerParallelMode::SingleWorker, 0, 3_r);
  PerActorPrec<real> prec({ActorPreconditionerEntry<real>{0, 10, actor}});
  ColumnVector<real> x(10), Px(10);
  x.SetRandom(21);
  Px.SetZero();

  RunConcurrentSolve(prec, x, Px, {0, 2, 8, 10});

  EXPECT_EQ((DynamicArray<int>{1}), actor.SolvePhysicalWorkerIds());
  ColumnVector<real> const expected = 3_r * x;
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, SingleWorkerPrefersMidpointOwnerForUnevenRanges) {
  ActorPreconditionerCost const cheapCost{
      .fixedCost = 1.0, .parallelCost = 0.0, .maxUsefulWorkers = 1};
  ActorPreconditionerCost const targetCost{
      .fixedCost = 100.0, .parallelCost = 0.0, .maxUsefulWorkers = 1};
  RecordingActorPreconditioner prefix(
      ActorPreconditionerParallelMode::SingleWorker, 0, 1_r, cheapCost);
  RecordingActorPreconditioner target(
      ActorPreconditionerParallelMode::SingleWorker, 0, 1_r, targetCost);
  RecordingActorPreconditioner suffix(
      ActorPreconditionerParallelMode::SingleWorker, 0, 1_r, cheapCost);
  PerActorPrec<real> prec({
      {0, 9, prefix},
      {9, 1, target},
      {10, 1, suffix},
  });
  ColumnVector<real> x(11), Px(11);
  x.SetRandom(22);
  Px.SetZero();

  RunConcurrentSolve(prec, x, Px, {0, 9, 11});

  EXPECT_EQ((DynamicArray<int>{1}), target.SolvePhysicalWorkerIds());
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, x, real{0}));
}

TEST(PerActorPreconditioner, NonlocalWritesCompleteBeforeWorkersReturn) {
  RecordingActorPreconditioner actor(ActorPreconditionerParallelMode::SingleWorker, 1, 2_r);
  PerActorPrec<real> prec({ActorPreconditionerEntry<real>{0, 4, actor}});
  ColumnVector<real> x(4), Px(4);
  x.SetRandom(26);
  Px.SetZero();
  ColumnVector<real> const expected = 2_r * x;
  DynamicArray<int> const workerRowRanges{0, 1, 4};
  prec.PrepareConcurrentSolve(MakeConstSpan(workerRowRanges));
  ParallelBarrier barrier(2);

  // Wait until worker 1 completes the nonlocal solve, then verify the wrapper still blocks it.
  std::promise<void> firstSolveCompletedPromise;
  auto firstSolveCompleted = firstSolveCompletedPromise.get_future();
  actor.SignalAfterNextSolve(firstSolveCompletedPromise);
  auto firstWorker = StartConcurrentSolveWorker(prec, x, Px, workerRowRanges, barrier, 1);
  firstSolveCompleted.wait();
  EXPECT_EQ(std::future_status::timeout, firstWorker.wait_for(std::chrono::seconds{0}));
  auto secondWorker = StartConcurrentSolveWorker(prec, x, Px, workerRowRanges, barrier, 0);
  firstWorker.get();
  secondWorker.get();

  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, SynchronizedTeamUsesDenseIdsAndReusableBarrier) {
  RecordingActorPreconditioner actor(
      ActorPreconditionerParallelMode::SynchronizedTeam,
      2,
      4_r,
      /*numTeamBarriers*/ 2);
  PerActorPrec<real> prec({ActorPreconditionerEntry<real>{0, 10, actor}});
  ColumnVector<real> x(10), Px(10);
  x.SetRandom(13);
  Px.SetZero();

  RunConcurrentSolve(prec, x, Px, {0, 5, 10}, /*repetitions*/ 8);

  auto const calls = actor.Calls();
  ASSERT_FALSE(calls.empty());
  int const teamSize = calls.front().numWorkers;
  ASSERT_EQ(2, teamSize);
  EXPECT_EQ(static_cast<size_t>(teamSize * 8), calls.size());
  DynamicArray<ConcurrentCall> laneCalls;
  for (int workerId = 0; workerId < teamSize; ++workerId) {
    auto const call = std::find_if(calls.begin(), calls.end(), [&](ConcurrentCall const& value) {
      return value.workerId == workerId;
    });
    ASSERT_NE(calls.end(), call);
    EXPECT_EQ(8, std::count(calls.begin(), calls.end(), *call));
    laneCalls.push_back(*call);
  }
  std::sort(laneCalls.begin(), laneCalls.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.rowBegin < rhs.rowBegin;
  });
  EXPECT_EQ(0, laneCalls.front().rowBegin);
  EXPECT_EQ(10, laneCalls.back().rowEnd);
  for (int i = 0; i < static_cast<int>(laneCalls.size()); ++i) {
    EXPECT_EQ(teamSize, laneCalls[i].numWorkers);
    EXPECT_EQ(0, laneCalls[i].rowBegin % 2);
    EXPECT_EQ(0, laneCalls[i].rowEnd % 2);
    EXPECT_LT(laneCalls[i].rowBegin, laneCalls[i].rowEnd);
    if (i > 0) {
      EXPECT_EQ(laneCalls[i - 1].rowEnd, laneCalls[i].rowBegin);
    }
  }
  ColumnVector<real> const expected = 4_r * x;
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, OverlappingSynchronizedTeamsCompleteWithoutDeadlock) {
  ActorPreconditionerCost const actor0Cost{
      .fixedCost = 40000.0, .parallelCost = 80000.0, .maxUsefulWorkers = 2, .numTeamBarriers = 2};
  ActorPreconditionerCost const actor1Cost{
      .fixedCost = 0.0, .parallelCost = 100000.0, .maxUsefulWorkers = 2, .numTeamBarriers = 2};
  ActorPreconditionerCost const actor2Cost{
      .fixedCost = 0.0, .parallelCost = 160000.0, .maxUsefulWorkers = 2, .numTeamBarriers = 2};
  RecordingActorPreconditioner actor0(
      ActorPreconditionerParallelMode::SynchronizedTeam, 2, 2_r, actor0Cost);
  RecordingActorPreconditioner actor1(
      ActorPreconditionerParallelMode::SynchronizedTeam, 2, 3_r, actor1Cost);
  RecordingActorPreconditioner actor2(
      ActorPreconditionerParallelMode::SynchronizedTeam, 2, 4_r, actor2Cost);
  PerActorPrec<real> prec({
      {0, 4, actor0},
      {4, 4, actor1},
      {8, 4, actor2},
  });
  ColumnVector<real> x(12), Px(12), expected(12);
  x.SetRandom(14);
  Px.SetZero();
  expected.TopRows(4) = 2_r * x.TopRows(4);
  expected.MiddleRows(4, 4) = 3_r * x.MiddleRows(4, 4);
  expected.BottomRows(4) = 4_r * x.BottomRows(4);

  RunConcurrentSolve(prec, x, Px, {0, 3, 6, 9, 12}, /*repetitions*/ 8);

  auto const calls0 = actor0.Calls();
  auto const calls1 = actor1.Calls();
  auto const calls2 = actor2.Calls();
  EXPECT_EQ(16, calls0.size());
  EXPECT_EQ(16, calls1.size());
  EXPECT_EQ(16, calls2.size());
  EXPECT_EQ((DynamicArray<int>{0, 1}), SortedUniquePhysicalWorkerIds(calls0));
  EXPECT_EQ((DynamicArray<int>{1, 2}), SortedUniquePhysicalWorkerIds(calls1));
  EXPECT_EQ((DynamicArray<int>{2, 3}), SortedUniquePhysicalWorkerIds(calls2));
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
}

TEST(PerActorPreconditioner, MixedModesMatchSerialSolve) {
  RecordingActorPreconditioner independent(
      ActorPreconditionerParallelMode::IndependentRows, 2, 2_r);
  RecordingActorPreconditioner single(ActorPreconditionerParallelMode::SingleWorker, 1, 3_r);
  RecordingActorPreconditioner synchronized(
      ActorPreconditionerParallelMode::SynchronizedTeam, 2, 4_r, 2);
  PerActorPrec<real> prec({
      {0, 4, independent},
      {4, 2, single},
      {6, 6, synchronized},
  });
  ColumnVector<real> x(12), parallelResult(12), serialResult(12);
  x.SetRandom(15);
  parallelResult.SetZero();
  serialResult.SetZero();

  prec.Solve(x, serialResult);
  RunConcurrentSolve(prec, x, parallelResult, {0, 3, 6, 9, 12});

  EXPECT_TRUE(mochi::test::NearEqualMatrices(parallelResult, serialResult, real{0}));
  EXPECT_EQ(2, single.SolvePhysicalWorkerIds().size());
  EXPECT_FALSE(independent.Calls().empty());
  EXPECT_FALSE(synchronized.Calls().empty());
}

TEST(PerActorPreconditioner, RepreparesForDifferentWorkerCount) {
  ActorPreconditionerCost const cost{
      .fixedCost = 0.0, .parallelCost = 100000.0, .maxUsefulWorkers = 4, .numTeamBarriers = 2};
  RecordingActorPreconditioner actor(
      ActorPreconditionerParallelMode::SynchronizedTeam, 1, 2_r, cost);
  PerActorPrec<real> prec({ActorPreconditionerEntry<real>{0, 12, actor}});
  ColumnVector<real> x(12), Px(12);
  x.SetRandom(16);
  ColumnVector<real> const expected = 2_r * x;

  Px.SetZero();
  RunConcurrentSolve(prec, x, Px, {0, 6, 12});
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));

  Px.SetZero();
  RunConcurrentSolve(prec, x, Px, {0, 0, 4, 8, 12});
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));

  auto const calls = actor.Calls();
  ASSERT_EQ(6, calls.size());
  EXPECT_EQ(2, std::count_if(calls.begin(), calls.end(), [](auto const& call) {
              return call.numWorkers == 2;
            }));
  EXPECT_EQ(4, std::count_if(calls.begin(), calls.end(), [](auto const& call) {
              return call.numWorkers == 4;
            }));
}

TEST(PerActorPreconditioner, EqualSynchronizedActorsUseSingleWorkerTeams) {
  ActorPreconditionerCost const cost{
      .fixedCost = 0.0, .parallelCost = 100000.0, .maxUsefulWorkers = 4};
  RecordingActorPreconditioner actor0(
      ActorPreconditionerParallelMode::SynchronizedTeam, 1, 1_r, cost);
  RecordingActorPreconditioner actor1(
      ActorPreconditionerParallelMode::SynchronizedTeam, 1, 1_r, cost);
  RecordingActorPreconditioner actor2(
      ActorPreconditionerParallelMode::SynchronizedTeam, 1, 1_r, cost);
  RecordingActorPreconditioner actor3(
      ActorPreconditionerParallelMode::SynchronizedTeam, 1, 1_r, cost);
  PerActorPrec<real> prec({
      {0, 4, actor0},
      {4, 4, actor1},
      {8, 4, actor2},
      {12, 4, actor3},
  });
  ColumnVector<real> x(16), Px(16);
  x.SetRandom(18);
  Px.SetZero();

  RunConcurrentSolve(prec, x, Px, {0, 4, 8, 12, 16});

  DynamicArray<int> physicalWorkers;
  for (auto const* actor : {&actor0, &actor1, &actor2, &actor3}) {
    auto const calls = actor->Calls();
    ASSERT_EQ(1, calls.size());
    EXPECT_EQ(0, calls.front().workerId);
    EXPECT_EQ(1, calls.front().numWorkers);
    physicalWorkers.push_back(calls.front().physicalWorkerId);
  }
  std::sort(physicalWorkers.begin(), physicalWorkers.end());
  EXPECT_EQ((DynamicArray<int>{0, 1, 2, 3}), physicalWorkers);
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, x, real{0}));
}

TEST(PerActorPreconditioner, IdleWorkersJoinRemainingSynchronizedActor) {
  ActorPreconditionerCost const cost{
      .fixedCost = 25000.0, .parallelCost = 100000.0, .maxUsefulWorkers = 4};
  RecordingActorPreconditioner actor0(
      ActorPreconditionerParallelMode::SynchronizedTeam, 1, 1_r, cost);
  RecordingActorPreconditioner actor1(
      ActorPreconditionerParallelMode::SynchronizedTeam, 1, 1_r, cost);
  RecordingActorPreconditioner actor2(
      ActorPreconditionerParallelMode::SynchronizedTeam, 1, 1_r, cost);
  RecordingActorPreconditioner actor3(
      ActorPreconditionerParallelMode::SynchronizedTeam, 1, 1_r, cost);
  RecordingActorPreconditioner actor4(
      ActorPreconditionerParallelMode::SynchronizedTeam, 1, 1_r, cost);
  PerActorPrec<real> prec({
      {0, 4, actor0},
      {4, 4, actor1},
      {8, 4, actor2},
      {12, 4, actor3},
      {16, 4, actor4},
  });
  ColumnVector<real> x(20), Px(20);
  x.SetRandom(19);
  Px.SetZero();

  RunConcurrentSolve(prec, x, Px, {0, 5, 10, 15, 20});

  for (auto const* actor : {&actor0, &actor1, &actor2, &actor3}) {
    auto const calls = actor->Calls();
    ASSERT_EQ(1, calls.size());
    EXPECT_EQ(1, calls.front().numWorkers);
  }
  auto const finalCalls = actor4.Calls();
  EXPECT_EQ(4, finalCalls.size());
  for (auto const& call : finalCalls) {
    EXPECT_EQ(4, call.numWorkers);
  }
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, x, real{0}));
}

TEST(PerActorPreconditioner, SynchronizedBarrierCostLimitsTeamWidth) {
  auto selectedWidth = [](int numTeamBarriers) {
    ActorPreconditionerCost const cost{
        .fixedCost = 0.0,
        .parallelCost = 100000.0,
        .maxUsefulWorkers = 4,
        .numTeamBarriers = numTeamBarriers};
    RecordingActorPreconditioner actor(
        ActorPreconditionerParallelMode::SynchronizedTeam, 1, 2_r, cost);
    PerActorPrec<real> prec({ActorPreconditionerEntry<real>{0, 4, actor}});
    ColumnVector<real> x(4), Px(4);
    x.SetRandom(27);
    Px.SetZero();

    RunConcurrentSolve(prec, x, Px, {0, 1, 2, 3, 4});

    ColumnVector<real> const expected = 2_r * x;
    EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
    return actor.Calls().size();
  };

  EXPECT_EQ(4, selectedWidth(0));
  EXPECT_EQ(1, selectedWidth(100));
}

// Returns the team width of a synchronized actor sharing 2 workers with a single-worker actor.
[[nodiscard]] static int SynchronizedWidthBesideShortActor(
    ActorPreconditionerCost const& synchronizedCost,
    double shortCost) {
  RecordingActorPreconditioner synchronizedActor(
      ActorPreconditionerParallelMode::SynchronizedTeam, 1, 2_r, synchronizedCost);
  RecordingActorPreconditioner shortActor(
      ActorPreconditionerParallelMode::SingleWorker,
      1,
      2_r,
      ActorPreconditionerCost{.fixedCost = shortCost, .maxUsefulWorkers = 1});
  PerActorPrec<real> prec({
      {0, 4, synchronizedActor},
      {4, 2, shortActor},
  });
  ColumnVector<real> x(6), Px(6);
  x.SetRandom(28);
  Px.SetZero();

  RunConcurrentSolve(prec, x, Px, {0, 3, 6});

  ColumnVector<real> const expected = 2_r * x;
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
  return isize(synchronizedActor.Calls());
}

TEST(PerActorPreconditioner, ShortActorLeavesSpareWorkerToSynchronizedActorWhenFaster) {
  ActorPreconditionerCost const synchronizedCost{
      .fixedCost = 100000.0, .parallelCost = 20000.0, .maxUsefulWorkers = 4};
  // The busiest worker's load is 112000 with a 2-wide team and 120000 on one worker.
  EXPECT_EQ(2, SynchronizedWidthBesideShortActor(synchronizedCost, 2000.0));
  // The short actor would run after the 2-wide team, so the busiest worker's load is 120000 either
  // way and the tie keeps one worker.
  EXPECT_EQ(1, SynchronizedWidthBesideShortActor(synchronizedCost, 10000.0));
}

TEST(PerActorPreconditioner, ShortActorsHoldingMoreThanHalfAWorkerKeepIt) {
  // At 40000, the short actor holds exactly half of each worker's 80000 share of the total cost. A
  // 2-wide team is predicted faster in both cases.
  ActorPreconditionerCost const synchronizedCost{.parallelCost = 120000.0, .maxUsefulWorkers = 4};
  EXPECT_EQ(2, SynchronizedWidthBesideShortActor(synchronizedCost, 40000.0));
  EXPECT_EQ(1, SynchronizedWidthBesideShortActor(synchronizedCost, 40001.0));
}

// Returns the calls of a 2-worker independent-row actor placed after a single-worker actor, on 2
// workers with the given matvec ranges.
[[nodiscard]] static DynamicArray<ConcurrentCall> WideCallsBesideShortActor(
    DynamicArray<int> const& workerRowRanges) {
  RecordingActorPreconditioner shortActor(
      ActorPreconditionerParallelMode::IndependentRows,
      1,
      2_r,
      ActorPreconditionerCost{.parallelCost = 10000.0, .maxUsefulWorkers = 1});
  RecordingActorPreconditioner wide(
      ActorPreconditionerParallelMode::IndependentRows,
      1,
      2_r,
      ActorPreconditionerCost{.parallelCost = 100000.0, .maxUsefulWorkers = 2});
  PerActorPrec<real> prec({{0, 2, shortActor}, {2, 4, wide}});
  ColumnVector<real> x(6), Px(6);
  x.SetRandom(30);
  Px.SetZero();

  RunConcurrentSolve(prec, x, Px, workerRowRanges);

  ColumnVector<real> const expected = 2_r * x;
  EXPECT_TRUE(mochi::test::NearEqualMatrices(Px, expected, real{0}));
  return SortedCalls(wide);
}

TEST(PerActorPreconditioner, SpareWorkerDoesNotOverrideReuseMatVecRanges) {
  // The 1:3 ownership split beats Broad without the spare worker, but not with it.
  EXPECT_EQ(
      (DynamicArray<ConcurrentCall>{{0, 2, 0, 1, 0}, {1, 2, 1, 4, 1}}),
      WideCallsBesideShortActor({0, 3, 6}));
}

TEST(PerActorPreconditioner, SpareWorkerAppliesWhenReuseMatVecRangesIsInvalidOrSlower) {
  // The short actor, limited to one worker, straddles both matvec ranges.
  EXPECT_EQ(2, isize(WideCallsBesideShortActor({0, 1, 6})));
  // One worker owns every row, so reusing the matvec ranges is slower than Broad.
  EXPECT_EQ(2, isize(WideCallsBesideShortActor({0, 0, 6})));
}

TEST(PerActorPreconditioner, BuiltInCostsAreValidAndRepresentativeModesMatchSerialExecution) {
  int constexpr kBlockSize = 3;
  int constexpr kActorSize = 2 * kBlockSize;

  auto denseActorMatrix = Matrix<real>::Zero(kActorSize, kActorSize);
  for (int i = 0; i < kBlockSize; ++i) {
    denseActorMatrix(i, i) = 4_r;
    denseActorMatrix(i + kBlockSize, i + kBlockSize) = 4_r;
    denseActorMatrix(i, i + kBlockSize) = -1_r;
    denseActorMatrix(i + kBlockSize, i) = -1_r;
  }
  auto actorMatrix = ToBlockSparseMatrix<kBlockSize>(denseActorMatrix, true);
  AnyMatrixView<real const> const actorMatrixView{AsConstView(actorMatrix)};

  ActorPseudoMatrix<real> const firstActorMatrix{0, actorMatrixView, {}};
  ActorPseudoMatrix<real> const secondActorMatrix{kActorSize, actorMatrixView, {}};
  ActorPseudoMatrix<real> const thirdActorMatrix{2 * kActorSize, actorMatrixView, {}};
  BlockJacobiActorPrec<real, kBlockSize> blockJacobiPrec{firstActorMatrix};
  SymInverseActorPrec<real> symInversePrec{firstActorMatrix};
  AMGActorPrec<real, kBlockSize> amgPrec{thirdActorMatrix};
  ILU0ActorPrec<real, kBlockSize> ilu0Prec{firstActorMatrix};
  ColoredSSORActorPrec<real, kBlockSize> coloredSSORPrec{secondActorMatrix};

  ASSERT_EQ(
      ActorPreconditionerParallelMode::SingleWorker,
      ilu0Prec.GetConcurrentSolveRequirements().mode);
  ASSERT_EQ(
      ActorPreconditionerParallelMode::SynchronizedTeam,
      coloredSSORPrec.GetConcurrentSolveRequirements().mode);
  ASSERT_EQ(kBlockSize, coloredSSORPrec.GetConcurrentSolveRequirements().rowBlockSize);
  ASSERT_EQ(
      ActorPreconditionerParallelMode::SynchronizedTeam,
      amgPrec.GetConcurrentSolveRequirements().mode);
  ASSERT_EQ(kBlockSize, amgPrec.GetConcurrentSolveRequirements().rowBlockSize);
  ExpectCost(blockJacobiPrec.GetConcurrentSolveCost(), {0.0, 36.0, 2, 0});
  ExpectCost(symInversePrec.GetConcurrentSolveCost(), {0.0, 66.0, 6, 0});
  ExpectCost(amgPrec.GetConcurrentSolveCost(), {14.4, 201.6, 2, 4});
  ExpectCost(ilu0Prec.GetConcurrentSolveCost(), {78.0, 0.0, 1, 0});
  ExpectCost(coloredSSORPrec.GetConcurrentSolveCost(), {84.0, 0.0, 1, 6});
  EXPECT_EQ(2, coloredSSORPrec.prec->NumColors());

  // ILU0 covers SingleWorker; colored SSOR and AMG cover SynchronizedTeam.
  PerActorPrec<real> prec({
      {0, kActorSize, ilu0Prec},
      {kActorSize, kActorSize, coloredSSORPrec},
      {2 * kActorSize, kActorSize, amgPrec},
  });

  ColumnVector<real> x(3 * kActorSize);
  ColumnVector<real> serialResult(3 * kActorSize);
  ColumnVector<real> parallelResult(3 * kActorSize);
  x.SetRandom(17);
  serialResult.SetZero();
  prec.Solve(x, serialResult);

  auto expectPlanMatchesSerial = [&](DynamicArray<int> const& workerRowRanges) {
    parallelResult.SetZero();
    RunConcurrentSolve(prec, x, parallelResult, workerRowRanges);
    EXPECT_TRUE(mochi::test::NearEqualMatrices(serialResult, parallelResult, 1e-5_r));
  };

  expectPlanMatchesSerial({0, 3, 6, 9, 12, 15, 18});
  expectPlanMatchesSerial({0, 6, 12, 18});
}
