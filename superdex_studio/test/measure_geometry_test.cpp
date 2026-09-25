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

// Covers the geometry the Measure tool is built on (rendering/measure_geometry.h): vertex welding,
// coplanar facet grouping, the ray cast, and the measurements derived from a selection. All
// fixtures are constructed here from first principles rather than from recorded output.

#include "rendering/measure_geometry.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/math_utils.h>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

using namespace superdex::studio;

using mochi::real;
using mochi::Real3;
using mochi::operator""_r; // real-valued literals (0_r, -1_r, ...)

namespace {

// A unit cube spanning [0, 1]^3, written the way a glb / obj reader emits it: every triangle
// carries its own three corners, so nothing is shared until welding merges the duplicates.
// 6 sides x 2 triangles x 3 corners = 36 vertices for 8 distinct corners.
std::vector<float> MakeDeindexedUnitCubePositions() {
  // The 8 corners, indexed as bit 0 = x, bit 1 = y, bit 2 = z.
  float const corner[8][3] = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
  // Two triangles per side, wound consistently outward. Sides in -x, +x, -y, +y, -z, +z order.
  int const side[6][4] = {
      {0, 4, 6, 2}, // -x
      {1, 3, 7, 5}, // +x
      {0, 1, 5, 4}, // -y
      {2, 6, 7, 3}, // +y
      {0, 2, 3, 1}, // -z
      {4, 5, 7, 6}, // +z
  };
  std::vector<float> positions;
  auto push = [&](int c) {
    positions.push_back(corner[c][0]);
    positions.push_back(corner[c][1]);
    positions.push_back(corner[c][2]);
  };
  for (auto const& quad : side) {
    push(quad[0]);
    push(quad[1]);
    push(quad[2]);
    push(quad[0]);
    push(quad[2]);
    push(quad[3]);
  }
  return positions;
}

// 0, 1, 2, ... one index per de-indexed corner.
std::vector<int> MakeSequentialIndices(std::size_t vertexCount) {
  std::vector<int> indices(vertexCount);
  for (std::size_t i = 0; i < vertexCount; ++i) {
    indices[i] = static_cast<int>(i);
  }
  return indices;
}

std::shared_ptr<MeasureMesh const> BuildFromPositions(
    std::vector<float> const& positions,
    std::vector<int> const& indices,
    real facetAngleToleranceDeg = 1) {
  std::vector<MeasureMeshSection> const sections{
      {mochi::MakeConstSpan(positions), mochi::MakeConstSpan(indices)}};
  MeasureMeshParams params;
  params.facetAngleToleranceDeg = facetAngleToleranceDeg;
  return BuildMeasureMesh(mochi::MakeConstSpan(sections), params);
}

std::shared_ptr<MeasureMesh const> BuildUnitCube(real facetAngleToleranceDeg = 1) {
  std::vector<float> const positions = MakeDeindexedUnitCubePositions();
  std::vector<int> const indices = MakeSequentialIndices(positions.size() / 3);
  return BuildFromPositions(positions, indices, facetAngleToleranceDeg);
}

// An open cylinder of `segments` side quads about the Z axis, de-indexed the same way.
std::vector<float> MakeCylinderSidePositions(int segments, real radius, real height) {
  std::vector<float> positions;
  auto push = [&](real x, real y, real z) {
    positions.push_back(static_cast<float>(x));
    positions.push_back(static_cast<float>(y));
    positions.push_back(static_cast<float>(z));
  };
  real const step = real(2) * std::numbers::pi_v<real> / static_cast<real>(segments);
  for (int i = 0; i < segments; ++i) {
    real const a0 = step * static_cast<real>(i);
    real const a1 = step * static_cast<real>(i + 1);
    real const x0 = radius * std::cos(a0);
    real const y0 = radius * std::sin(a0);
    real const x1 = radius * std::cos(a1);
    real const y1 = radius * std::sin(a1);
    push(x0, y0, 0);
    push(x1, y1, 0);
    push(x1, y1, height);
    push(x0, y0, 0);
    push(x1, y1, height);
    push(x0, y0, height);
  }
  return positions;
}

MeasureEntity PickAt(
    MeasureMesh const& mesh,
    Real3 const& origin,
    Real3 const& direction,
    MeasurePickMode pickMode) {
  std::optional<MeasureRayHit> const hit = RayCastMeasureMesh(mesh, origin, direction);
  EXPECT_TRUE(hit.has_value());
  return MakeMeasureEntity(
      mesh, *hit, pickMode, "target", MeasureTargetKind::Render, [](Real3 const& p) { return p; });
}

} // namespace

TEST(MeasureGeometry, WeldingMergesDeindexedCorners) {
  std::vector<float> const positions = MakeDeindexedUnitCubePositions();
  auto const mesh = BuildUnitCube();

  EXPECT_EQ(positions.size() / 3, 36u); // the fixture really is de-indexed
  EXPECT_EQ(mesh->vertices.size(), 8u);
  EXPECT_EQ(mesh->GetNumTriangles(), 12);
}

TEST(MeasureGeometry, CubeHasOneFacetPerSide) {
  auto const mesh = BuildUnitCube();

  EXPECT_EQ(mesh->GetNumFacets(), 6);
  for (int facet = 0; facet < mesh->GetNumFacets(); ++facet) {
    EXPECT_EQ(mesh->facets[static_cast<std::size_t>(facet)].size(), 2u); // two triangles per side
  }
}

TEST(MeasureGeometry, RayCastHitsNearestFacetAndSnapsToCorner) {
  auto const mesh = BuildUnitCube();

  // Straight down onto the +z side, nearest the (1, 1, 1) corner.
  std::optional<MeasureRayHit> const hit =
      RayCastMeasureMesh(*mesh, Real3(0.9_r, 0.9_r, 5_r), Real3(0_r, 0_r, -1_r));
  ASSERT_TRUE(hit.has_value());
  EXPECT_NEAR_TOL(hit->t, real(4), real(1e-4));
  EXPECT_NEAR_EQ(
      mesh->vertices[static_cast<std::size_t>(hit->nearestVertex)], Real3(1_r, 1_r, 1_r));
  // The facet it landed on is the +z side: both its triangles point along +z.
  for (int triangle : mesh->facets[static_cast<std::size_t>(hit->facet)]) {
    EXPECT_NEAR_TOL(
        mesh->triangleNormals[static_cast<std::size_t>(triangle)],
        Real3(0_r, 0_r, 1_r),
        real(1e-4));
  }
}

TEST(MeasureGeometry, RayCastMissesWhenAimedAway) {
  auto const mesh = BuildUnitCube();

  EXPECT_FALSE(
      RayCastMeasureMesh(*mesh, Real3(0.5_r, 0.5_r, 5_r), Real3(0_r, 0_r, 1_r)).has_value());
  EXPECT_FALSE(RayCastMeasureMesh(*mesh, Real3(5_r, 5_r, 5_r), Real3(0_r, 0_r, -1_r)).has_value());
}

TEST(MeasureGeometry, FacetPickReportsSideAreaAndExtents) {
  auto const mesh = BuildUnitCube();

  MeasureEntity const facet =
      PickAt(*mesh, Real3(0.5_r, 0.5_r, 5_r), Real3(0_r, 0_r, -1_r), MeasurePickMode::Facet);

  EXPECT_EQ(facet.kind, MeasureEntityKind::Facet);
  EXPECT_EQ(facet.points.size(), 4u); // the side's four corners, welded
  EXPECT_EQ(facet.triangles.size(), 6u);
  EXPECT_NEAR_TOL(facet.area, real(1), real(1e-4));
  EXPECT_NEAR_TOL(facet.centroid, Real3(0.5_r, 0.5_r, 1_r), real(1e-4));
  EXPECT_NEAR_TOL(facet.normal, Real3(0_r, 0_r, 1_r), real(1e-4));
  // A square outline: four edges of unit length.
  EXPECT_EQ(facet.boundaryEdges.size(), 8u);
}

TEST(MeasureGeometry, TriangleModeNarrowsToOneTriangle) {
  auto const mesh = BuildUnitCube();

  MeasureEntity const triangle =
      PickAt(*mesh, Real3(0.4_r, 0.4_r, 5_r), Real3(0_r, 0_r, -1_r), MeasurePickMode::Triangle);

  EXPECT_EQ(triangle.kind, MeasureEntityKind::Triangle);
  EXPECT_EQ(triangle.triangles.size(), 3u);
  EXPECT_EQ(triangle.points.size(), 3u);
  EXPECT_NEAR_TOL(triangle.area, real(0.5), real(1e-4));
}

TEST(MeasureGeometry, VertexPairReportsDeltaAndDistance) {
  auto const mesh = BuildUnitCube();

  // Two opposite corners of the cube: (0, 0, 1) from above, (1, 1, 0) from below.
  MeasureEntity const a =
      PickAt(*mesh, Real3(0.1_r, 0.1_r, 5_r), Real3(0_r, 0_r, -1_r), MeasurePickMode::Vertex);
  MeasureEntity const b =
      PickAt(*mesh, Real3(0.9_r, 0.9_r, -5_r), Real3(0_r, 0_r, 1_r), MeasurePickMode::Vertex);
  ASSERT_TRUE(mochi::NearEqual(a.points.front(), Real3(0_r, 0_r, 1_r)));
  ASSERT_TRUE(mochi::NearEqual(b.points.front(), Real3(1_r, 1_r, 0_r)));

  MeasureReportParams params;
  params.unit = MeasureUnit::Millimeters;
  MeasurementReport const report = ComputeMeasurements({a, b}, params);

  // 1 m along x, 1 m along y, -1 m along z => sqrt(3) m = 1732.051 mm.
  EXPECT_NE(report.text.find("1732.051 mm"), std::string::npos) << report.text;
  // One span plus three axis legs.
  EXPECT_EQ(report.annotations.size(), 4u);
  EXPECT_EQ(report.annotations.front().style, MeasureAnnotation::Style::Span);
}

TEST(MeasureGeometry, OppositeFacetsAreParallelOneUnitApart) {
  auto const mesh = BuildUnitCube();

  MeasureEntity const top =
      PickAt(*mesh, Real3(0.5_r, 0.5_r, 5_r), Real3(0_r, 0_r, -1_r), MeasurePickMode::Facet);
  MeasureEntity const bottom =
      PickAt(*mesh, Real3(0.5_r, 0.5_r, -5_r), Real3(0_r, 0_r, 1_r), MeasurePickMode::Facet);

  MeasureReportParams params;
  params.unit = MeasureUnit::Meters;
  MeasurementReport const report = ComputeMeasurements({top, bottom}, params);

  EXPECT_NE(report.text.find("Parallel"), std::string::npos) << report.text;
  EXPECT_NE(report.text.find("1.000000 m"), std::string::npos) << report.text;
}

TEST(MeasureGeometry, AdjacentFacetsMeetAtNinetyDegrees) {
  auto const mesh = BuildUnitCube();

  MeasureEntity const top =
      PickAt(*mesh, Real3(0.5_r, 0.5_r, 5_r), Real3(0_r, 0_r, -1_r), MeasurePickMode::Facet);
  MeasureEntity const side =
      PickAt(*mesh, Real3(5_r, 0.5_r, 0.5_r), Real3(-1_r, 0_r, 0_r), MeasurePickMode::Facet);

  MeasureReportParams params;
  params.unit = MeasureUnit::Meters;
  MeasurementReport const report = ComputeMeasurements({top, side}, params);

  EXPECT_NE(report.text.find("90.000 deg"), std::string::npos) << report.text;
  // Not parallel, so no plane gap is reported; the extremes come from the aggregate section.
  EXPECT_EQ(report.text.find("Plane gap"), std::string::npos) << report.text;
  EXPECT_NE(report.text.find("Max distance"), std::string::npos) << report.text;
}

TEST(MeasureGeometry, TightToleranceKeepsCylinderSidesSeparate) {
  int constexpr kSegments = 32;
  std::vector<float> const positions = MakeCylinderSidePositions(kSegments, real(0.5), real(2));
  std::vector<int> const indices = MakeSequentialIndices(positions.size() / 3);

  auto const tight = BuildFromPositions(positions, indices, /*facetAngleToleranceDeg=*/real(1));
  EXPECT_EQ(tight->GetNumFacets(), kSegments);

  // A tolerance wider than the 11.25 deg turn between neighbouring quads merges the whole wall.
  auto const loose = BuildFromPositions(positions, indices, /*facetAngleToleranceDeg=*/real(20));
  EXPECT_EQ(loose->GetNumFacets(), 1);
}

TEST(MeasureGeometry, UnitScalingIsConsistent) {
  EXPECT_NEAR_EQ(GetMeasureUnitScale(MeasureUnit::Meters), real(1));
  EXPECT_NEAR_EQ(GetMeasureUnitScale(MeasureUnit::Centimeters), real(100));
  EXPECT_NEAR_EQ(GetMeasureUnitScale(MeasureUnit::Millimeters), real(1000));
}

TEST(MeasureGeometry, EmptyInputYieldsEmptyMeshAndReport) {
  MeasureMeshParams const params;
  auto const mesh = BuildMeasureMesh(mochi::Span<MeasureMeshSection const>{}, params);
  ASSERT_NE(mesh, nullptr);
  EXPECT_TRUE(mesh->IsEmpty());
  EXPECT_FALSE(RayCastMeasureMesh(*mesh, Real3(0_r, 0_r, 1_r), Real3(0_r, 0_r, -1_r)).has_value());

  EXPECT_TRUE(ComputeMeasurements({}, MeasureReportParams{}).text.empty());
}
