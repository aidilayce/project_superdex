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

// The geometry and arithmetic behind the viewport Measure tool (@ref MeasureTool), split out from
// it so it can be tested on its own: this unit touches no Filament, ImGui, or file I/O.
//
//   1. @ref MeasureMesh / @ref RayCastMeasureMesh work in one object's LOCAL space, the space its
//      cached triangles are stored in. No transforms appear here.
//   2. @ref MeasureEntity / @ref ComputeMeasurements work in the MEASUREMENT space, editor/Mochi
//      space (Z-up, metres), which is what the tool reports in. Points arrive already transformed.
// The tool bridges the two by passing a local -> measurement function to @ref MakeMeasureEntity.

#include <mochi_core/geometry/bvh_tree.h>
#include <mochi_core/geometry/mesh_data.h>
#include <mochi_core/geometry/ray.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace superdex::studio {

//--------------------------------------------------------------------------------------------------
// OPTIONS
//--------------------------------------------------------------------------------------------------

// Unit lengths are reported in. All geometry is metres; this only scales the formatted output.
enum class MeasureUnit { Meters, Centimeters, Millimeters };

char const* GetMeasureUnitSuffix(MeasureUnit unit);
mochi::real GetMeasureUnitScale(MeasureUnit unit);

// What a click selects. A facet is the coplanar group of triangles around the one that was hit,
// which is what makes box dimensions and parallel-face distances usable on tessellated geometry;
// Triangle narrows the pick to the single triangle under the cursor.
enum class MeasurePickMode { Vertex, Triangle, Facet };

// Which representations of an object may be picked.
enum class MeasureTargetKind { Render, Collision };

char const* GetMeasureTargetKindLabel(MeasureTargetKind kind);

struct MeasureMeshParams {
  // Two edge-adjacent triangles join the same facet when the angle between their normals is at or
  // below this.
  mochi::real facetAngleToleranceDeg = 1;
  // Grid size vertices are quantized onto before being merged. Specifically for glb and obj
  // Only bitwise-equal (or grid-equal) positions merge, this is not a proximity weld.
  mochi::real weldToleranceMeters = mochi::real(1e-6);
};

//--------------------------------------------------------------------------------------------------
// MESH (LOCAL SPACE)
//--------------------------------------------------------------------------------------------------

// Cached analysis of one measurable object's triangle mesh, in the local space its
// mochi_renderer::SceneObject renders it in: welded vertices, per-triangle normals and areas,
// coplanar facet groups, and a BVH for the ray cast.
//
// Built once by @ref BuildMeasureMesh and then treated as immutable. Neither copyable nor movable:
// @ref bvh holds a pointer to @ref soup.
struct MeasureMesh {
  MeasureMesh() = default;
  MeasureMesh(MeasureMesh const&) = delete;
  MeasureMesh& operator=(MeasureMesh const&) = delete;
  MeasureMesh(MeasureMesh&&) = delete;
  MeasureMesh& operator=(MeasureMesh&&) = delete;

  // Welded vertex positions.
  std::vector<mochi::Real3> vertices;
  // Triangle corners as indices into @ref vertices, three per triangle.
  std::vector<int> triangles;
  // Unit normal and area per triangle. A degenerate (zero-area) triangle gets a zero normal, is
  // left out of every facet, and is never hit by a ray cast.
  std::vector<mochi::Real3> triangleNormals;
  std::vector<mochi::real> triangleAreas;
  // Facet id per triangle; -1 for a degenerate triangle.
  std::vector<int> triangleFacets;
  // Triangle ids per facet.
  std::vector<std::vector<int>> facets;
  // The same triangles as a soup, plus a BVH over it, for @ref RayCastMeasureMesh. `bvh` is null
  // when the mesh is empty.
  mochi::TriangleSoup soup;
  std::unique_ptr<mochi::AabbTree> bvh;

  bool IsEmpty() const {
    return triangles.empty();
  }
  int GetNumTriangles() const {
    return static_cast<int>(triangles.size() / 3);
  }
  int GetNumFacets() const {
    return static_cast<int>(facets.size());
  }
};

// One contiguous chunk of triangle geometry to measure: flat (x, y, z) positions and triangle
// indices local to them. Mirrors the two fields of mochi_renderer::MeshSection this unit needs, so
// it does not have to depend on the renderer.
struct MeasureMeshSection {
  mochi::Span<float const> positions;
  mochi::Span<int const> indices;
};

// Builds the cached analysis of @p sections (concatenated, each section's indices offset by the
// running vertex count) or of a triangle @p mesh. A non-triangle @ref mochi::MeshDataView, or input
// with no triangles, yields an empty @ref MeasureMesh rather than null.
std::shared_ptr<MeasureMesh const> BuildMeasureMesh(
    mochi::Span<MeasureMeshSection const> sections,
    MeasureMeshParams const& params);
std::shared_ptr<MeasureMesh const> BuildMeasureMesh(
    mochi::MeshDataView const& mesh,
    MeasureMeshParams const& params);

struct MeasureRayHit {
  int triangle = -1;
  // Facet the triangle belongs to. Always valid for a hit (degenerate triangles are not hit).
  int facet = -1;
  // Welded vertex index of the triangle corner with the largest barycentric weight, i.e. the corner
  // nearest the hit point.
  int nearestVertex = -1;
  // Distance along the ray in the units of the @p direction that was passed in.
  mochi::real t = 0;
  mochi::Real3 point = {};
};

// Casts a local-space ray against @p mesh, nearest hit wins. Triangles are two-sided: imported
// geometry has inconsistent winding, so a back face is a legitimate hit.
//
// @p direction need not be unit length. @ref MeasureRayHit::t comes back in its units, so a caller
// that transformed a world ray into several objects' local spaces can compare hits by transforming
// the hit points back rather than trusting t across objects of different scale.
std::optional<MeasureRayHit> RayCastMeasureMesh(
    MeasureMesh const& mesh,
    mochi::Real3 const& origin,
    mochi::Real3 const& direction);

//--------------------------------------------------------------------------------------------------
// ENTITIES (MEASUREMENT SPACE)
//--------------------------------------------------------------------------------------------------

enum class MeasureEntityKind { Vertex, Facet, Triangle };

// One picked vertex or face, resolved into measurement space at pick time.
//
// The geometry is baked rather than referenced (no SceneObject pointer, no mesh indices) so that a
// recorded measurement stays valid across a restage, which destroys and recreates every
// SceneObject. The cost is that a recorded highlight stays where the geometry was when it was
// picked, even if the scene is later re-posed.
struct MeasureEntity {
  MeasureEntityKind kind = MeasureEntityKind::Vertex;
  MeasureTargetKind targetKind = MeasureTargetKind::Render;
  // Name of the object the pick landed on, for the report and the selection list.
  std::string targetLabel;
  // A vertex pick holds its single point. A face pick holds the distinct corner positions of the
  // facet (or triangle).
  std::vector<mochi::Real3> points;
  // Face picks only: corner triples indexing @ref points.
  std::vector<int> triangles;
  // Face picks only: index pairs into @ref points for the edges on the facet's outline (an edge
  // used by exactly one of its triangles).
  std::vector<int> boundaryEdges;
  // Face picks only: area-weighted unit normal, area-weighted centroid, and total area [m^2].
  mochi::Real3 normal = {};
  mochi::Real3 centroid = {};
  mochi::real area = 0;

  [[nodiscard]] bool IsFace() const {
    return kind != MeasureEntityKind::Vertex;
  }
  // A vertex pick's point, or a face pick's centroid.
  [[nodiscard]] mochi::Real3 GetAnchor() const;
  // Short description for the selection list, e.g. "Facet (8 tris)".
  [[nodiscard]] std::string GetDescription() const;
};

// Maps a point from an object's local space into measurement space. The tool composes the object's
// world transform with its renderer -> editor space conversion.
using MeasureLocalToMeasureFn = std::function<mochi::Real3(mochi::Real3 const&)>;

// Resolves @p hit on @p mesh into a picked entity, transforming its geometry with @p
// localToMeasure.
// @p targetLabel and @p targetKind identify the object that was hit.
MeasureEntity MakeMeasureEntity(
    MeasureMesh const& mesh,
    MeasureRayHit const& hit,
    MeasurePickMode pickMode,
    std::string targetLabel,
    MeasureTargetKind targetKind,
    MeasureLocalToMeasureFn const& localToMeasure);

//--------------------------------------------------------------------------------------------------
// MEASUREMENTS
//--------------------------------------------------------------------------------------------------

// One line (and its label) the viewport draws for a measurement. Endpoints are in measurement
// space; the tool converts them back to render space to submit them.
struct MeasureAnnotation {
  enum class Style {
    // The measurement itself, e.g. vertex to vertex or plane to plane.
    Span,
    // One axis-aligned leg of a delta.
    Component,
  };
  mochi::Real3 from = {};
  mochi::Real3 to = {};
  std::string label;
  Style style = Style::Span;
  // Component legs only: 0, 1 or 2 for the X, Y or Z axis the leg runs along, so the viewport can
  // color it to match that axis on the transform gizmo. -1 for a span.
  int axis = -1;
};

struct MeasureReportParams {
  MeasureUnit unit = MeasureUnit::Millimeters;
  // Two faces count as parallel when the angle between their normals is within this of 0 or 180
  // degrees, which switches their pair report to a plane-to-plane perpendicular distance.
  mochi::real parallelAngleToleranceDeg = 1;
  // Selections with more points than this skip the O(n^2) min/max pairwise search
  int maxPairwisePoints = 4096;
};

struct MeasurementReport {
  // The panel's copyable readout.
  std::string text;
  // What the viewport should draw for it.
  std::vector<MeasureAnnotation> annotations;
};

// Computes everything reportable about @p entities: per-entity properties, the pair-specific
// measurement for exactly two, and the aggregate extents / extremes for any two or more. Returns an
// empty report for an empty selection.
MeasurementReport ComputeMeasurements(
    std::vector<MeasureEntity> const& entities,
    MeasureReportParams const& params);

} // namespace superdex::studio
