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

#include "assets/asset.h"

#include <mochi_physics/cpp_api/mochi_context.h>
#include <mochi_renderer/resource.h>
#include <mochi_renderer/scene.h>

#include <mochi_core/utils/span.h>

#include <math/vec3.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mochi_renderer {
class Mesh;
class WireframeMesh;
} // namespace mochi_renderer

namespace superdex::studio {

class AssetManager;

//--------------------------------------------------------------------------------------------------
// MOCHI MODEL ASSET
//--------------------------------------------------------------------------------------------------

class MochiModelAsset : public Asset {
 public:
  ~MochiModelAsset() override;
  std::unique_ptr<mochi_renderer::SceneObject> GetRenderModelInstance() const;
  mochi::ShapeHandle GetShape(
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& bakeTransform,
      mochi::Error& error);
  void ClearShapeCache();
  void UpdateRenderModel();
  mochi::ModelData const& GetModelData() const;
  mochi::ModelData& GetModelData();
  filament::math::float3 GetSurfaceColor() const;

  // Soft Dynamic Mesh Utils
  int GetSoftSurfaceVertexCount();
  // Surface node count of the shape's render surface (tetrahedral boundary or triangle surface),
  // regardless of element type. Used to stage an articulated skin (a triangle surface) through the
  // same deforming dynamic-mesh path as soft actors.
  int GetSurfaceVertexCount();
  bool CreateSoftDynamicMeshes(
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform,
      std::unique_ptr<mochi_renderer::Mesh>& outSolid,
      std::unique_ptr<mochi_renderer::WireframeMesh>& outWireframe);
  void UpdateSoftDynamicMeshes(
      mochi_renderer::Mesh* solid,
      mochi_renderer::WireframeMesh* wireframe,
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform);

  // Build a GPU-skinned wireframe of the shape's collision surface, for previewing an articulated
  // skin's deformation at editor time (and driving it at sim time) without per-vertex CPU updates.
  // The surface must carry skinning (Shape::GetSurfaceMeshData exposes it for skinned mesh shapes);
  // returns false otherwise. `boneCount` is the number of bones the caller will drive (articulation
  // link count); the renderable allocates at least enough bones to cover the referenced indices.
  // Deform it via WireframeMesh::SetBoneMatrices.
  bool CreateSkinnedCollisionWireframe(
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform,
      int boneCount,
      std::unique_ptr<mochi_renderer::WireframeMesh>& outWireframe);

  // Asset overrides
  char const* GetTypeLabel() const override;
  bool RendersThumbnail() const override;
  bool IsSavable() const override;
  bool Save() const override;
  bool ReloadFromDisk() override;
  void StageThumbnailScene(mochi_renderer::Scene& scene) override;
  void ShowAssetTileTooltipItems() const override;
  std::unique_ptr<AssetEditor> CreateEditor(SuperDexStudio* studio) override;

 private:
  friend class AssetManager;
  using Asset::Asset;
  static std::unique_ptr<MochiModelAsset> Create(
      std::string const& name,
      mochi::Path const& path,
      AssetManager* manager,
      mochi_renderer::ResourceManager& resourceManager);

  // Load the shape's render surface (native, unbaked): the tetrahedral boundary for a tet mesh, or
  // the triangle surface itself for a surface mesh. Vertex ordering matches the physics engine's
  // SurfaceNodePositions query, so the same buffer can be updated in place during simulation.
  mochi::MeshDataView GetNativeSurfaceMesh(mochi::Error& error);
  // Load the model's tetrahedral boundary surface (native, unbaked); empty view + one warning if
  // the model is not a tetrahedral mesh.
  mochi::MeshDataView GetNativeSoftSurface(mochi::Error& error);
  // Bake the native surface for `bakeScale` + `shapeTransform` into flat renderer-space buffers
  // (area-weighted normals). Works for both tetrahedral boundaries and triangle surfaces (soft
  // callers are already gated to tet meshes upstream). Returns false if the shape has no surface.
  bool BakeSoftSurface(
      mochi::Real3 const& bakeScale,
      mochi::TransformRT const& shapeTransform,
      std::vector<float>& positions,
      std::vector<float>& normals,
      std::vector<int>& indices);
  // Read the (surface-aligned) skinning from the native surface mesh into per-node bone indices and
  // weights (numNodes * weightsPerNode each, in the same node order as BakeSoftSurface). Returns
  // false if the shape carries no surface skinning. `outMaxBoneIndex` is the largest bone index
  // referenced (-1 if none).
  bool BakeSoftSurfaceSkinning(
      std::vector<int>& boneIndices,
      std::vector<float>& boneWeights,
      int& weightsPerNode,
      int& outMaxBoneIndex);

 private:
  std::unique_ptr<mochi_renderer::WireframeMesh> _renderModel;
  filament::math::float3 _color = {0.5f, 0.7f, 1.0f};
  filament::math::float3 _wireframeColor = {1.0f, 1.0f, 1.0f};
  mochi::ModelData _modelData;
  std::unordered_map<std::string, mochi::ShapeHandle> _shapeCache;
};

} // namespace superdex::studio
