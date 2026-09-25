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

#include "mochi_physics_test_fixture.h"

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_physics/src/mochi_articulated_body.h>
#include <mochi_physics/src/mochi_common_components.h>
#include <mochi_physics/src/mochi_context.h>
#include <mochi_physics/src/mochi_deformable.h>
#include <mochi_physics/src/mochi_rigid.h>
#include <mochi_physics/src/mochi_shell.h>
#include <mochi_physics/src/mochi_soft.h>
#include <mochi_physics/src/mochi_step.h>

#include <gtest/gtest.h>

using namespace mochi;

namespace {
void InitializeTwoNodeContactSkinningJacobian(
    real weight0,
    real weight1,
    CContactSkinningData& outSkinning) {
  outSkinning.jacobian = SparseMatrix<Real3>(
      /*nCol=*/6,
      DynamicArray<int>{0, 6},
      DynamicArray<int>{0, 1, 2, 3, 4, 5},
      DynamicArray<Real3>{
          {weight0, 0_r, 0_r},
          {0_r, weight0, 0_r},
          {0_r, 0_r, weight0},
          {weight1, 0_r, 0_r},
          {0_r, weight1, 0_r},
          {0_r, 0_r, weight1}});
}

ShapeHandle CreateUnitCubeTetMeshShapeWithSkinning(Context* context) {
  auto&& [coords, connectivity] = test::CreateMinimalTetMeshUnitCube();
  auto mesh = std::make_shared<TetrahedralMesh const>(coords, connectivity);
  auto skinning = std::make_shared<SkinningData const>(
      test::MakeSingleBoneSkinning(mesh->GetNumNodes(), /*boneIndex=*/0));
  auto shape = std::make_shared<TetrahedralMeshShape>(mesh, skinning);
  return assert_cast<ContextImpl*>(context)->RegisterShape(shape, test::ExpectOK{});
}

ShapeHandle CreateUnitCubeTetMeshShapeWithTwoBoneSkinning(Context* context) {
  auto&& [coords, connectivity] = test::CreateMinimalTetMeshUnitCube();
  auto mesh = std::make_shared<TetrahedralMesh const>(coords, connectivity);
  SkinningData skinningData;
  skinningData.weightsPerNode = 2;
  skinningData.indices.resize(2 * mesh->GetNumNodes());
  skinningData.weights.resize(2 * mesh->GetNumNodes());
  for (int node = 0; node < mesh->GetNumNodes(); ++node) {
    skinningData.indices[2 * node] = 0;
    skinningData.indices[2 * node + 1] = 1;
    skinningData.weights[2 * node] = 0.25_r;
    skinningData.weights[2 * node + 1] = 0.75_r;
  }
  auto shape = std::make_shared<TetrahedralMeshShape>(
      mesh, std::make_shared<SkinningData const>(std::move(skinningData)));
  return assert_cast<ContextImpl*>(context)->RegisterShape(shape, test::ExpectOK{});
}

std::shared_ptr<BlendingDataMap const>
MakeUniformBlendingMap(DynamicString const& softName, int numTargetNodes, real weight) {
  BlendingDataTargetMesh target;
  target.indices.resize(numTargetNodes);
  target.weights.resize(numTargetNodes);
  for (int targetNode = 0; targetNode < numTargetNodes; ++targetNode) {
    target.indices[targetNode] = targetNode;
    target.weights[targetNode] = weight;
  }
  auto map = std::make_shared<BlendingDataMap>();
  map->perSourceShapeData.emplace(softName, std::move(target));
  return map;
}

ShapeHandle
CreateUnitCubeTetBlendedSkinShape(Context* context, DynamicString const& softName, real weight) {
  auto&& [coords, connectivity] = test::CreateMinimalTetMeshUnitCube();
  auto mesh = std::make_shared<TetrahedralMesh const>(coords, connectivity);
  int const numNodes = mesh->GetNumNodes();
  auto skinning =
      std::make_shared<SkinningData const>(test::MakeSingleBoneSkinning(numNodes, /*boneIndex=*/0));
  auto shape = std::make_shared<TetrahedralMeshShape>(
      mesh,
      skinning,
      /*constrainedNodesData=*/nullptr,
      MakeUniformBlendingMap(softName, numNodes, weight));
  return assert_cast<ContextImpl*>(context)->RegisterShape(shape, test::ExpectOK{});
}

class ConservativeStepBoundsVelocityTest : public test::MochiSceneTestBase {
 protected:
  static constexpr real kBoundsTimeStep = 0.1_r;
  static constexpr real kAngularSpeed = 8_r;
  static_assert(kBoundsTimeStep * kAngularSpeed < 1_r);

  static real TransverseVSym(real angularSpeed, real timeStep = kBoundsTimeStep) {
    return (Sqrt(1_r - Sqr(timeStep * angularSpeed)) - 1_r) / timeStep;
  }

  // At the far unit-cube corner, include the finite-step rotation/soft-velocity cross-term.
  static real BlendedFullGradientMaxSpeed(real softWeight, real softVelocityY) {
    real const vSym = TransverseVSym(kAngularSpeed);
    real const weightedSoftVelocityY = softWeight * softVelocityY;
    real const velocityX =
        -kAngularSpeed + 0.5_r * vSym - kBoundsTimeStep * kAngularSpeed * weightedSoftVelocityY;
    real const velocityY =
        kAngularSpeed + 0.5_r * vSym + (1_r + kBoundsTimeStep * vSym) * weightedSoftVelocityY;
    return Sqrt(Sqr(velocityX) + Sqr(velocityY));
  }

  static real FullGradientMaxSpeed() {
    return BlendedFullGradientMaxSpeed(0_r, 0_r);
  }

  // Component-wise rigid AABB bounds add the transverse gradient magnitudes at half-extents 0.5.
  static real RigidRotationBoundMaxSpeed(real angularSpeed) {
    return Sqrt(2_r) * 0.5_r * (angularSpeed + Abs(TransverseVSym(angularSpeed)));
  }

  ArticulatedActorParams MakeArticulatedSkinParams(ShapeHandle skinShape) {
    ArticulatedActorParams params;
    params.joints = {{.type = ArticulatedJointType::Free}};
    params.links = {
        {.parentLink = -1,
         .shape = test::CreateUnitCubeTetMeshShape(_mochiContext),
         .colliderType = ColliderType::None,
         .centerOfMass = Real3{0.5_r, 0.5_r, 0.5_r},
         .momentOfInertia = Real6{1_r, 0_r, 0_r, 1_r, 0_r, 1_r}}};
    params.skin = ArticulatedSkinParams{.shape = skinShape};
    return params;
  }

  ArticulatedActorParams MakeTwoLinkArticulatedSkinParams(ShapeHandle skinShape) {
    ArticulatedActorParams params;
    params.joints = {
        {.type = ArticulatedJointType::Free},
        {.type = ArticulatedJointType::Prismatic, .axis = Real3{0_r, 1_r, 0_r}},
    };
    ShapeHandle const linkShape = test::CreateUnitCubeTetMeshShape(_mochiContext);
    params.links = {
        {.parentLink = -1, .shape = linkShape, .colliderType = ColliderType::None},
        {.parentLink = 0, .shape = linkShape, .colliderType = ColliderType::None},
    };
    params.skin = ArticulatedSkinParams{.shape = skinShape};
    return params;
  }

  SoftSkinnedActorParams MakeSoftSkinnedParams(std::optional<ShapeHandle> skinShape) {
    SoftSkinnedActorParams params;
    params.skeletonParams.joints = {{.type = ArticulatedJointType::Free}};
    params.skeletonParams.links = {
        {.parentLink = -1,
         .shape = test::CreateUnitCubeTetMeshShape(_mochiContext),
         .colliderType = ColliderType::None,
         .centerOfMass = Real3{0.5_r, 0.5_r, 0.5_r},
         .momentOfInertia = Real6{1_r, 0_r, 0_r, 1_r, 0_r, 1_r}}};
    if (skinShape) {
      params.skeletonParams.skin = ArticulatedSkinParams{.shape = *skinShape};
    }
    SoftActorParams softParams;
    softParams.name = "soft";
    softParams.shape = test::CreateUnitCubeTetSoftShape(_mochiContext);
    softParams.hasGravity = false;
    params.softParams = {softParams};
    return params;
  }

  void PrepareVelocityGradientsForBounds() {
    auto& reg = GetRegistry();
    reg.ctx<CSceneTime>().Reset(
        /*totalSeconds=*/0.0,
        /*deltaSecondsPrev=*/CSceneTime::kDefaultTimeStep,
        /*deltaSeconds=*/kBoundsTimeStep);
    ecs::InvokeForEachGlobal(&rigid::UpdateVSym, reg);
    ecs::InvokeForEachGlobal(&articulated::compound::UpdateVSym, reg);

    auto const articulatedView = reg.view<CArticulatedLinkVels const, CGroupMembers const>();
    for (entt::entity const articulated : articulatedView) {
      auto const& linkVelocities = articulatedView.get<CArticulatedLinkVels const>(articulated);
      auto const& links = articulatedView.get<CGroupMembers const>(articulated).actors;
      ASSERT_EQ(linkVelocities.size(), links.size());
      for (int i = 0; i < isize(links); ++i) {
        EXPECT_FALSE(linkVelocities[i].IsVSymDirty());
        EXPECT_FALSE(reg.get<CRigidVel<TimeStep::Current> const>(links[i]).value.IsVSymDirty());
      }
    }
  }

  void SetNestedSoftVelocity(Actor* nested, Real3 velocity) {
    auto& values =
        GetRegistry().get<CVelocitySlice<real, TimeStep::Current>>(GetEntity(nested)).value;
    for (int node = 0; node < values.Rows() / kSpaceDim3; ++node) {
      for (int axis = 0; axis < kSpaceDim3; ++axis) {
        values[kSpaceDim3 * node + axis] = velocity[axis];
      }
    }
  }

  void ExpectBoundsContainPostStepUnitCube(entt::entity actor) {
    auto const& reg = GetRegistry();
    Aabb const bounds = reg.get<CConservativeStepBounds const>(actor).worldAabb;
    TransformRT const& worldFromLocal = reg.get<CRootTransform const>(actor).worldFromLocal;
    auto const cube = test::CreateMinimalTetMeshUnitCube();
    for (Real3 const& vertexLocal : cube.first) {
      EXPECT_TRUE(ContainsPoint(bounds, worldFromLocal.TransformPoint(vertexLocal)));
    }
  }

  void ExpectBoundsContainPostStepSkin(entt::entity actor) {
    auto const& reg = GetRegistry();
    Aabb const bounds = reg.get<CConservativeStepBounds const>(actor).worldAabb;
    auto const& rest = reg.get<CArticulatedSkinningData const>(actor).restCoords;
    auto const& displacement =
        reg.get<CDisplacementSlice<real, TimeStep::Current, DisplacementLayer::Skinned> const>(
               actor)
            .value;
    ColumnVector<real> positions(rest.Rows());
    positions = rest + displacement;
    TransformRT const& worldFromLocal = reg.get<CRootTransform const>(actor).worldFromLocal;
    for (Real3 const& positionLocal : Unflatten<Real3>(positions.GetSpan())) {
      EXPECT_TRUE(ContainsPoint(bounds, worldFromLocal.TransformPoint(positionLocal)));
    }
  }
};
} // namespace

TEST_F(ConservativeStepBoundsVelocityTest, RigidMaxGeometrySpeedMatchesLinearSpeed) {
  RigidActorParams params;
  params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  params.colliderType = ColliderType::None;
  params.hasGravity = false;
  Actor* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
  entt::entity const actorEntity = GetEntity(actor);

  actor->SetVelocity({2_r, -3_r, 6_r}, {}, test::ExpectOK{});
  PrepareVelocityGradientsForBounds();
  UpdateMaxGeometrySpeeds(GetRegistry());

  EXPECT_NEAR_EQ(
      7_r, GetRegistry().get<CConservativeStepBounds const>(actorEntity).maxGeometrySpeed);
}

TEST(ConservativeStepBoundsVelocitySystemTest, DeformableMaxGeometrySpeedUsesEuclideanNorm) {
  CVelocitySlice<real, TimeStep::Current> velocity(6);
  velocity.value[3] = 3_r;
  velocity.value[4] = -4_r;
  CColliderInfo const collider;
  CConservativeStepBounds stepBounds;
  stepBounds.maxGeometrySpeed = std::numeric_limits<real>::quiet_NaN();

  deformable::UpdateMaxGeometrySpeed({}, {}, velocity, collider, nullptr, stepBounds);

  EXPECT_NEAR_EQ(5_r, stepBounds.maxGeometrySpeed);
}

TEST(ConservativeStepBoundsVelocitySystemTest, DeformableSkinOnlyMaxGeometrySpeedUsesContactSkin) {
  CVelocitySlice<real, TimeStep::Current> velocity(6);
  velocity.value[0] = 1_r;
  velocity.value[3] = -1_r;
  CColliderInfo const collider;
  CContactSkinningData contactSkinning;
  InitializeTwoNodeContactSkinningJacobian(2_r, -1_r, contactSkinning);
  CConservativeStepBounds stepBounds;

  deformable::UpdateMaxGeometrySpeed({}, {}, velocity, collider, &contactSkinning, stepBounds);

  EXPECT_NEAR_EQ(3_r, stepBounds.maxGeometrySpeed);
}

TEST(ConservativeStepBoundsVelocitySystemTest, DeformableSkinOnlyExcludesPhysicsMeshSpeed) {
  CVelocitySlice<real, TimeStep::Current> velocity(6);
  velocity.value[0] = 5_r;
  velocity.value[3] = -3_r;
  CColliderInfo const collider;
  CContactSkinningData contactSkinning;
  InitializeTwoNodeContactSkinningJacobian(0.5_r, 0.5_r, contactSkinning);
  CConservativeStepBounds stepBounds;

  deformable::UpdateMaxGeometrySpeed({}, {}, velocity, collider, &contactSkinning, stepBounds);

  EXPECT_NEAR_EQ(1_r, stepBounds.maxGeometrySpeed);
}

TEST(ConservativeStepBoundsVelocitySystemTest, DeformableSdfColliderIncludesPhysicsMeshSpeed) {
  CVelocitySlice<real, TimeStep::Current> velocity(6);
  velocity.value[0] = 5_r;
  velocity.value[3] = -3_r;
  CColliderInfo collider;
  collider.type = ColliderType::Sdf;
  CContactSkinningData contactSkinning;
  InitializeTwoNodeContactSkinningJacobian(0.5_r, 0.5_r, contactSkinning);
  CConservativeStepBounds stepBounds;

  deformable::UpdateMaxGeometrySpeed({}, {}, velocity, collider, &contactSkinning, stepBounds);

  EXPECT_NEAR_EQ(5_r, stepBounds.maxGeometrySpeed);
}

TEST_F(ConservativeStepBoundsVelocityTest, ShellMaxGeometrySpeedUsesDeformableDispatch) {
  auto&& [coords, connectivity] = test::CreateMinimalTriMeshUnitCube();
  experimental::ShellActorParams params;
  params.shape = _mochiContext->CreateTriMeshShape(
      Flatten(MakeSpan(coords)), Flatten(MakeSpan(connectivity)), test::ExpectOK{});
  params.hasGravity = false;
  params.colliderType = ColliderType::None;
  Actor* actor = experimental::CreateShellActor(_scene, params, test::ExpectOK{});
  entt::entity const actorEntity = GetEntity(actor);

  DynamicArray<real> velocity(actor->GetNumDofs(), 0_r);
  velocity[3] = 3_r;
  velocity[4] = -4_r;
  actor->SetNodeVelocitiesLocal(MakeConstSpan(velocity), test::ExpectOK{});
  UpdateMaxGeometrySpeeds(GetRegistry());

  EXPECT_NEAR_EQ(
      5_r, GetRegistry().get<CConservativeStepBounds const>(actorEntity).maxGeometrySpeed);
}

TEST_F(ConservativeStepBoundsVelocityTest, RigidMaxGeometrySpeedRotationAndRecomputation) {
  RigidActorParams params;
  params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  params.colliderType = ColliderType::None;
  params.hasGravity = false;
  Actor* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
  entt::entity const actorEntity = GetEntity(actor);

  actor->SetVelocity({}, {0_r, 0_r, 2_r}, test::ExpectOK{});
  PrepareVelocityGradientsForBounds();
  UpdateMaxGeometrySpeeds(GetRegistry());
  EXPECT_NEAR_EQ(
      RigidRotationBoundMaxSpeed(2_r),
      GetRegistry().get<CConservativeStepBounds const>(actorEntity).maxGeometrySpeed);

  actor->SetVelocity({}, {}, test::ExpectOK{});
  PrepareVelocityGradientsForBounds();
  UpdateMaxGeometrySpeeds(GetRegistry());
  EXPECT_NEAR_EQ(
      0_r, GetRegistry().get<CConservativeStepBounds const>(actorEntity).maxGeometrySpeed);
}

TEST_F(ConservativeStepBoundsVelocityTest, SoftMaxGeometrySpeedUsesResolvedNodalVelocity) {
  SoftActorParams params;
  params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  params.hasGravity = false;
  params.hasStress = false;
  Actor* actor = _scene->CreateSoftActor(params, test::ExpectOK{});
  entt::entity const actorEntity = GetEntity(actor);

  DynamicArray<real> velocity(actor->GetNumDofs(), 0_r);
  for (int i = 0; i < isize(velocity); i += kSpaceDim3) {
    velocity[i] = 1_r;
    velocity[i + 1] = 2_r;
    velocity[i + 2] = 2_r;
  }
  actor->SetNodeVelocitiesLocal(MakeConstSpan(velocity), test::ExpectOK{});
  UpdateMaxGeometrySpeeds(GetRegistry());

  EXPECT_NEAR_EQ(
      3_r, GetRegistry().get<CConservativeStepBounds const>(actorEntity).maxGeometrySpeed);
}

TEST_F(ConservativeStepBoundsVelocityTest, ArticulatedSkinMaxGeometrySpeedUsesSkinVelocity) {
  Actor* actor = _scene->CreateArticulatedActor(
      MakeArticulatedSkinParams(CreateUnitCubeTetMeshShapeWithSkinning(_mochiContext)),
      test::ExpectOK{});
  DynamicArray<real> velocity(actor->GetNumDofs(), 0_r);
  velocity[0] = 3_r;
  velocity[1] = 4_r;
  actor->SetArticulatedJointVelocities(velocity, test::ExpectOK{});

  PrepareVelocityGradientsForBounds();
  UpdateMaxGeometrySpeeds(GetRegistry());

  EXPECT_NEAR_EQ(
      5_r, GetRegistry().get<CConservativeStepBounds const>(GetEntity(actor)).maxGeometrySpeed);
}

TEST_F(ConservativeStepBoundsVelocityTest, ArticulatedSkinMaxGeometrySpeedUsesFullGradient) {
  Actor* actor = _scene->CreateArticulatedActor(
      MakeArticulatedSkinParams(CreateUnitCubeTetMeshShapeWithSkinning(_mochiContext)),
      test::ExpectOK{});
  DynamicArray<real> velocity(actor->GetNumDofs(), 0_r);
  velocity[5] = kAngularSpeed;
  actor->SetArticulatedJointVelocities(velocity, test::ExpectOK{});

  PrepareVelocityGradientsForBounds();
  UpdateMaxGeometrySpeeds(GetRegistry());

  real const maxGeometrySpeed =
      GetRegistry().get<CConservativeStepBounds const>(GetEntity(actor)).maxGeometrySpeed;
  EXPECT_NEAR_EQ(FullGradientMaxSpeed(), maxGeometrySpeed);
  EXPECT_GT(maxGeometrySpeed, Sqrt(2_r) * kAngularSpeed);
}

TEST_F(ConservativeStepBoundsVelocityTest, ArticulatedSkinUsesActorOrderForMultipleBones) {
  Actor* actor = _scene->CreateArticulatedActor(
      MakeTwoLinkArticulatedSkinParams(
          CreateUnitCubeTetMeshShapeWithTwoBoneSkinning(_mochiContext)),
      test::ExpectOK{});
  ASSERT_EQ(7, actor->GetNumDofs());
  DynamicArray<real> velocity(actor->GetNumDofs(), 0_r);
  velocity[0] = 3_r;
  velocity[6] = 4_r;
  actor->SetArticulatedJointVelocities(velocity, test::ExpectOK{});

  PrepareVelocityGradientsForBounds();
  UpdateMaxGeometrySpeeds(GetRegistry());

  // Bone velocities are (3, 0, 0) and (3, 4, 0), blended with weights 0.25 and 0.75.
  EXPECT_NEAR_EQ(
      Sqrt(18_r),
      GetRegistry().get<CConservativeStepBounds const>(GetEntity(actor)).maxGeometrySpeed);
}

TEST_F(ConservativeStepBoundsVelocityTest, NestedSoftMaxGeometrySpeedUsesFinalVelocity) {
  Actor* parent =
      _scene->CreateSoftSkinnedActor(MakeSoftSkinnedParams(std::nullopt), test::ExpectOK{});
  ASSERT_EQ(1, isize(parent->GetNestedSoftActors(test::ExpectOK{})));
  Actor* nested = _scene->GetActor(parent->GetNestedSoftActors(test::ExpectOK{})[0]);
  SetNestedSoftVelocity(nested, {3_r, 4_r, 0_r});

  PrepareVelocityGradientsForBounds();
  UpdateMaxGeometrySpeeds(GetRegistry());

  EXPECT_NEAR_EQ(
      5_r, GetRegistry().get<CConservativeStepBounds const>(GetEntity(nested)).maxGeometrySpeed);
}

TEST_F(ConservativeStepBoundsVelocityTest, NestedSoftMaxGeometrySpeedUsesFullParentGradient) {
  Actor* parent =
      _scene->CreateSoftSkinnedActor(MakeSoftSkinnedParams(std::nullopt), test::ExpectOK{});
  ASSERT_EQ(1, isize(parent->GetNestedSoftActors(test::ExpectOK{})));
  Actor* nested = _scene->GetActor(parent->GetNestedSoftActors(test::ExpectOK{})[0]);
  DynamicArray<real> velocity(parent->GetNumDofs(), 0_r);
  velocity[5] = kAngularSpeed;
  parent->SetArticulatedJointVelocities(velocity, test::ExpectOK{});
  SetNestedSoftVelocity(nested, {});

  PrepareVelocityGradientsForBounds();
  UpdateMaxGeometrySpeeds(GetRegistry());

  EXPECT_NEAR_EQ(
      FullGradientMaxSpeed(),
      GetRegistry().get<CConservativeStepBounds const>(GetEntity(nested)).maxGeometrySpeed);
}

TEST_F(ConservativeStepBoundsVelocityTest, NestedSoftStateVelocityIsNotRecomputed) {
  SoftSkinnedActorParams params = MakeSoftSkinnedParams(std::nullopt);
  params.hasInertia = true;
  params.softParams[0].hasInertia = false;
  Actor* parent = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
  ASSERT_EQ(1, isize(parent->GetNestedSoftActors(test::ExpectOK{})));
  Actor* nested = _scene->GetActor(parent->GetNestedSoftActors(test::ExpectOK{})[0]);
  entt::entity const nestedEntity = GetEntity(nested);
  auto& reg = GetRegistry();
  ASSERT_TRUE(reg.all_of<CIntegrationVelocitySlices<DisplacementLayer::Skinned>>(nestedEntity));

  DynamicArray<real> articulatedVelocity(parent->GetNumDofs(), 0_r);
  articulatedVelocity[5] = kAngularSpeed;
  parent->SetArticulatedJointVelocities(articulatedVelocity, test::ExpectOK{});
  PrepareVelocityGradientsForBounds();

  SetNestedSoftVelocity(nested, {20_r, 0_r, 0_r});
  auto& skinnedVelocity =
      reg.get<CVelocitySlice<real, TimeStep::Current, DisplacementLayer::Skinned>>(nestedEntity)
          .value;
  for (int node = 0; node < skinnedVelocity.Rows() / kSpaceDim3; ++node) {
    skinnedVelocity[kSpaceDim3 * node] = 0_r;
    skinnedVelocity[kSpaceDim3 * node + 1] = 3_r;
    skinnedVelocity[kSpaceDim3 * node + 2] = 4_r;
  }
  ColumnVector<real> const expectedVelocity = skinnedVelocity;
  UpdateMaxGeometrySpeeds(reg);

  EXPECT_SPAN_EQ(expectedVelocity.GetConstSpan(), skinnedVelocity.GetConstSpan());
  EXPECT_NEAR_EQ(5_r, reg.get<CConservativeStepBounds const>(nestedEntity).maxGeometrySpeed);
}

TEST_F(ConservativeStepBoundsVelocityTest, BlendedMaxGeometrySpeedUsesFinalVelocity) {
  real constexpr kSoftWeight = 0.5_r;
  ShapeHandle const skin =
      CreateUnitCubeTetBlendedSkinShape(_mochiContext, DynamicString{"soft"}, kSoftWeight);
  Actor* parent = _scene->CreateSoftSkinnedActor(MakeSoftSkinnedParams(skin), test::ExpectOK{});
  ASSERT_EQ(1, isize(parent->GetNestedSoftActors(test::ExpectOK{})));
  Actor* nested = _scene->GetActor(parent->GetNestedSoftActors(test::ExpectOK{})[0]);

  DynamicArray<real> articulatedVelocity(parent->GetNumDofs(), 0_r);
  articulatedVelocity[0] = 1_r;
  parent->SetArticulatedJointVelocities(articulatedVelocity, test::ExpectOK{});
  SetNestedSoftVelocity(nested, {0_r, 2_r, 0_r});

  PrepareVelocityGradientsForBounds();
  UpdateMaxGeometrySpeeds(GetRegistry());

  EXPECT_NEAR_EQ(
      Sqrt(2_r),
      GetRegistry().get<CConservativeStepBounds const>(GetEntity(parent)).maxGeometrySpeed);
}

TEST_F(ConservativeStepBoundsVelocityTest, BlendedMaxGeometrySpeedUsesFullGradientInputs) {
  real constexpr kSoftWeight = 0.5_r;
  ShapeHandle const skin =
      CreateUnitCubeTetBlendedSkinShape(_mochiContext, DynamicString{"soft"}, kSoftWeight);
  Actor* parent = _scene->CreateSoftSkinnedActor(MakeSoftSkinnedParams(skin), test::ExpectOK{});
  ASSERT_EQ(1, isize(parent->GetNestedSoftActors(test::ExpectOK{})));
  Actor* nested = _scene->GetActor(parent->GetNestedSoftActors(test::ExpectOK{})[0]);

  DynamicArray<real> articulatedVelocity(parent->GetNumDofs(), 0_r);
  articulatedVelocity[5] = kAngularSpeed;
  parent->SetArticulatedJointVelocities(articulatedVelocity, test::ExpectOK{});
  SetNestedSoftVelocity(nested, {0_r, 2_r, 0_r});

  PrepareVelocityGradientsForBounds();
  UpdateMaxGeometrySpeeds(GetRegistry());

  EXPECT_NEAR_EQ(
      BlendedFullGradientMaxSpeed(kSoftWeight, 2_r),
      GetRegistry().get<CConservativeStepBounds const>(GetEntity(parent)).maxGeometrySpeed);
}

TEST_F(ConservativeStepBoundsVelocityTest, BlendedBoundsContainCombinedMotion) {
  real constexpr kTimeStep = 0.01_r;
  real constexpr kAngularSpeed = 5_r;
  ShapeHandle const skin =
      CreateUnitCubeTetBlendedSkinShape(_mochiContext, DynamicString{"soft"}, 1_r);
  SoftSkinnedActorParams params = MakeSoftSkinnedParams(skin);
  Real3 const centerOfMass = *params.skeletonParams.links[0].centerOfMass;
  params.skeletonParams.links[0].hasGravity = false;
  Actor* parent = _scene->CreateSoftSkinnedActor(params, test::ExpectOK{});
  ASSERT_EQ(1, isize(parent->GetNestedSoftActors(test::ExpectOK{})));
  Actor* nested = _scene->GetActor(parent->GetNestedSoftActors(test::ExpectOK{})[0]);

  DynamicArray<real> articulatedVelocity(parent->GetNumDofs(), 0_r);
  articulatedVelocity[5] = kAngularSpeed;
  parent->SetArticulatedJointVelocities(articulatedVelocity, test::ExpectOK{});

  real const vSym = TransverseVSym(kAngularSpeed, kTimeStep);
  auto const coords = test::CreateMinimalTetMeshUnitCube().first;
  auto& softVelocity =
      GetRegistry().get<CVelocitySlice<real, TimeStep::Current>>(GetEntity(nested)).value;
  ASSERT_EQ(kSpaceDim3 * isize(coords), softVelocity.Rows());
  for (int node = 0; node < isize(coords); ++node) {
    Real3 const position = coords[node] - centerOfMass;
    softVelocity[kSpaceDim3 * node] = -vSym * position[0] + kAngularSpeed * position[1];
    softVelocity[kSpaceDim3 * node + 1] = -kAngularSpeed * position[0] - vSym * position[1];
    softVelocity[kSpaceDim3 * node + 2] = 0_r;
  }

  entt::entity const parentEntity = GetEntity(parent);
  GetRegistry().get<CConservativeStepBounds>(parentEntity).needsNextStepRelaxation = false;
  _scene->Step(kTimeStep);

  ExpectBoundsContainPostStepSkin(parentEntity);
}

TEST_F(ConservativeStepBoundsVelocityTest, RigidLinearVelocity) {
  RigidActorParams params;
  params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  params.colliderType = ColliderType::None;
  params.hasGravity = false;
  Actor* actor = _scene->CreateRigidActor(params, test::ExpectOK{});
  entt::entity const actorEntity = GetEntity(actor);

  actor->SetVelocity({12_r, 16_r, 0_r}, {}, test::ExpectOK{});
  GetRegistry().get<CConservativeStepBounds>(actorEntity).needsNextStepRelaxation = false;

  _scene->Step(1e-2);

  EXPECT_NEAR_EQ(
      20_r, GetRegistry().get<CConservativeStepBounds const>(actorEntity).maxGeometrySpeed);
  ExpectBoundsContainPostStepUnitCube(actorEntity);
}

TEST_F(ConservativeStepBoundsVelocityTest, RigidAngularVelocity) {
  RigidActorParams params;
  params.shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  params.colliderType = ColliderType::None;
  params.hasGravity = false;
  params.momentOfInertia = Real6{10_r, 0_r, 0_r, 10_r, 0_r, 10_r};

  // Offset center-of-mass rotation exercises the AABB-center velocity contribution.
  params.centerOfMass = Real3{3_r, 0.5_r, 0.5_r};
  Actor* offsetActor = _scene->CreateRigidActor(params, test::ExpectOK{});
  entt::entity const offsetActorEntity = GetEntity(offsetActor);
  offsetActor->SetVelocity({}, {0_r, 0_r, 10_r}, test::ExpectOK{});
  GetRegistry().get<CConservativeStepBounds>(offsetActorEntity).needsNextStepRelaxation = false;

  // Centered rotation at this speed makes containment depend on the velocity-radius contribution.
  params.centerOfMass = Real3{0.5_r, 0.5_r, 0.5_r};
  Actor* centeredActor = _scene->CreateRigidActor(params, test::ExpectOK{});
  entt::entity const centeredActorEntity = GetEntity(centeredActor);
  centeredActor->SetVelocity({}, {0_r, 0_r, 20_r}, test::ExpectOK{});
  GetRegistry().get<CConservativeStepBounds>(centeredActorEntity).needsNextStepRelaxation = false;

  _scene->Step(1e-2);

  ExpectBoundsContainPostStepUnitCube(offsetActorEntity);
  ExpectBoundsContainPostStepUnitCube(centeredActorEntity);
}

TEST_F(ConservativeStepBoundsVelocityTest, ArticulatedSkinLargeRotation) {
  ArticulatedActorParams params =
      MakeArticulatedSkinParams(CreateUnitCubeTetMeshShapeWithSkinning(_mochiContext));
  params.links[0].hasGravity = false;
  params.jointVelocities = DynamicArray<real>{0_r, 0_r, 0_r, 0_r, 0_r, kAngularSpeed};
  Actor* actor = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  entt::entity const actorEntity = GetEntity(actor);

  _scene->Step(kBoundsTimeStep);
  ASSERT_FALSE(
      GetRegistry().get<CConservativeStepBounds const>(actorEntity).needsNextStepRelaxation);

  _scene->Step(kBoundsTimeStep);
  _scene->Step(kBoundsTimeStep);

  auto const& linkTransforms =
      GetRegistry().get<CArticulatedLinkTransforms<TimeStep::Current> const>(actorEntity);
  ASSERT_EQ(1, isize(linkTransforms));
  EXPECT_GT(Norm(linkTransforms[0].GetRotation().ToRotationVector()), kPI / 2_r);
  ExpectBoundsContainPostStepSkin(actorEntity);
}

TEST_F(ConservativeStepBoundsVelocityTest, ArticulatedLinkJointVelocity) {
  ShapeHandle const shape = test::CreateUnitCubeTetMeshShape(_mochiContext);
  ArticulatedActorParams params;
  params.joints = {
      {.type = ArticulatedJointType::Hard},
      {.type = ArticulatedJointType::Revolute, .axis = {0_r, 0_r, 1_r}}};
  params.links = {
      {.parentLink = -1, .shape = shape, .colliderType = ColliderType::None, .hasGravity = false},
      {.parentLink = 0, .shape = shape, .colliderType = ColliderType::None, .hasGravity = false}};
  Actor* actor = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  auto const& links = actor->GetNestedLinkActors(test::ExpectOK{});
  ASSERT_EQ(2, isize(links));

  DynamicArray<real> const velocities = {20_r};
  actor->SetArticulatedJointVelocities(velocities, test::ExpectOK{});
  for (ActorHandle const link : links) {
    GetRegistry().get<CConservativeStepBounds>(GetEntity(link)).needsNextStepRelaxation = false;
  }

  _scene->Step(1e-2);

  ExpectBoundsContainPostStepUnitCube(GetEntity(links[1]));
}
