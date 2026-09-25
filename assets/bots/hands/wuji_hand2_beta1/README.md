# Wuji Hand 2 (Beta 1) Description

## Overview

This package contains a simplified robot description of the [Wuji Hand 2 (Beta 1)](https://github.com/wuji-technology/wuji-description/tree/1407beed7f478f6ba472d1c42bbdee6f4eec8f7f/hand2/hand2_beta1), based on URDF and STEP assets from the upstream [wuji-description](https://github.com/wuji-technology/wuji-description) repository.

## Modifications

One or more of the following modifications were made to adapt the original assets:

- Simplified and/or re-authored collision geometry for compatibility with SuperDex Physics.
- Adjusted kinematic parameters (relative joint/link transforms, axes, and limits) for SuperDex Physics conventions.
- Adjusted visual geometry and/or materials to improve rendering fidelity.
- Added or modified actuator descriptions.
- Added or modified sensor descriptions where applicable.

## Skinned Variants

This package also provides skinned descriptions, which cover the articulated hand with a deformable skin:

- [`left/wuji_hand2_beta1_skinned_left.superdex_bot`](left/wuji_hand2_beta1_skinned_left.superdex_bot)
- [`right/wuji_hand2_beta1_skinned_right.superdex_bot`](right/wuji_hand2_beta1_skinned_right.superdex_bot)

Each attaches a skin render mesh (`render/skin_render.glb`) and a skin collision mesh (`collision/skin_low_poly_cut_tips_collision.mochi.h5`) from its own side's directory.

Skinned render assets were derived from 3D scans of the Wuji v2 Beta 2 hand, retopologized and bound to the skeleton by a professional technical artist.

## License

The upstream Wuji Hand 2 assets are provided under the [MIT License](https://github.com/wuji-technology/wuji-description/blob/main/LICENSE) (Copyright (c) 2025 Wuji Technology).

The skin assets used by the skinned variants were created by Meta Platforms, Inc. and affiliates and are released under the same MIT License.

**You are responsible for ensuring your use is compatible with all third-party licenses.**
