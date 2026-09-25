# FR3 V2 with Wuji Hand 2 (Beta 1)

This directory contains SuperDex assemblies for both hands:

- Left: [`left/fr3_v2_wuji2_beta1_left.superdex_bot`](left/fr3_v2_wuji2_beta1_left.superdex_bot)
- Right: [`right/fr3_v2_wuji2_beta1_right.superdex_bot`](right/fr3_v2_wuji2_beta1_right.superdex_bot)

Each references:

- Franka Research 3 (FR3) V2 arm: [`../../arms/fr3_v2`](../../arms/fr3_v2) - see its README, LICENSE, and NOTICE.
- Wuji Hand 2 (Beta 1), skinned: [`../../hands/wuji_hand2_beta1`](../../hands/wuji_hand2_beta1) - see its README, LICENSE, and NOTICE.

The angled mount that joins the hand to the FR3 V2 tool flange is Meta proprietary and is distributed under CC BY 4.0; see [`LICENSE`](LICENSE). It is supplied per side as `Wuji_AngledMounted_30_10_Left` / `Wuji_AngledMounted_30_10_Right`, with a render mesh under `render/`, a collision mesh under `collision/`, and its mesh-processing recipe under `intermediates/`.

**You are responsible for ensuring your use is compatible with all third-party licenses of the referenced sub-assets.**
