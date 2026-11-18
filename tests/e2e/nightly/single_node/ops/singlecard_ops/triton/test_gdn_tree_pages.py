# SPDX-License-Identifier: Apache-2.0
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.spec_decode.tree.gdn_tree_pages import GDNTreePagePointers


def test_refresh_tree_pages_across_steps():
    updater = GDNTreePagePointers()
    slots = 33
    rows = []
    for group in range(10):
        src = (torch.arange(40, device="npu", dtype=torch.int32) * (group + 3) + 5400 * group).view(1, -1)
        rows.append((src, torch.full_like(src, -1), torch.full_like(src, -1)))

    for step in range(4):
        cols = torch.tensor([[step + 1] + [i // (step + 1) for i in range(slots - 1)]], device="npu")
        if step == 3:
            src, dst, init = rows[-1]
            rows[-1] = (src.clone().add_(7777), dst, init)
        for src, _, _ in rows:
            src.add_(7)
        updater.refresh(cols, [(src, dst[:, :slots], init[:, :slots]) for src, dst, init in rows])
        for src, dst, init in rows:
            torch.testing.assert_close(dst[:, :slots], src[:, :slots])
            torch.testing.assert_close(init[:, :slots], src.gather(1, cols))
            assert (dst[:, slots:] == -1).all() and (init[:, slots:] == -1).all()
