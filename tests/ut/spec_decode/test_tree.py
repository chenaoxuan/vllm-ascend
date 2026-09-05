from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from vllm_ascend.worker.v2.input_batch import prepare_tree_spec_pos_seq_lens
from vllm_ascend.worker.v2.spec_decode.tree.beam import BeamTreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import (
    compact_tree_kv_along_path,
    compact_tree_query_along_path,
    mask_rejected_dflash_context_slots,
)
from vllm_ascend.worker.v2.spec_decode.tree.layout import empty_tree_layout
from vllm_ascend.worker.v2.spec_decode.tree.prefix import PrefixTreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.priority import PriorityTreeBuilder


def test_batch_spine_and_sibling_trees() -> None:
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            [
                [3.0, 2.9, 0.0],
                [-20.0, -20.0, -20.0],
                [-20.0, -20.0, -20.0],
            ],
        ]
    )
    out = empty_tree_layout(2, 3, device=logits.device)
    layout = PriorityTreeBuilder(budget=3, topk=3).build(logits, out)

    assert layout is out
    assert layout.num_nodes.tolist() == [3, 3]

    assert layout.tokens[0].tolist() == [0, 1, 2]
    assert layout.depths[0].tolist() == [1, 2, 3]
    assert layout.parents[0].tolist() == [0, 1, 2]
    assert layout.visibility[0].tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]
    assert layout.first_child[0, :4].tolist() == [1, 2, 3, -1]
    assert layout.next_sibling[0, :4].tolist() == [-1, -1, -1, -1]

    assert layout.tokens[1].tolist() == [0, 1, 0]
    assert layout.depths[1].tolist() == [1, 1, 2]
    assert layout.parents[1].tolist() == [0, 0, 1]
    assert layout.visibility[1].tolist() == [
        [True, False, False],
        [False, True, False],
        [True, False, True],
    ]
    # Newest child is prepended: node 2 then sibling 1 under root.
    assert layout.first_child[1, :4].tolist() == [2, 3, -1, -1]
    assert layout.next_sibling[1, :4].tolist() == [-1, -1, 1, -1]


def test_build_beam_trees_with_markov() -> None:
    """DSpark beam: map draft→target before markov_embed; topk>1 fills budget.

    Zero markov bias keeps base-logit topology so parents/depths stay checkable,
    while ``seen`` proves the second depth embeds mapped target ids.
    """

    class _FakeDSparkDraft:
        def __init__(self, vocab: int) -> None:
            self.seen: list[torch.Tensor] = []
            self._vocab = vocab

        def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
            self.seen.append(token_ids.clone())
            return token_ids.new_zeros(*token_ids.shape, 2, dtype=torch.float32)

        def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
            return markov_embed.new_zeros(markov_embed.shape[0], self._vocab)

        def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
            return draft_ids + 10

    logits = torch.tensor(
        [
            [
                [100.0, 60.0, 20.0, 0.0, 0.0, 0.0],
                [100.0, 90.0, 80.0, 0.0, 0.0, 0.0],
                [100.0, 90.0, 80.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    budget = 9
    draft = _FakeDSparkDraft(vocab=6)
    out = empty_tree_layout(1, budget, device=logits.device)
    layout = BeamTreeBuilder(budget=budget, topk=3, draft_model=draft).build(
        logits,
        out,
        root_token_ids=torch.tensor([0]),
    )

    assert layout is out
    assert layout.num_nodes[0].tolist() == budget
    assert layout.tokens[0, :9].tolist() == [10, 11, 12, 10, 11, 12, 10, 11, 12]
    assert layout.depths[0, :9].tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert layout.parents[0, :9].tolist() == [0, 0, 0, 1, 1, 1, 4, 4, 4]
    assert layout.first_child[0, :10].tolist() == [3, 6, -1, -1, 9, -1, -1, -1, -1, -1]
    assert torch.equal(draft.seen[0], torch.tensor([[0]]))
    assert torch.equal(draft.seen[1], torch.tensor([[10, 11, 12]]))


def test_build_prefix_trees_with_domino_correction() -> None:
    """Domino scorer path: correction hooks + Top-B prune to budget."""

    class _FakeDominoScorer:
        gru_hidden_dim = 2

        def __init__(self, vocab: int) -> None:
            mid = 2
            self.fc2_weight = torch.zeros(vocab, mid)
            self.fc2_bias = None
            self.w_s = torch.zeros(mid, self.gru_hidden_dim)
            self.middle = nn.Identity()

        def project_z(self, parallel_hiddens: torch.Tensor) -> torch.Tensor:
            return parallel_hiddens.new_zeros(
                *parallel_hiddens.shape[:-1], self.fc2_weight.shape[-1]
            )

        def update_hidden(
            self, token_ids: torch.Tensor, h_state: torch.Tensor
        ) -> torch.Tensor:
            return h_state

    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            [
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
                [10.0, 0.0, 0.0],
            ],
        ]
    )
    scorer = _FakeDominoScorer(vocab=3)
    draft_hidden = torch.zeros(2, 3, 4)
    out = empty_tree_layout(2, budget=2, device=logits.device)
    layout = PrefixTreeBuilder(
        budget=2,
        topk=1,
        correction_scorer=scorer,
        prefix_len=0,
    ).build(
        logits,
        out,
        root_token_ids=torch.tensor([7, 8]),
        draft_hidden=draft_hidden,
    )

    assert layout is out
    assert layout.num_nodes.tolist() == [2, 2]
    assert layout.tokens[0, :2].tolist() == [0, 1]
    assert layout.parents[0, :2].tolist() == [0, 1]
    assert layout.depths[0, :2].tolist() == [1, 2]
    assert layout.tokens[1, :2].tolist() == [1, 2]
    assert layout.parents[1, :2].tolist() == [0, 1]
    assert layout.depths[1, :2].tolist() == [1, 2]


def test_tree_kv_slot_layout_and_compact() -> None:
    """Siblings share RoPE positions but unique KV slots; compact packs the path."""
    num_computed = 10
    query_len = 4
    tree_depths = torch.zeros((1, 8), dtype=torch.int32)
    tree_depths[0, :3] = torch.tensor([1, 1, 2], dtype=torch.int32)
    idx_mapping = torch.zeros(1, dtype=torch.int32)
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    is_prefilling = torch.zeros(1, dtype=torch.int32)
    computed = torch.tensor([num_computed], dtype=torch.int32)
    pos = torch.zeros(query_len, dtype=torch.int64)
    slot_pos = torch.zeros(query_len, dtype=torch.int64)
    seq_lens = torch.zeros(2, dtype=torch.int32)

    prepare_tree_spec_pos_seq_lens(
        idx_mapping,
        query_start_loc,
        is_prefilling,
        computed,
        tree_depths,
        pos,
        slot_pos,
        seq_lens,
    )
    assert pos.tolist() == [10, 11, 11, 12]
    assert slot_pos.tolist() == [10, 11, 12, 13]
    assert seq_lens[0].tolist() == 14

    block_size = 16
    cache = torch.arange(2 * block_size, dtype=torch.float32).view(2, block_size, 1)
    before = cache.clone()
    block_table = torch.zeros((1, 4), dtype=torch.int32)
    block_table[0, 1] = 1
    path_node_ids = torch.tensor([[2, -1, -1]], dtype=torch.long)
    compact_tree_kv_along_path(
        [cache],
        block_table,
        block_size,
        idx_mapping,
        computed,
        path_node_ids,
    )
    # node 2 lives at logical 12; accepted path depth 1 dest is logical 11.
    assert cache[0, 11].tolist() == before[0, 12].tolist()
    assert cache[0, 12].tolist() == before[0, 12].tolist()

    # Overlapping src/dst: gather then scatter so 13→12 does not read the
    # already-written 12→11 result.
    cache = before.clone()
    compact_tree_kv_along_path(
        [cache],
        block_table,
        block_size,
        idx_mapping,
        computed,
        torch.tensor([[2, 3, -1]], dtype=torch.long),
    )
    assert cache[0, 11].tolist() == before[0, 12].tolist()
    assert cache[0, 12].tolist() == before[0, 13].tolist()


def test_tree_query_compact_along_non_prefix_path() -> None:
    """Packed siblings are not a prefix; compact gathers path rows and linear RoPE."""
    query_len = 4
    hidden = torch.arange(query_len * 2, dtype=torch.float32).view(query_len, 2)
    aux = hidden.clone() + 100
    pos = torch.tensor([10, 11, 11, 12], dtype=torch.int64)
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    before_h = hidden.clone()
    before_aux = aux.clone()

    compact_tree_query_along_path(
        [hidden, aux],
        query_start_loc,
        torch.tensor([[2, -1, -1]], dtype=torch.long),
        linearize_positions=pos,
    )
    assert torch.equal(hidden[0], before_h[0])
    assert torch.equal(hidden[1], before_h[2])
    assert torch.equal(hidden[2], before_h[2])
    assert torch.equal(hidden[3], before_h[3])
    assert torch.equal(aux[1], before_aux[2])
    assert torch.equal(pos, torch.tensor([10, 11, 11, 12]))

    hidden = before_h.clone()
    pos = torch.tensor([10, 11, 11, 12], dtype=torch.int64)
    compact_tree_query_along_path(
        [hidden],
        query_start_loc,
        torch.tensor([[2, 1, -1]], dtype=torch.long),
        linearize_positions=pos,
    )
    assert torch.equal(hidden[0], before_h[0])
    assert torch.equal(hidden[1], before_h[2])
    assert torch.equal(hidden[2], before_h[1])
    assert torch.equal(hidden[3], before_h[3])
    assert torch.equal(pos, torch.tensor([10, 11, 12, 12]))

    # path=[1] with a same-depth leftover sibling: compact leaves the sibling
    # on the rejected suffix sharing RoPE 11. DFlash must PAD that suffix so
    # it cannot clobber the accepted node's draft KV slot.
    hidden = before_h.clone()
    pos = torch.tensor([10, 11, 11, 12], dtype=torch.int64)
    compact_tree_query_along_path(
        [hidden],
        query_start_loc,
        torch.tensor([[1, -1, -1]], dtype=torch.long),
        linearize_positions=pos,
    )
    assert torch.equal(hidden[0], before_h[0])
    assert torch.equal(hidden[1], before_h[1])
    assert torch.equal(hidden[2], before_h[2])
    assert torch.equal(pos, torch.tensor([10, 11, 11, 12]))
    slots = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
    mask_rejected_dflash_context_slots(
        slots,
        query_start_loc,
        torch.tensor([2], dtype=torch.int32),
        pad_slot_id=-1,
    )
    assert torch.equal(slots, torch.tensor([100, 101, -1, -1]))


def test_prefix_domino_shift_label_samples_bonus_hidden() -> None:
    """Domino+shift_label uses N queries and samples the bonus as slot 0.

    priority / no-shift_label keep vanilla DFlash 1+N (mask-only) sampling.
    """
    from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
        AscendDFlashSpeculator,
    )
    from vllm_ascend.worker.v2.spec_decode.tree.speculator import (
        AscendTreeSpeculator,
    )

    num_spec = 4
    device = torch.device("cpu")

    def _speculator(method: str, *, shift_label: bool, projector: str | None):
        hf = SimpleNamespace(
            dflash_config={
                "projector_type": projector,
                "shift_label": shift_label,
                "pure_draft_prefix_len": 1,
            }
        )
        vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(
                num_speculative_tokens=num_spec,
                use_dflash=lambda: True,
                use_dspark=lambda: False,
                draft_model_config=SimpleNamespace(hf_config=hf),
            )
        )

        def _parent_init(self, vllm_config, device):
            self.vllm_config = vllm_config
            self.device = device
            self.speculative_config = vllm_config.speculative_config
            self.num_speculative_steps = num_spec
            self.max_num_reqs = 2
            self.sample_from_anchor = False
            self.num_query_per_req = 1 + num_spec
            self.use_local_argmax_reduction = False
            self.draft_tokens = torch.zeros(2, num_spec, dtype=torch.long)

        tree_cfg = SimpleNamespace(method=method, budget=8, topk=4)
        with (
            patch.object(AscendDFlashSpeculator, "__init__", _parent_init),
            patch(
                "vllm_ascend.worker.v2.spec_decode.tree.speculator.get_ascend_config",
                return_value=SimpleNamespace(tree_spec_config=tree_cfg),
            ),
        ):
            return AscendTreeSpeculator(vllm_config, device)

    domino = _speculator("prefix", shift_label=True, projector="domino")
    assert domino.sample_from_anchor is True
    assert domino.num_query_per_req == num_spec
    assert domino._domino_shift_label is True

    priority = _speculator("priority", shift_label=True, projector="domino")
    assert priority.sample_from_anchor is False
    assert priority.num_query_per_req == 1 + num_spec

    no_shift = _speculator("prefix", shift_label=False, projector="domino")
    assert no_shift.sample_from_anchor is False
    assert no_shift.num_query_per_req == 1 + num_spec
