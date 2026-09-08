from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import (
    merge_draft_aclgraph_capture_sizes,
)


def test_draft_capture_sizes_include_query_len_when_list_is_target_only():
    """Tree target gears [17, 49] must still yield a 16-token Domino draft graph."""
    sizes = merge_draft_aclgraph_capture_sizes(
        [17, 49],
        decode_query_len=16,
        max_num_reqs=1,
        max_cudagraph_capture_size=49,
    )
    assert 16 in sizes
    assert 17 in sizes
    assert 49 in sizes
    assert 32 not in sizes
