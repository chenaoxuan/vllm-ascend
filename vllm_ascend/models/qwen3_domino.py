import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM


def _dflash_config_dict(config) -> dict:
    raw = getattr(config, "dflash_config", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    return vars(raw)


class AscendDominoQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """DFlash Qwen3 draft plus Domino correction heads.

    Matches ``Huang2020/Qwen3-8B-Domino-b16``: ``prefix_gru`` and
    ``embed_proj`` sit on the inner ``model`` so DFlash ``load_weights``
    maps HF keys (``embed_proj.*``, ``prefix_gru.*``) to ``model.*``.
    Vanilla DFlash checkpoints (no ``projector_type=domino``) keep the
    parent class unchanged.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        dflash_cfg = _dflash_config_dict(self.config)
        self.projector_type = dflash_cfg.get("projector_type")
        self.pure_draft_prefix_len = int(dflash_cfg.get("pure_draft_prefix_len", 0))
        if self.projector_type != "domino":
            return
        hidden = int(self.config.hidden_size)
        gru_hidden = int(dflash_cfg.get("gru_hidden_dim", 1024))
        emb_dim = int(dflash_cfg.get("emb_dim", getattr(self.config, "emb_dim", 256)))
        vocab = int(self.config.vocab_size)
        self.model.prefix_gru = nn.GRU(
            input_size=hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.model.embed_proj = nn.Sequential(
            nn.Linear(hidden + gru_hidden, emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(emb_dim, vocab, bias=False),
        )
        # Same objects the HF checkpoint names ``embed_proj`` / ``prefix_gru``.
        # DFlash ``load_weights`` prefixes them onto ``self.model``.
        self.embed_proj = self.model.embed_proj
        self.prefix_gru = self.model.prefix_gru
