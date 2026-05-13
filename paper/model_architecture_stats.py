"""
Compute exact model architecture statistics for Methods section 2.3.

Instantiates the model from the published training config and reports
parameter counts per component, head dimension, FFN expansion ratio,
and on-disk size.

Output: paper/model_architecture_stats.json
Run from project root: python paper/model_architecture_stats.py
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import torch

from src.smart_rxn_embeddings.models.smarts_transformer import (
    SmartsMLMModel,
    TransformerConfig,
)

# ── Config (from models/smarts_transformer_20260428_160310.json) ───────────
CFG = TransformerConfig(
    vocab_size=4466,
    d_model=256,
    nhead=8,
    num_encoder_layers=6,
    dim_feedforward=1024,
    dropout=0.1,
    max_seq_len=512,
    pad_id=0,
)

OUTPUT_JSON = Path("paper/model_architecture_stats.json")


def n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ── Instantiate ────────────────────────────────────────────────────────────
model  = SmartsMLMModel(CFG)
enc    = model.encoder
layer0 = enc.encoder.layers[0]
head   = model.mlm_head

# ── Parameter counts ───────────────────────────────────────────────────────
p_tok      = n_params(enc.token_embedding)
p_pos      = n_params(enc.pos_embedding)
p_stack    = n_params(enc.encoder)
p_per_layer = p_stack // CFG.num_encoder_layers
p_sa       = n_params(layer0.self_attn)
p_ffn      = n_params(layer0.linear1) + n_params(layer0.linear2)
p_ln       = n_params(layer0.norm1) + n_params(layer0.norm2)
p_encoder  = n_params(enc)
p_mlm_head = n_params(head)
p_total    = n_params(model)

results = {
    "config": CFG.to_dict(),
    "derived": {
        "head_dim": CFG.d_model // CFG.nhead,
        "ffn_expansion_ratio": CFG.dim_feedforward // CFG.d_model,
        "ffn_activation_encoder_layers": "relu",
        "ffn_activation_mlm_head": "gelu",
        "normalization": "post-LN (norm_first=False)",
        "positional_encoding": "learned (nn.Embedding)",
        "architecture_type": "encoder-only (BERT-style)",
        "framework": "PyTorch nn.TransformerEncoder",
    },
    "parameters": {
        "token_embedding": p_tok,
        "positional_embedding": p_pos,
        "transformer_stack_total": p_stack,
        "per_encoder_layer": p_per_layer,
        "per_layer_self_attention": p_sa,
        "per_layer_ffn": p_ffn,
        "per_layer_layernorms": p_ln,
        "encoder_total": p_encoder,
        "mlm_head": p_mlm_head,
        "full_model_total": p_total,
    },
    "size_mb_float32": {
        "full_model": round(p_total * 4 / 1e6, 2),
        "encoder_only": round(p_encoder * 4 / 1e6, 2),
    },
}

# ── Print summary ──────────────────────────────────────────────────────────
c = results["config"]
d = results["derived"]
p = results["parameters"]
s = results["size_mb_float32"]

print("=== MODEL CONFIGURATION ===")
print(f"Architecture        : {d['architecture_type']}")
print(f"Encoder layers      : {c['num_encoder_layers']}")
print(f"d_model             : {c['d_model']}")
print(f"Attention heads     : {c['nhead']}  (head_dim = {d['head_dim']})")
print(f"FFN dim             : {c['dim_feedforward']}  ({d['ffn_expansion_ratio']}× expansion)")
print(f"FFN activation      : {d['ffn_activation_encoder_layers']} (encoder) / {d['ffn_activation_mlm_head']} (MLM head)")
print(f"Normalization       : {d['normalization']}")
print(f"Positional encoding : {d['positional_encoding']}")
print(f"Dropout             : {c['dropout']}")
print(f"Max seq length      : {c['max_seq_len']}")
print(f"Vocab size          : {c['vocab_size']:,}  (4,465 static + 1 [MASK])")

print("\n=== PARAMETER COUNTS ===")
print(f"Token embedding     : {p['token_embedding']:>10,}")
print(f"Positional embedding: {p['positional_embedding']:>10,}")
print(f"Transformer stack   : {p['transformer_stack_total']:>10,}  ({p['per_encoder_layer']:,} / layer)")
print(f"  Self-attention    : {p['per_layer_self_attention']:>10,}  / layer")
print(f"  FFN               : {p['per_layer_ffn']:>10,}  / layer")
print(f"  LayerNorms (×2)   : {p['per_layer_layernorms']:>10,}  / layer")
print(f"Encoder total       : {p['encoder_total']:>10,}  ({p['encoder_total']/1e6:.2f}M)")
print(f"MLM head            : {p['mlm_head']:>10,}")
print(f"Full model total    : {p['full_model_total']:>10,}  ({p['full_model_total']/1e6:.2f}M)")
print(f"\nModel size (FP32)   : encoder {s['encoder_only']} MB  /  full {s['full_model']} MB")

# ── Save ───────────────────────────────────────────────────────────────────
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUTPUT_JSON}")