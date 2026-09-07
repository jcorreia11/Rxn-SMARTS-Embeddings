# Quick Start

## Command line

```bash
smarts-embed "[C:1]-[O:2]>>[C:1]=[O:2]"                       # → JSON to stdout
smarts-embed --file smarts.txt --output embeddings.npy        # batch → numpy array
smarts-embed "[C:1]-[O:2]>>[C:1]=[O:2]" --size large           # small | medium | large
```

To use a specific local checkpoint instead of the auto-discovered/downloaded one:

```bash
smarts-embed "[C:1]-[O:2]>>[C:1]=[O:2]" \
    --weights models/smarts_transformer_20260715_111009.pt \
    --config  models/smarts_transformer_20260715_111009.json
```

Run `smarts-embed --help` for the full option list (pooling, batch size, output
format, device, etc.).

## Python API

```python
from rxn_smarts_embeddings.predict import predict, load_embedder

emb = predict("[C:1]-[O:2]>>[C:1]=[O:2]")          # (d_model,)
embs = predict(["[C:1]-[O:2]>>[C:1]=[O:2]", "c1ccccc1>>c1cccnc1"])  # (N, d_model)

embedder = load_embedder()                          # reuse across calls
embs = embedder.embed(smarts_list, batch_size=128)
```

Both auto-discover the latest local checkpoint in `models/`, falling back to a
Hugging Face Hub download when none is found. Pass `weights=`/`config=`/`vocab=`
(or `--weights`/`--config`/`--vocab`) to use a specific run instead.

Pooling defaults to `"mean"` (average of non-padding token states), validated
as strictly better than `"cls"` (BOS token) across every model size and
classifier head — see [`SmartsEmbedder`][rxn_smarts_embeddings.models.embed.SmartsEmbedder]
in the API reference.

See the [API Reference](api.md) for the full `predict()`/`load_embedder()`
signatures and the lower-level `SmartsEmbedder` class.