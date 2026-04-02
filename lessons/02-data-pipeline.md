# Lesson 2: The Data Pipeline

How text becomes numbers, gets fed to the model, and gets measured.

*[Styled HTML version](html/lesson-2-data-pipeline.html) -- download and open in a browser for diagrams and visualizations.*

---

## The Full Pipeline

Everything in ML follows this flow. The model never sees text. It only sees numbers.

```
Raw text (parquet files) -> Tokenizer (text -> integers) -> Dataloader (pack into batches) -> Model (predict next token) -> Loss / BPB (how wrong was it?)
```

`prepare.py` owns everything except "Model." It's the lab equipment: it prepares the data, feeds it to whatever model `train.py` builds, and measures the result.

---

## Part 1: The Constants -- What's Fixed

These are the rules of the experiment that nobody gets to change. They exist in `prepare.py` so that every experiment is measured on equal terms.

| Constant | Value | What it means |
|----------|-------|---------------|
| `MAX_SEQ_LEN` | 2,048 | How many tokens the model sees at once (its "context window") |
| `TIME_BUDGET` | 300 | Training time in seconds. 5 minutes, hard stop. |
| `EVAL_TOKENS` | ~21M | How many tokens to evaluate on (40 batches of 524,288) |
| `VOCAB_SIZE` | 8,192 | Total number of unique tokens the model can recognize |

> **Why this matters:** `VOCAB_SIZE = 8,192` is unusually small. GPT-4 uses ~100,000. A smaller vocabulary means each token represents less text, so you need more tokens to encode the same content. But the model has fewer things to learn. It's a deliberate tradeoff for a 5-minute experiment: simplify the problem so small models can make progress.

---

## Part 2: Tokenization -- Text to Numbers

The tokenizer converts text into a sequence of integer IDs. It uses **BPE** (Byte Pair Encoding): start with individual characters, then repeatedly merge the most common adjacent pairs into single tokens.

### Example: "Hello world! 42"

```
[BOS]  "Hell"  "o"  " world"  "!"  " "  "42"
 8188   2401   111    1844     33   32   5318
```

Notice a few things:
- **BOS token** (Beginning of Sequence) marks where a new document starts. ID 8188.
- **Spaces get attached** to the word that follows them (" world" is one token, not "world").
- **Common words** become single tokens. Rare words get split into pieces.
- **Numbers are capped at 2 digits** per token (the regex pattern `\p{N}{1,2}`). "123" would be split into "12" + "3".

> **Connection to prepare.py:** The tokenizer is trained once by `prepare.py` and saved to `~/.cache/autoresearch/tokenizer/`. The training process reads a billion characters of text and learns which byte pairs to merge. Lines 141-203.

---

## Part 3: The Dataloader -- Packing Tokens into Batches

The model doesn't process one document at a time. It processes a **batch** of fixed-size rows simultaneously. Each row is exactly `MAX_SEQ_LEN + 1 = 2,049` tokens long.

The dataloader's job: take a stream of variable-length documents and pack them tightly into fixed-size rows, with zero wasted space.

### How packing works

Imagine you have documents of different lengths and rows that hold 12 tokens each:

```
Row 0: [B][a][a][a][a][B][b][b][b][b][b][b]
Row 1: [B][c][c][c][c][c][c][c][c][B][d][d]

B = BOS token | a = Doc A (4 tokens) | b = Doc B (6 tokens) | c = Doc C (8 tokens) | d = Doc D (cropped to fit)
```

Key details:
- Every document starts with a **BOS token**, so the model always knows "a new document begins here."
- It uses **best-fit packing**: find the largest document that fits the remaining space. Like Tetris.
- If nothing fits, it **crops the shortest document** in the buffer to fill exactly. 100% utilization, no padding.
- The buffer holds ~1,000 documents to choose from, giving it good packing options.

> **Why packing matters:** The naive approach would be: one document per row, pad the rest with zeros. But padding tokens are wasted compute: the model processes them but learns nothing. With packing, every single token the model sees is real data. At 524K tokens per step, even 5% padding waste would mean ~26K wasted tokens per step.

---

## Part 4: Inputs and Targets -- The Shift

The model's entire job is: **given a sequence of tokens, predict the next one**. To train this, we take each packed row and create two versions: **inputs** (everything except the last token) and **targets** (everything except the first token).

```
Packed row:   [B]  [The]  [cat]  [sat]  [on]  [the]  [mat]
Input  (x):   [B]  [The]  [cat]  [sat]  [on]  [the]   -
Target (y):    -   [The]  [cat]  [sat]  [on]  [the]  [mat]
```

At each position, the model sees all tokens up to that point and tries to predict what comes next. After seeing "The cat sat," it should predict "on." After seeing "The cat sat on," it should predict "the."

This is why the row is `T + 1 = 2,049` tokens, but the model's sequence length is `T = 2,048`. You need one extra token so the last input position has a target to predict.

```python
# prepare.py lines 334-336: the shift
cpu_inputs.copy_(row_buffer[:, :-1])   # everything except last
cpu_targets.copy_(row_buffer[:, 1:])   # everything except first
gpu_buffer.copy_(cpu_buffer, non_blocking=True)
```

---

## Part 5: Evaluation -- Bits Per Byte (BPB)

This is the north star metric. You already know it means "how efficiently the model can predict text it hasn't seen." Here's what's actually happening in the code.

1. Feed validation data through the model. For each token position, the model outputs a probability distribution over all 8,192 possible next tokens.

2. Compute **cross-entropy loss** for each position: how surprised was the model by the actual next token? If the model assigned high probability to the correct token, the loss is low. If it was surprised, the loss is high. This is measured in **nats** (natural log units).

3. Here's the clever part: different tokens represent different amounts of text. The token " world" represents 6 bytes of UTF-8 text. The token "!" represents 1 byte. BPB normalizes by the number of **bytes**, not tokens. This makes the metric independent of vocabulary size.

4. Convert nats to bits (divide by ln(2)). The final number: **total bits of surprise / total bytes of text**.

```python
# prepare.py lines 344-365: the actual evaluation
for _ in range(steps):
    x, y, _ = next(val_loader)
    loss_flat = model(x, y, reduction='none').view(-1)
    y_flat = y.view(-1)
    nbytes = token_bytes[y_flat]   # how many UTF-8 bytes each target token represents
    mask = nbytes > 0              # skip special tokens (0 bytes)
    total_nats += (loss_flat * mask).sum()
    total_bytes += nbytes.sum()

return total_nats / (log(2) * total_bytes)  # nats -> bits, per byte
```

> **Why BPB instead of just loss?** If you change the vocabulary size, the raw loss number changes even if the model is equally good. A model with 100K tokens will have lower per-token loss than one with 8K tokens (fewer, more informative tokens = easier predictions). BPB normalizes by bytes of actual text, so it's comparable across any tokenizer or vocabulary size. It's the metric equivalent of "cost per byte of text understood."

---

## Part 6: Putting It Together

Here's the complete data flow with actual numbers from this repo:

| Stage | What happens | Shape / Size |
|-------|-------------|-------------|
| Parquet files | Raw text documents on disk | ~6,500 shards |
| Tokenize | BPE encodes text to integer sequences | Variable-length lists of ints (0-8191) |
| Pack into rows | Best-fit pack documents into fixed rows | [128 rows x 2,049 tokens] |
| Shift | Split into inputs (x) and targets (y) | x: [128 x 2,048]  y: [128 x 2,048] |
| Model forward | Predict next token at every position | logits: [128 x 2,048 x 8,192] |
| Loss / BPB | Compare predictions to actual targets | Single number (e.g., 0.9979) |

That logits shape is worth pausing on: `[128 x 2,048 x 8,192]`. That's 128 rows, each with 2,048 positions, each outputting a probability over 8,192 possible next tokens. That's **~2.1 billion numbers** the model produces per batch. And then all of that gets collapsed down to a single loss number that says "you were this wrong."

---

**Next up:** What's inside the "Model" box? That's `train.py`, and it's where the architecture lives: embeddings, attention, MLPs, and how they stack together. That's [Lesson 3](03-model-architecture.md).
