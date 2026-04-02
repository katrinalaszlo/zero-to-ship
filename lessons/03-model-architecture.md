# Lesson 3: Inside the Model

What happens between "tokens go in" and "predictions come out" -- the architecture of `train.py`.

*[Styled HTML version](html/lesson-3-model-architecture.html) -- download and open in a browser for diagrams and visualizations.*

---

## The Config: Model Dimensions

Before anything else: the model's shape is defined by a handful of numbers. Everything else derives from these.

| Parameter | Value | What it means |
|-----------|-------|---------------|
| `DEPTH` | 8 | Number of transformer blocks stacked on top of each other |
| `ASPECT_RATIO` | 64 | Controls width: model_dim = depth x 64 = 512 |
| `n_embd` | 512 | The "width" of the model: every token is a vector of 512 numbers |
| `n_head` | 4 | Number of attention heads (parallel attention computations) |
| `HEAD_DIM` | 128 | Dimension per head. n_head = n_embd / HEAD_DIM |
| `vocab_size` | 8,192 | Number of possible tokens (from prepare.py) |
| `sequence_len` | 2,048 | Context window (from prepare.py) |

> **Depth vs width:** The model has two axes: **depth** (how many layers) and **width** (how big each layer is). The ASPECT_RATIO ties them together: wider models need more layers to be effective, so width scales with depth. This is one of the key knobs the AI researcher can turn in experiments.

---

## Part 1: The Tower -- Full Architecture

The model is a vertical stack. Data enters at the bottom and exits at the top:

```
Input Token IDs        [128 x 2,048] integers from the dataloader
        |
  Token Embedding (wte)    Look up each token ID -> vector of 512 numbers
        |
     RMS Norm          Normalize embeddings before entering blocks
        |
  +---------------------------+
  |  x 8 blocks:              |
  |  lambda_resid * x +       |
  |  lambda_x0 * x0           |  Mix current representation with original embedding
  |        |                   |
  |    Attention               |  Each token looks at previous tokens to gather context
  |        |                   |
  |      MLP                   |  Feed-forward: expand to 2048, activate, compress back to 512
  +---------------------------+
        |
     RMS Norm          Normalize before final projection
        |
     lm_head           Linear projection: 512 -> 8,192 (one score per vocabulary token)
        |
  Softcap + Loss       Cap extreme values, compute cross-entropy against targets
```

---

## Part 2: Token Embedding -- IDs to Vectors

The first step is the most important conceptual leap: **every token becomes a point in 512-dimensional space**.

The embedding layer (`wte`) is just a lookup table. It has 8,192 rows (one per vocabulary token) and 512 columns. When you feed in token ID 2401 ("Hell"), it returns row 2401: a vector of 512 numbers.

```python
# The embedding is literally a giant table
wte = nn.Embedding(8192, 512)  # shape: [8192, 512]

# Feed in token IDs, get vectors
x = wte(idx)  # [128, 2048] -> [128, 2048, 512]
```

At first, these 512 numbers are random. They don't mean anything. But during training, the model learns to arrange them so that tokens with similar meanings end up near each other in this 512-dimensional space. "cat" and "dog" drift closer together. "cat" and "invoice" drift apart.

> **Why 512 dimensions?** Think of it as the model's "vocabulary for describing tokens." Two dimensions could capture simple things (positive/negative, concrete/abstract). But language is complex: you need dimensions for grammar, topic, sentiment, formality, position in a sentence, and thousands of other features. 512 gives the model enough room to encode rich meaning. This is the "width" of the model, and making it bigger lets the model capture more nuance, at the cost of more computation.

---

## Part 3: Attention -- "What Should I Pay Attention To?"

This is the core mechanism that makes transformers work. At every position in the sequence, the model asks: **"Which previous tokens are relevant to predicting what comes next here?"**

### The three projections: Q, K, V

Each token's 512-dim vector gets transformed into three different roles:

- **Q**uery -- "What am I looking for?"
- **K**ey -- "What do I contain?"
- **V**alue -- "What do I offer?"

The process:

1. **Compute scores:** Each token's Query is compared against every previous token's Key. High score = "that token is relevant to me."
2. **Normalize:** Scores get turned into percentages (they sum to 1). This is the "attention pattern."
3. **Gather information:** Each token collects a weighted mix of previous tokens' Values, weighted by those attention scores.
4. **Project out:** The gathered information gets projected back to 512 dimensions via `c_proj`.

```
Q x K^T -> scores -> softmax x V -> Output
```

### Multi-head: Parallel attention

The model doesn't run attention once. It runs it **4 times in parallel** (n_head = 4), each with its own Q, K, V projections. Each head can learn to pay attention to different things: one head might focus on grammar, another on topic, another on recent context.

The 512-dim vector gets split into 4 heads of 128 dimensions each. Each head runs attention independently, then they get concatenated back to 512 and projected through `c_proj`.

```python
# train.py lines 79-81: splitting into heads
q = self.c_q(x).view(B, T, 4, 128)  # 4 heads, 128 dim each
k = self.c_k(x).view(B, T, 4, 128)
v = self.c_v(x).view(B, T, 4, 128)
```

### Causal masking: No peeking

Critical constraint: **a token can only attend to tokens that came before it**, never tokens after it. This is what makes it "causal" attention. If position 5 could see position 6, the model would be cheating: it would already know the answer it's supposed to predict.

This is enforced by Flash Attention 3 (`fa3`), which handles the masking internally via `causal=True`.

### Sliding window: A cost optimization

Full attention means every token looks at ALL previous tokens. With 2,048 tokens, that's expensive. The `window_pattern = "SSSL"` means:

- **S** (Short) layers: each token only looks back 1,024 positions
- **L** (Long) layers: full 2,048 context
- Pattern repeats: layers 0,1,2 are Short, layer 3 is Long, layers 4,5,6 are Short, layer 7 (last) is always Long

Most layers do "local" attention (nearby context), and periodically one layer does the expensive "global" attention (full context). The last layer always sees everything.

---

## Part 4: Rotary Embeddings -- "Where Am I?"

Attention compares tokens, but it doesn't inherently know **where** they are. Without position information, "the cat sat on the mat" and "mat the on sat cat the" would look identical to attention.

Rotary Position Embeddings (RoPE) solve this by **rotating** the Q and K vectors based on their position. Token at position 0 gets rotated 0 degrees. Token at position 100 gets rotated 100x the base frequency. When Q and K are compared, the rotation difference encodes the distance between them.

```python
# train.py lines 90: applied to Q and K before attention
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
```

The intuition: two tokens close together have similar rotations, so their dot product (the attention score) is naturally higher. Tokens far apart have very different rotations, making it harder to attend strongly. This gives the model a smooth, continuous sense of "how far apart are these two tokens?"

---

## Part 5: The MLP -- "What Does This Mean?"

After attention gathers context from other tokens, the MLP processes each token individually. If attention is "what should I pay attention to?", the MLP is "given what I've gathered, what should I understand?"

```
Input (512) -> c_fc expand (512 -> 2,048) -> ReLU squared (activation) -> c_proj compress (2,048 -> 512)
```

Three steps:

1. **Expand** from 512 to 2,048 dimensions (4x wider). This gives the model a bigger workspace to think in.
2. **Activate** with ReLU-squared: zero out negative values, then square the positives. This introduces non-linearity (without this, stacking layers would be pointless, because multiple linear transforms collapse into one).
3. **Compress** back to 512 dimensions, keeping only what matters.

```python
# train.py lines 102-108
def forward(self, x):
    x = self.c_fc(x)          # [B, T, 512] -> [B, T, 2048]
    x = F.relu(x).square()    # zero negatives, square positives
    x = self.c_proj(x)        # [B, T, 2048] -> [B, T, 512]
    return x
```

> **Why expand then compress?** Think of it as the model temporarily thinking in a higher-dimensional space to do complex pattern matching, then summarizing the result back into the standard size. The MLP is where the model stores "facts" and "rules" it has learned. Research shows that individual neurons in the expanded layer often correspond to recognizable concepts.

---

## Part 6: The Block -- Attention + MLP Together

Each of the 8 layers is a "Block" that combines attention and MLP, with an important pattern: **residual connections**.

```python
# train.py lines 118-121: one block
def forward(self, x, ve, cos_sin, window_size):
    x = x + self.attn(norm(x), ve, cos_sin, window_size)  # attend + add back
    x = x + self.mlp(norm(x))                              # process + add back
    return x
```

Notice the `x = x + ...` pattern. The output of attention gets **added to** the input, not replacing it. Same for MLP. This is a residual connection: the original signal always passes through, and each layer just adds a small correction.

Without residuals, information from early layers would degrade as it passes through 8 layers. With residuals, early information flows directly to the top. Each layer only needs to learn "what should I add or adjust?", not "reproduce everything and also add my bit."

### The lambda trick: mixing in the original

This model goes further. Before each block, it mixes the current representation with the original embedding:

```python
# train.py lines 276-277: inside the forward pass
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
```

`resid_lambdas` starts at 1.0 and `x0_lambdas` starts at 0.1. So at the beginning, each layer mostly uses its current state (1.0 x x) with a small contribution from the original embedding (0.1 x x0). These values are **learnable**: the model figures out the right mix during training.

---

## Part 7: Value Embeddings -- A Second Memory

This is a feature from a recent paper called ResFormer, and it's one of the more unusual parts of this architecture.

Normal attention: the Value (V) is computed from the current token's representation, which has been transformed by all previous layers. Value embeddings add a second source: a **direct lookup from the original token IDs**, bypassing all intermediate layers.

```python
# Alternating layers get value embeddings (0, 2, 4, 6 or 1, 3, 5, 7)
# train.py lines 84-87
if ve is not None:
    ve = ve.view(B, T, n_kv_head, head_dim)
    gate = 2 * torch.sigmoid(self.ve_gate(x[..., :32]))
    v = v + gate * ve  # mix in the direct embedding
```

The gate is input-dependent: the model looks at the first 32 channels of the current representation and decides **how much** of the value embedding to mix in. Initialized so the gate output is 1.0 (neutral), meaning the model starts by including them equally and learns to adjust.

> **Why does this help?** In deep networks, information from the original tokens gets increasingly "processed" as it moves through layers. Sometimes the later layers need to recall the raw identity of a token, but that signal has been diluted. Value embeddings provide a shortcut: direct access to "what token was actually here?" without going through the full transformation chain.

---

## Part 8: The Output Head -- Vectors to Predictions

After 8 blocks, each token position holds a 512-dim vector that encodes everything the model "thinks" about what comes next. The final step converts this back to a prediction over the vocabulary.

```python
# train.py lines 281-291
x = norm(x)                        # normalize one last time

softcap = 15
logits = self.lm_head(x)           # [B, T, 512] -> [B, T, 8192]
logits = logits.float()            # full precision for stability
logits = softcap * tanh(logits / softcap)  # cap extreme values

if targets is not None:
    loss = cross_entropy(logits, targets)  # how wrong were we?
```

`lm_head` is a linear projection from 512 to 8,192: for each position, it produces one score per token in the vocabulary. Higher score = model thinks that token is more likely to come next.

The **softcap** (tanh capping at +/-15) prevents any single prediction from being too extreme. Without it, the model might become overconfident in wrong answers, which destabilizes training.

Finally, **cross-entropy loss** compares the model's prediction distribution against the actual next token. This single number is what flows backward through the entire model to update all the weights.

---

## Part 9: Shape Tracker -- Following the Data

Every shape transformation from input to output:

| Stage | Shape | Notes |
|-------|-------|-------|
| Input token IDs | [128, 2048] | Integers 0-8191 |
| After embedding | [128, 2048, 512] | Each int -> 512-dim vector |
| After norm | [128, 2048, 512] | Same shape, normalized values |
| Q, K, V (in attention) | [128, 2048, 4, 128] | Split into 4 heads of 128 |
| After attention | [128, 2048, 512] | Heads concatenated, projected |
| MLP expanded | [128, 2048, 2048] | 4x wider for processing |
| MLP compressed | [128, 2048, 512] | Back to model width |
| ... repeat 8x ... | | |
| Logits (lm_head) | [128, 2048, 8192] | One score per vocab token |
| Loss | [1] | Single number: how wrong |

---

## Where the 50M Parameters Live

Every number the model learns is a "parameter." Here's where they are:

| Component | What it does | Params |
|-----------|-------------|--------|
| `wte` | Token embedding table | 4.2M |
| `value_embeds` | Value embedding tables (4 layers) | ~16.8M |
| `lm_head` | Output projection | 4.2M |
| Transformer blocks | Attention + MLP x 8 layers | ~25M |
| Scalars | Lambda weights (16 total) | 16 |
| **Total** | | **~50M** |

Notice that the embedding tables (wte + value_embeds + lm_head) account for roughly half the parameters. This is characteristic of small models with small vocabularies. As models get larger, the transformer blocks dominate.

---

**Next up:** [Lesson 4: The Training Loop](04-training-loop.md) -- How does the model actually **learn**? Forward pass, loss, backward pass, optimizer step, and the learning rate schedules that control it all.
