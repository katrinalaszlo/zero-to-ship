# Autoresearch Learning Map

Tracking what we've covered, where we are, and what's ahead.

*[Styled HTML version](html/learning-map.html) -- download and open in a browser for the card layout.*

---

## prepare.py -- The Lab Equipment ✓

| Concept | What it means |
|---------|---------------|
| **Training Data** | Text downloaded from the internet, stored as .parquet files. One shard set aside for validation (the honest test). |
| **Tokenizer** | Converts text to numbers and back. Uses BPE to find common patterns. Vocabulary size is a choice with trade-offs: big vocabulary = fewer tokens but bigger embedding table. Small vocabulary = more tokens but simpler model. |
| **Bytes vs Tokens** | Bytes are fixed (1 English letter = 1 byte). Tokens depend on the tokenizer. Same text can be different numbers of tokens but is always the same number of bytes. |
| **Bits Per Byte (val_bpb)** | The scoring metric. Measures how well the model predicts text, normalized by bytes so vocabulary size changes don't break comparisons. Lower is better. |
| **Fixed Constants** | MAX_SEQ_LEN = 2048 tokens context, TIME_BUDGET = 300 seconds, EVAL_TOKENS = ~20M tokens for testing. These never change. They make experiments comparable. |
| **Evaluation Function** | evaluate_bpb() is the judge. Feeds unseen text to the model, scores surprisal, converts to bits per byte. The ground truth metric no one can modify. |

---

## train.py -- The Experiment ✓

| Concept | What it means |
|---------|---------------|
| **Model Configuration (GPTConfig)** | The blueprint: sequence_len, vocab_size, n_layer (depth), n_head (attention heads), n_embd (width), window_pattern. These define the model's shape and capacity. |
| **Trade-offs** | Four constraints when changing the model: time (fixed at 5 min), money (fixed by GPU choice), memory (hard wall, crash if exceeded), quality (bigger isn't always better, overfitting is possible). |
| **Embeddings** | A lookup table that turns each token number into a list of 768 numbers. Random at first, then learned during training so similar words get similar numbers. The lm_head does the reverse at the end: 768 numbers back to a score for each token. |
| **Attention** | How the model decides which previous tokens matter for predicting the next one. Uses Query/Key/Value: each token asks "what am I looking for?" and scores every previous token on relevance. Multiple heads (6) look for different relationships in parallel. Window pattern (SSSL) limits how far back some layers look to save compute. Causal: can only look backward, never forward. |
| **Temperature** | Controls how the model picks from its scored predictions. Low temperature: always picks the highest-scored token (precise, repetitive). High temperature: willing to pick lower-scored tokens (creative, less accurate). Same trade-off as likelihood thresholds in data matching. |
| **MLP (Feed-Forward Layer)** | Three steps after attention: expand (768 to 3,072 numbers), filter (ReLU removes negatives, squaring amplifies what remains), compress (3,072 back to 768). Attention notices who matters. MLP processes what was noticed. |
| **The Block and Forward Pass** | A Block is attention then MLP with residual connections (output added to input, so nothing is lost, like annotations on a document). Stack 8 Blocks and you get the full model. Full flow: tokens -> embed -> Block 1-8 -> lm_head -> prediction. Each layer deepens understanding because every token has been refined by previous layers. |
| **The Optimizer (Muon + AdamW)** | Adjusts every number in the model to reduce loss. Gradients say which direction. Learning rate says how much. AdamW (cautious, steady) for embeddings. Muon (aggressive, fast) for big matrix weights. Different parts get different learning rates because some are more sensitive. Schedule: ramp up, full speed, then slow down as training ends. |
| **Hyperparameters** | Two categories: architecture knobs (DEPTH, ASPECT_RATIO, WINDOW_PATTERN) change what the model is. Optimization knobs (learning rates, batch size, warmup/warmdown, weight decay) change how it learns. The most common things the agent experiments with. |
| **The Training Loop** | Core loop: forward pass (predict and score), backward pass (compute gradients for every number), optimizer step (adjust weights). Gradient accumulation splits large batches into GPU-sized chunks. First 10 steps excluded from timer (compilation overhead). Fast-fail: aborts if loss explodes or becomes NaN. After 5 minutes: evaluate on unseen text, print val_bpb. |

---

## program.md -- The Research Director ✓

| Concept | What it means |
|---------|---------------|
| **Setup Protocol** | Agent onboarding: create a git branch per session, read all files, verify data, initialize results.tsv. The branch enables clean keep/discard via git reset. |
| **Scope and Constraints** | Agent can modify train.py (anything). Cannot modify prepare.py, install packages, or change evaluation. Prevents cheating: you can change the model but not the test. Simplicity criterion: don't keep messy improvements. |
| **The Experiment Loop** | Edit, commit, run, read results, log to TSV, keep or discard. Output redirected to file to protect agent context. Commit before running so every idea is in git history. The output is better code, not a better model (models are thrown away each run). |
| **Autonomy Rules and Measurement Integrity** | "NEVER STOP" overrides the agent's default instinct to check in. Tonal divergence signals importance. Fixed harness ensures internal consistency (valid keep/discard decisions) and external comparability (findings are transferable to the community). |

---

## Connections to Tanso

| Connection | How it maps |
|-----------|-------------|
| **Training step = Usage event** | Each training step processes ~524K tokens at a computable cost. This maps directly to Tanso's event ingestion: a unit of work with cost and usage attached. |
| **Time budget = Entitlement check** | The 5-minute hard stop is the same concept as Tanso returning "allowed: false" when a customer hits their usage limit. Both are gates that enforce resource constraints. |
| **Architecture choices = Variable cost to serve** | Different model configurations consume different compute for the same token count. This is why two customers on the same plan can have wildly different margins. It's the core problem Tanso solves. |
| **Autoresearch loop = Pricing optimizer** | Try, measure, keep or discard. Applied to pricing rules instead of neural network architectures, using Tanso's historical data as the test set. |
| **Optimizer schedule = Billing period lifecycle** | Warmup, steady state, warmdown mirrors subscription start (onboarding), active usage, and end-of-period reconciliation. Both follow a structured lifecycle within a fixed window. |
| **Fixed harness = Consistent metric definition** | Changing the evaluation mid-session breaks comparability, just like changing how a usage unit is defined mid-month breaks a customer's margin trend. Internal consistency matters within a session; external consistency matters for shared findings. |
| **Parallel attention heads = Multi-agent orchestration** | Multiple agents working in parallel with a coordinator that merges their outputs. Same architecture Kat built with multiple Claude terminals and an orchestrator. |
