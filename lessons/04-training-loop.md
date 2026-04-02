# Lesson 4: The Training Loop

Where the model actually learns.

*[Styled HTML version](html/lesson-4-training-loop.html) -- download and open in a browser for diagrams and visualizations.*

---

## The Big Picture: What Happens When You Run train.py

You now know the two big pieces: how data gets prepared (Lesson 2) and how the model processes it (Lesson 3). This lesson connects them. The training loop is the engine that takes a model full of random numbers and, step by step, turns it into something that can predict text.

The entire loop fits on one screen of code. Here's the skeleton:

```python
while True:
    # 1. FORWARD: feed data through the model, get loss
    loss = model(x, y)

    # 2. BACKWARD: compute gradients (which direction to nudge each parameter)
    loss.backward()

    # 3. UPDATE: nudge all 50M parameters to reduce loss
    optimizer.step()

    # 4. RESET: clear gradients for the next step
    model.zero_grad()

    # 5. CHECK: are we out of time?
    if total_training_time >= TIME_BUDGET:
        break
```

That's it. Every AI model you've ever used was trained by some version of this loop. The details differ, but the structure is always: predict, measure how wrong you were, figure out which direction to adjust, adjust, repeat.

```
FORWARD (predict next token) -> LOSS (how wrong?) -> BACKWARD (blame assignment) -> UPDATE (nudge weights) -> REPEAT (until time's up)
```

---

## Part 1: Forward Pass + Loss -- "How Wrong Are We?"

The forward pass is everything from Lessons 2 and 3 happening in sequence: tokens go in, a prediction for each next token comes out, and cross-entropy loss measures the gap between the prediction and reality.

```python
# train.py line 548
with autocast_ctx:          # use bfloat16 for speed
    loss = model(x, y)       # x = input tokens, y = target tokens
train_loss = loss.detach()   # save the number for logging (don't track gradients on it)
```

What happens inside `model(x, y)`:

1. Tokens -> embeddings -> 8 transformer blocks -> logits (one score per vocab word, per position)
2. Cross-entropy compares those logits against the actual next tokens
3. Returns a single number: the loss

The loss starts high (around 10+, basically random guessing across 32,768 vocabulary tokens) and should drop below 2 by the end of training. Lower loss means the model's predictions are closer to reality.

> **autocast and bfloat16:** `autocast_ctx` tells PyTorch to use bfloat16 (16-bit) precision instead of float32 (32-bit) for most operations. Half the bits means roughly half the memory and double the speed on modern GPUs. The "bf" stands for "brain floating point," a format designed by Google specifically for ML: it keeps the same range as float32 but with less precision. Good enough for training, and the final loss calculation is done in full float32 for accuracy.

---

## Part 2: Gradient Accumulation -- Fitting a Big Batch in a Small GPU

The model wants to process **524,288 tokens** per training step (`TOTAL_BATCH_SIZE = 2**19`). But the GPU can only fit 128 sequences of 2,048 tokens at once, which is 262,144 tokens. That's half of what we need.

```python
# train.py lines 495-497
tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN   # 128 * 2048 = 262,144
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd  # 524,288 / 262,144 = 2
```

The solution: **gradient accumulation**. Instead of processing 524K tokens at once, process them in 2 chunks of 262K. Run the forward and backward pass on each chunk, and the gradients *add up*. Then do one optimizer step using the accumulated gradients.

```python
# train.py lines 546-552
for micro_step in range(grad_accum_steps):   # 2 micro-steps
    with autocast_ctx:
        loss = model(x, y)
    train_loss = loss.detach()
    loss = loss / grad_accum_steps   # scale so gradients average correctly
    loss.backward()                  # gradients ACCUMULATE (add to existing)
    x, y, epoch = next(train_loader) # prefetch next batch while GPU works
```

The `loss / grad_accum_steps` is important: if you run two backward passes and add the gradients, you need to divide each by 2 so the total is an average, not a sum. Otherwise you'd be taking steps twice as large as intended.

> **Why such a big batch?** Bigger batches give more stable gradient estimates. With 128 sequences, the gradient might point in a slightly wrong direction just because of the random sample. With 524K tokens, the noise averages out and each step is more reliable. The tradeoff: bigger batches use more compute per step, so you take fewer steps in the same time budget. 524K tokens is the sweet spot this codebase has settled on.

---

## Part 3: Backward Pass -- "Whose Fault Is It?"

`loss.backward()` is where the magic of learning happens. It's a single line of code that triggers **backpropagation**, the algorithm that figures out how much each of the 50 million parameters contributed to the error.

Here's the intuition. The loss is a single number that depends on a long chain of operations:

```
tokens -> embeddings -> attention -> MLP -> ... 8 layers ... -> logits -> loss
```

Backpropagation walks this chain **in reverse**. Starting from the loss, it asks at each step: "If I tweaked this parameter slightly, would the loss go up or down? By how much?" The answer is the **gradient** for that parameter.

- A large positive gradient means "increasing this parameter would increase the loss (make things worse)"
- A large negative gradient means "increasing this parameter would decrease the loss (make things better)"
- A near-zero gradient means "this parameter doesn't matter much for this batch"

After `loss.backward()`, every parameter in the model has a `.grad` attribute: a number (or tensor of numbers, matching the parameter's shape) that says which direction to nudge it.

> **Why "backward"?** The forward pass goes input -> output. Backpropagation goes output -> input, using the chain rule from calculus. You don't need to understand the math, but the key property is: it computes all 50 million gradients in roughly the same time as one forward pass. That's what makes training neural networks practical. Without backpropagation, you'd have to wiggle each parameter individually and measure the effect, which would take 50 million forward passes per step.

---

## Part 4: The Two Optimizers -- AdamW and Muon

Now we have gradients for every parameter. The optimizer's job: use those gradients to actually update the parameters. This model uses **two different optimizers** for different types of parameters.

### AdamW

The industry standard. Used for:
- **Token embeddings** (wte), lr: 0.6
- **Value embeddings**, lr: 0.6
- **Output head** (lm_head), lr: 0.004
- **Lambda scalars**, lr: 0.5

Tracks a running average of gradients AND a running average of squared gradients. Parameters that have been getting consistent gradients get bigger updates. Parameters with noisy gradients get smaller, more cautious updates.

### Muon

A newer optimizer. Used for:
- **All matrix parameters** inside the 8 transformer blocks (Q, K, V, projections, MLP weights), lr: 0.04

Uses "polar express orthogonalization" to find update directions that are maximally diverse, preventing different parameters from all trying to learn the same thing. Especially effective for the large matrix multiplications inside transformer blocks.

```python
# train.py line 421-426: the optimizer dispatches by kind
def step(self):
    for group in self.param_groups:
        if group['kind'] == 'adamw':
            self._step_adamw(group)
        elif group['kind'] == 'muon':
            self._step_muon(group)
```

Notice the wildly different learning rates. The output head (lm_head) gets lr: 0.004, while token embeddings get lr: 0.6, which is 150x larger. This isn't arbitrary. The output head transforms 768-dimensional vectors into 32,768 vocabulary scores. A small change in those weights has a huge effect on predictions. The embeddings are a lookup table: changing one token's embedding only affects sequences containing that token, so you can afford to be more aggressive.

> **Why two optimizers?** Different types of parameters have different geometry. Embedding tables are high-dimensional lookup tables where each row is mostly independent. Matrix parameters in attention and MLPs are densely interconnected: changing one entry affects every input that passes through it. Muon is designed specifically for that dense, interconnected case. Using the right optimizer for each type of parameter gets better results in the same number of steps.

---

## Part 5: Learning Rate Schedules -- How Fast to Learn, and When

The learning rate controls step size: how much to adjust parameters on each step. But the right step size changes over the course of training. This model uses a three-phase schedule:

```
|████████████████████████████▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒░░░░|
0%                          50%                   100%
     Full speed              Warmdown -> 0
```

```python
# train.py lines 518-525
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:           # Phase 1: warmup (0% here)
        return progress / WARMUP_RATIO
    elif progress < 1.0 - WARMDOWN_RATIO:  # Phase 2: full speed (0% to 50%)
        return 1.0
    else:                                 # Phase 3: warmdown (50% to 100%)
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
```

In the default config: `WARMUP_RATIO = 0.0` (no warmup), `WARMDOWN_RATIO = 0.5` (second half is cooldown), `FINAL_LR_FRAC = 0.0` (learning rate drops to zero).

So the model runs at full learning rate for the first half of training, then linearly decays to zero. By the end, it's making extremely tiny adjustments, fine-tuning what it's already learned rather than making big swings.

### Three more schedules

The learning rate isn't the only thing that changes. Muon's parameters also shift over training:

```python
# Muon momentum: ramps from 0.85 to 0.95 over first 300 steps
def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# Weight decay: starts at 0.2, decays to 0 by end of training
def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)
```

**Momentum** controls how much the optimizer remembers from previous steps. Low momentum (0.85) early on means "be responsive to new gradients." High momentum (0.95) later means "stay the course, don't overreact to noise."

**Weight decay** is a regularization technique: it gently pushes all parameters toward zero, preventing any single parameter from growing too large. It starts at 0.2 and fades to 0, so the model is regularized while it's learning big patterns but left alone when it's fine-tuning.

---

## Part 6: The Full Step -- Putting It All Together

Here's every step of the loop with line numbers, showing how the pieces connect:

```python
# train.py lines 543-604
while True:
    t0 = time.time()

    # === ACCUMULATE GRADIENTS ===
    for micro_step in range(grad_accum_steps):   # 2 micro-steps
        loss = model(x, y)                        # forward pass
        loss = loss / grad_accum_steps             # scale for averaging
        loss.backward()                            # backward pass (gradients accumulate)
        x, y, epoch = next(train_loader)           # prefetch next batch

    # === COMPUTE SCHEDULES ===
    progress = total_training_time / TIME_BUDGET   # 0.0 to 1.0
    lrm = get_lr_multiplier(progress)              # learning rate decay
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm    # apply schedule to all groups

    # === UPDATE PARAMETERS ===
    optimizer.step()                               # AdamW + Muon update all 50M params
    model.zero_grad(set_to_none=True)              # clear gradients for next step

    # === TIMING ===
    dt = time.time() - t0
    if step > 10:                                # skip first 10 steps (compilation)
        total_training_time += dt

    # === STOP CONDITION ===
    if step > 10 and total_training_time >= TIME_BUDGET:
        break
```

Every line serves a purpose. There's no ceremony, no configuration framework, no abstraction layers. This is the inner loop that trains the model, and it's under 30 lines.

---

## Part 7: The Time Budget -- A Fixed Constraint

Most training runs are defined by "train for N steps" or "train for N epochs." Autoresearch does something different: **train for exactly 5 minutes** (300 seconds, set in `prepare.py`).

```python
# Progress is based on wall-clock time, not steps
progress = min(total_training_time / TIME_BUDGET, 1.0)
```

This is what makes experiments comparable. A model that's twice as deep takes roughly twice as long per step, so it gets half as many steps in 5 minutes. But the learning rate schedule still covers the full warmdown curve, because everything is pegged to progress (0.0 to 1.0), not step count.

### The 10-step exclusion

The first 10 steps don't count toward the time budget:

```python
if step > 10:
    total_training_time += dt
```

Why? PyTorch's `torch.compile` compiles the model on the first few forward passes. These steps are dramatically slower (sometimes 10-30x) as the compiler generates optimized GPU code. Counting them would penalize models that are harder to compile but might train faster once compiled. By excluding the first 10 steps, the time budget measures actual training speed.

> **The 5-minute budget as a design choice:** This constraint forces the autoresearch agent to find configurations that train efficiently within a fixed compute budget. A model that's theoretically better but too slow to converge in 5 minutes will lose to a simpler model that finishes its learning. This mirrors real-world constraints: you almost always have a fixed compute budget and need to find the best model that fits within it.

---

## Part 8: Monitoring -- What Gets Logged Each Step

Each step prints a single line with everything you need to know:

```
step 00142 (47.3%) | loss: 2.481903 | lrm: 1.00 | dt: 627ms | tok/sec: 835,421 | mfu: 42.1% | epoch: 1 | remaining: 158s
```

What each field means:

- **step 00142**: which training step we're on
- **(47.3%)**: progress through the time budget
- **loss: 2.481903**: smoothed training loss (exponential moving average). Should be going down.
- **lrm: 1.00**: learning rate multiplier. 1.0 = full speed, drops during warmdown
- **dt: 627ms**: wall-clock time for this step (both micro-steps combined)
- **tok/sec: 835,421**: tokens processed per second. Higher = faster training
- **mfu: 42.1%**: Model FLOPs Utilization. What percentage of the GPU's theoretical peak is actually being used
- **epoch: 1**: how many times we've cycled through the full training dataset
- **remaining: 158s**: seconds left in the time budget

### Smoothed loss

The logged loss isn't the raw value from the current step. It's an exponential moving average (EMA):

```python
ema_beta = 0.9
smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * train_loss_f
# Debias: corrects for the fact that early values are biased toward 0
debiased = smooth_train_loss / (1 - 0.9**(step + 1))
```

This smooths out the noise. Raw loss bounces around because each batch is a random sample. The EMA shows the trend: "are we generally improving?"

> **MFU: the efficiency metric.** MFU tells you how well you're using the hardware. An H100 can theoretically do 989.5 trillion bf16 operations per second. If your training is doing 416 trillion per second, that's 42.1% MFU. The gap comes from memory access time, data loading, Python overhead, and operations that can't fully saturate the GPU. 40-50% is typical for well-optimized single-GPU training. Getting above 60% usually requires multi-GPU setups with careful overlap of compute and communication.

---

## Part 9: Safety Checks and Housekeeping

### Fast fail: catch explosions early

```python
# train.py lines 570-572
if math.isnan(train_loss_f) or train_loss_f > 100:
    print("FAIL")
    exit(1)
```

If the loss becomes NaN (not a number) or explodes above 100, something has gone fundamentally wrong, usually a learning rate that's way too high. No point continuing. The script exits immediately so autoresearch can try a different configuration.

### Garbage collection management

```python
# train.py lines 593-598
if step == 0:
    gc.collect()    # clean up setup allocations
    gc.freeze()     # mark everything as permanent
    gc.disable()    # stop automatic garbage collection
elif (step + 1) % 5000 == 0:
    gc.collect()    # occasional manual cleanup
```

Python's automatic garbage collector can pause execution for ~500 milliseconds, which is nearly a full training step. The code disables it after the first step and only runs it manually every 5,000 steps. This is a performance optimization specific to tight training loops where every millisecond matters.

---

## Part 10: After the Loop -- Final Assessment

When time runs out, the loop breaks and the model gets its final exam:

```python
# train.py lines 611-613
model.eval()                          # switch to assessment mode
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
```

`model.eval()` turns off any training-specific behavior (this model doesn't have dropout, but it's good practice). Then `evaluate_bpb` from `prepare.py` runs the model on the held-out validation set and computes bits per byte: the single number that determines whether this experiment was better than the last one.

The final summary prints everything autoresearch needs to decide:

```
---
val_bpb:          1.187432     # THE metric. Lower is better.
training_seconds: 300.1        # How long actual training took
total_seconds:    312.4        # Including setup and assessment
peak_vram_mb:     14230.1      # Max GPU memory used
mfu_percent:      43.21        # Average hardware utilization
total_tokens_M:   251.7        # Millions of tokens processed
num_steps:        480          # Total optimizer steps taken
num_params_M:     50.3         # Model size in millions
depth:            8            # Number of layers
```

The autoresearch agent reads this output, compares val_bpb to the best previous run, and decides what to try next. If this run improved, the change sticks. If not, it gets discarded. That's the outer loop: the AI researcher running experiments on the AI model.

---

## The Complete Timeline of a Training Run

- **t = 0s** -- Setup. Build model (50M params, all random), create optimizers, load data, compile with torch.compile.
- **Steps 0-10 (not timed)** -- Compilation warmup. First passes trigger PyTorch compilation. Slow, doesn't count toward budget.
- **0% - 50% of budget** -- Full-speed training. Learning rate at maximum. Loss drops rapidly from ~10 to ~3. Model learns basic language patterns: common words, simple grammar, frequent phrases.
- **50% - 100% of budget** -- Warmdown. Learning rate decays linearly to 0. Loss drops more slowly from ~3 toward ~2. Model refines: better grammar, more coherent predictions, nuanced word choice.
- **Time's up** -- Final exam. Model switches to assessment mode. Runs on validation data it has never trained on. Computes val_bpb. Prints summary.

---

## Recap: The Training Loop in One Paragraph

The training loop feeds batches of tokens through the model (forward pass), measures how wrong the predictions are (loss), traces that error backward through every layer to compute a gradient for each of the 50M parameters (backward pass), then uses two optimizers to nudge those parameters in the direction that reduces loss (AdamW for embeddings and scalars, Muon for matrix weights). The learning rate starts high and decays to zero over the second half of the 5-minute budget. Every step processes 524K tokens, logging loss, speed, and GPU utilization. When time runs out, the model is assessed on held-out data and the result, val_bpb, is the single number that determines if this experiment was a success.

---

## What You Now Know: The Full Stack

With Lessons 1-4, you can now trace every step from raw text to trained model:

1. **Lesson 1:** The three-file structure. program.md directs the AI researcher, train.py is the experiment, prepare.py is the lab equipment.
2. **Lesson 2:** Data pipeline. Text -> tokens -> batches -> input/target pairs. The model always predicts the next token.
3. **Lesson 3:** Model architecture. Embeddings -> 8 layers of attention + MLP -> output head. 50M learnable parameters.
4. **Lesson 4:** Training loop. Forward, loss, backward, optimize, repeat for 5 minutes. Two optimizers, scheduled learning rates, time-based stopping.

Next could go in several directions: the hyperparameter search space (what the AI researcher actually tweaks between experiments), how program.md orchestrates the research agent, or running an actual experiment.
