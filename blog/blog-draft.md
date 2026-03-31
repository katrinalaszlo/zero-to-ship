# I Build Billing for AI Companies. So I Decided to Learn How AI Training Works.

*What I found by reading 600 lines of code in Karpathy's autoresearch repo, and how it changed the way I think about my own product.*

---

Three years ago, writing frontend code felt out of reach for me. I'm a product manager by background. I think in specs, user stories, and metrics. Code was something other people wrote.

Then AI coding tools arrived, and I found I could build functioning applications. That shift changed my career. I'm now building Tanso, a billing and observability platform for AI companies.

But something kept bothering me. I build infrastructure for companies that run AI models, and I didn't really understand what those models *are*. I knew the vocabulary (tokens, parameters, training runs) but not the intuition. When a customer mentioned that their costs changed after switching model architectures, I understood the business impact but not the technical reason.

I wanted to close that gap. I started by reading a 600-line Python file.

## The Repo

Andrej Karpathy released a project called **autoresearch**. The idea: give an AI agent a small language model training setup, let it experiment autonomously overnight, and review the results in the morning.

The entire project is three files:

- **`prepare.py`**: Downloads training data and builds a tokenizer. Fixed, not modified.
- **`train.py`**: The neural network, optimizer, and training loop. The AI agent modifies this.
- **`program.md`**: Instructions for the AI agent. The human writes this.

I didn't need to run it to learn from it. Reading it was enough. Here's what I took away.

## program.md Is a PRD

The first thing I noticed wasn't the neural network code. It was `program.md`.

It contains:
- A clear goal: "Get the lowest val_bpb" (a score measuring how well the model predicts text)
- Scope constraints: "You can only modify train.py. Do not install packages."
- A decision framework: "Simpler is better, all else equal."
- A process loop: try something, measure, keep or discard, repeat
- Autonomy rules: "NEVER STOP. The human might be asleep."

Goal. Scope. Decision criteria. Process. Escalation policy. That's a product requirements document. The same structure I write regularly, except the audience is an AI agent instead of an engineering team.

**The skill of writing clear specs with goals, constraints, and success metrics is the same skill needed to direct AI agents.** Product managers have been building this muscle for years. It transfers directly.

## The Training Loop Is a Usage Event

Here's what happens in `train.py`, in plain terms:

1. Build a neural network (a mathematical structure that learns patterns in text)
2. Feed it text and ask it to predict the next word
3. Score how wrong it was
4. Adjust the network to be less wrong
5. Repeat for 5 minutes
6. Measure the final score on text it hasn't seen before

Each cycle processes about 524,000 tokens. On an NVIDIA H100, each cycle takes roughly half a second. Over 5 minutes, that adds up to about 500 million tokens.

This is where my product thinking kicked in. Each of those cycles is, in effect, a usage event. At Tanso, we help AI companies track the cost of exactly this kind of computation. Every API call to OpenAI, every inference request, every training step is a unit of work with a cost attached.

About $0.25 buys 5 minutes on an H100. That's fine for one experiment. But 100 experiments overnight is $25, and across 1,000 customers, you need visibility into which ones are profitable. That's the problem Tanso solves.

## Architecture Choices Are the Cost Problem

This was the most useful insight for me.

In `train.py`, there's a section of "hyperparameters," the knobs that control the model:

```
DEPTH = 8             how many layers deep the network is
TOTAL_BATCH_SIZE      how much text to process at once
ASPECT_RATIO = 64     how wide each layer is
```

Change `DEPTH` from 8 to 16 and the model gets twice as deep. It might learn better, but each step takes longer and uses more memory. You might need a more expensive GPU.

The implication: **two customers using the same feature can have very different costs to serve.** Customer A runs a small model. Customer B runs a large one. Same number of tokens. Same pricing plan. But one costs $0.10 to serve and the other costs $0.90.

I hear this from Tanso customers regularly. "We charge a flat rate per token, but some customers cost much more to serve than others." Now I understand the mechanics behind it. Different model architectures consume different amounts of compute for the same unit of work. The token is not the cost. The computation behind it is.

## The Pattern Transfers

The autoresearch loop is straightforward:

1. Change something
2. Measure the result against a fixed metric
3. If it improved, keep it. If not, discard.
4. Repeat.

This isn't specific to machine learning. It's a general optimization pattern that works anywhere you have a clear metric, something to modify, and an agent that can run the loop.

For Tanso, I realized we could apply this to help customers optimize their pricing. We already have the data: usage events with both revenue and cost attached. An agent could simulate hundreds of pricing configurations against real historical data and surface the ones that improve margin without excessive churn.

Flat rate at $0.01 per token gives a 23% margin. Model-based graduated pricing gives 41%. The agent finds this by running the same try-measure-keep loop, just with pricing rules instead of neural network architectures.

This is possible because Tanso sees both sides of the equation. Stripe knows what you collected. Your AI provider knows what they charged you. Tanso connects the two.

## The Fixed Harness Problem

There's a design detail in autoresearch that took me a minute to fully appreciate. The evaluation function (the test the model takes after training) is locked. The agent cannot modify it. If it could, it might improve its score by making the test easier rather than making the model better.

This constraint matters when you translate the pattern to pricing. If a customer could modify how the simulation works, the recommendations would be unreliable. The historical usage data is immutable (it already happened). The simulation logic is standardized (Tanso controls it). The customer only controls their business constraints: minimum margin, maximum churn tolerance, competitive boundaries.

That separation, between what's fixed and what's variable, is what makes the optimization trustworthy. It's the same principle in any experiment: hold the test constant so you know the improvement is real.

## You Probably Already Understand More Than You Think

One thing that surprised me during this process: I already understood many of these concepts. I just knew them by different names.

I spent years at a data company where we scored search results by likelihood. Set a high threshold and you get fewer, more precise results. Set it low and you get more results with more noise. That's the same trade-off as "temperature" in language models, a setting that controls whether the model picks the most likely next word or explores less certain options.

I've also built multi-agent setups where several AI terminals work in parallel on the same codebase, with a coordinator that merges their work and prevents conflicts. That's structurally the same as "attention heads" in a neural network: parallel processes that each focus on different aspects of the input, combined by a merge layer.

The concepts aren't foreign. They just have unfamiliar names. If you've worked in data, search, product management, or any field where you make decisions under uncertainty, you have more foundation than you think.

## What I'd Share with Other PMs

You don't need to become an ML engineer. I haven't. But if you're building products in the AI space, or managing products that use AI features, understanding the fundamentals is worth the time. It gives you:

**Customer empathy.** When a customer says costs changed, you understand the mechanics, not just the symptom.

**Pattern recognition.** The autonomous experiment loop is a well-structured process with clear inputs and outputs. Writing those processes is something product managers already do well.

**Product instinct.** The most useful product idea I've had recently came from reading someone else's training script and asking "what if we applied this pattern to pricing?" You can't connect ideas you haven't been exposed to.

The barrier to understanding AI isn't math or engineering skill. It's the assumption that it's not meant for you. Four years ago I felt that way about writing code. I was wrong then. I would have been wrong now.

Andrej Karpathy called this moment "vibe coding," the idea that you can just talk to an AI and forget the code even exists. But here's the irony: the people who are best at "vibe coding" aren't vibing at all. They're the ones who've been running structured processes their whole career. Writing specs, defining scope, orchestrating teams, unblocking work. Strategy, orchestration, and design. That's not vibes. That's product management.

The code is only 600 lines. It's worth reading.

---

*Kat Laszlo is the founder of Tanso, a billing and observability platform for AI companies.*
