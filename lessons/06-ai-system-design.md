# Lesson 6: AI System Design

How to design an AI system from scratch, explained for product people. Based on a framework from [Aman Agarwal](https://www.linkedin.com/in/amanagarwal1/) -- the eight things most candidates miss in AI PM interviews, and why they matter whether you're interviewing or building.

*[Styled HTML version](html/lesson-6-ai-system-design.html) -- download and open in your browser for the interactive version.*

---

## Why This Matters Even If You're Not Interviewing

Lessons 1-5 taught you what AI is under the hood: how data flows, how models learn, how agents work. This lesson is about putting it all together. When someone says "let's add AI to the product," these are the eight decisions you'll actually make. In that order.

---

## Part 1: Start With Users, Not the System

When someone says "design an AI system for X," the instinct is to talk about models and architecture. Resist it.

Pick one user segment. Map their journey. Find the real pain point. Solutions come after.

| What most people say | What you should say |
|---------------------|---------------------|
| "I'd build an LLM-based chatbot" | "Let me start with the user. A power user on this telecom app calls support 3x/month about billing disputes. The pain isn't the call. It's that they can't self-serve a clear billing breakdown." |

This is product management. You already know how to do this. The AI part comes later. The user part comes first.

**PM analogy:** You'd never write a PRD starting with "we'll use PostgreSQL." You start with the user problem. Same thing here.

---

## Part 2: The Three Pillars -- Model, Data, Memory

Every AI system has three load-bearing components. Name them explicitly.

| Pillar | What it means | Questions to ask |
|--------|---------------|-----------------|
| **Model** | What does the thinking | What type? Why? What's it optimized for? |
| **Data** | What feeds the model | Where does it come from? How is it structured? How fresh does it need to be? |
| **Memory** | What persists across interactions | What does the system remember about this user? What's session-scoped vs. permanent? |

When you name all three out loud, you signal that you think architecturally. Most people only talk about the model. The data and memory are where most systems actually fail.

**Connection to Lesson 2:** The data pillar is the pipeline you learned about in Lesson 2 -- raw input to something usable. The difference is that here, the input isn't training data. It's customer records, chat logs, usage patterns. Same concept, different source.

---

## Part 3: LLM Isn't the Default

This is the thing that trips up the most people. "I'd use an LLM" is not a design decision. It's a reflex.

Two jobs in the same system can need completely different models:

| Job | Better model | Why |
|-----|-------------|-----|
| Predict which customers will churn | XGBoost (classical ML) | Structured tabular data, interpretable feature importance, sub-100ms inference, fraction of the cost |
| Talk to a customer about their bill | LLM | Needs natural language generation, context handling, open-ended input |

The decision framework:

| Signal | Favors classical ML | Favors LLM |
|--------|-------------------|------------|
| Data type | Structured, tabular | Unstructured text, images, multi-modal |
| Interpretability | Feature importance needed | Black box acceptable |
| Latency budget | <100ms | 1-30s acceptable |
| Cost sensitivity | High volume, low margin | High value per query |
| Adaptation need | Stable schema | New data sources, open-ended input |
| Output type | Classification, regression, ranking | Generation, conversation, search |

**Connection to Lesson 3:** You learned what a neural network is in Lesson 3 -- layers of attention and MLP processing tokens. An LLM is that architecture at massive scale. XGBoost is a completely different approach: it builds decision trees on structured features. The architecture matters because it determines what the model is good at.

**The bar:** "Here are the tradeoffs, here's my pick, here's why." Not "I'd use an LLM."

---

## Part 4: Orchestration Before Agents

This connects directly to Lesson 5. You learned that a multi-agent system is a DAG of loops connected by a shared workspace. Now you're designing one.

Before you design individual agents, design the router that decides which agent handles what.

```
User sends message
       |
   [ Router ]  ← classifies intent, checks confidence
       |
   +---+---+--------+
   |       |        |
Analyst  Voice Bot  Executor
   |       |        |
   +---+---+--------+
       |
   [ Router ]  ← merges response, checks quality
       |
Response to user
```

The router is the orchestration layer. It does four things:

1. **Classifies** the user's intent (billing question? plan change? open-ended?)
2. **Routes** to the right specialist agent
3. **Falls back** when confidence is low (below threshold → human handoff)
4. **Merges** the specialist's response back into the conversation

**PM analogy:** This is triage. An ER doesn't send you straight to a surgeon. There's a triage desk that assesses you and routes you to the right specialist. Without triage, you have a bunch of specialists standing around with no one directing traffic.

**Connection to Lesson 5:** The six knobs apply to every agent in this system. The router's termination condition is "response delivered." The analyst's tools include database access. The voice bot's isolation keeps it from touching account settings. Same framework, applied to a real system.

---

## Part 5: Memory Isn't One Thing

You need at least three kinds of memory, and they serve different purposes.

| Type | What it stores | Example | Persistence |
|------|---------------|---------|-------------|
| **Session** | Current conversation state | "User asked about billing, then changed to data usage" | Dies when conversation ends |
| **Episodic** | Past interactions with this specific user | "Called 3x in March about the same roaming charge" | Permanent, per-user |
| **Semantic** | Knowledge base, docs, policies | "Roaming charges apply outside home network after 2GB" | Permanent, shared |

For a churn-reduction system, episodic memory is the most important. A customer who called three times about the same unresolved billing issue is about to leave. If your AI doesn't know that, it'll give a generic response instead of saying "I see you've called about this roaming charge three times. Let me escalate this right now."

**Implementation:** Session memory is just the conversation context. Episodic memory lives in a vector database -- you embed past interactions and retrieve relevant ones by similarity. Semantic memory is RAG (retrieval-augmented generation) over your product docs.

**PM analogy:** A good barista knows three things. What you just ordered (session). That you always get oat milk and complained about it being out last Tuesday (episodic). That oat milk costs $0.50 extra and is in the back fridge (semantic).

---

## Part 6: Show Failure Modes

Candidates who only describe the happy path look junior. Systems that only handle the happy path break in production.

| Failure | How to detect it | What to do |
|---------|------------------|------------|
| Model down | Health check, timeout | Human handoff with conversation transcript |
| High latency (>30s) | p95 monitoring | Human handoff, async notification |
| User repeats the same question | Semantic similarity check on consecutive messages | Escalate immediately -- you've failed |
| Low confidence on intent | Score <0.7 on classification | Route to human with context summary |
| Hallucination | Grounding check against source documents | Flag, serve verified response instead |

The pattern: every failure mode has a detection mechanism and a mitigation. The default mitigation for anything serious is human handoff with context attached. The human gets the transcript so they don't ask the customer to repeat everything.

**PM analogy:** Every product has error states. 404 pages, failed payments, timeout screens. You design those before launch, not after. AI systems have the same need -- the error states are just different.

---

## Part 7: Plan for 10x Traffic

Your prototype works on 100 test calls. The telecom has 50 million subscribers. What breaks?

Three bottlenecks at scale:

| Bottleneck | Problem at scale | Solution |
|-----------|-----------------|----------|
| **Embedding search** | SQL can't do similarity search on millions of vectors fast enough | Vector database (Pinecone, Weaviate, pgvector) |
| **Model API rate limits** | External LLM APIs have throughput ceilings | Batch non-real-time work (churn scoring) nightly; self-host for latency-critical paths |
| **Cache misses** | Same questions get asked thousands of times, each hitting the model | Cache the top N most frequent query embeddings and responses |

**The key principle:** Load test the model APIs before launch, not after. Know your ceiling. If OpenAI rate-limits you at 10,000 requests/minute and you expect 50,000, that's a launch-blocking problem you can find in week one, not week twelve.

**PM analogy:** You wouldn't launch a product without knowing your database can handle the expected load. AI systems have a different bottleneck -- the model -- but the discipline is the same: test at expected scale before you ship.

---

## Part 8: Metrics Across Four Layers

If you only measure one thing, you'll miss three ways the system can fail.

| Layer | What it measures | Example metrics | Who cares |
|-------|-----------------|----------------|-----------|
| **Model** | Is the AI correct? | Recall, precision, hallucination rate | Engineering |
| **Latency** | Is the AI fast? | p95 response time, time to first token | Engineering + Product |
| **User** | Is the user happy? | CSAT, % resolved without escalation, repeat contact rate | Product |
| **Business** | Is this worth doing? | Retention lift, support cost per ticket, revenue impact | Leadership |

Each layer catches different problems:

- Model metrics are perfect but users hate it → the UX is broken, not the model
- Users love it but business metrics are flat → you're solving a real problem that doesn't move the needle
- Latency is great but model is hallucinating → you're confidently wrong, fast

**PM analogy:** You already measure products this way. You track uptime (system), page load (latency), NPS (user), and revenue (business). AI systems need the same stack. The model layer is just a new row in your metrics dashboard.

---

## Putting It All Together

Here's the full sequence. In an interview, you walk through these in order. In real product work, you revisit them as you learn more, but the first pass follows this structure.

| Step | What you do | What it shows |
|------|------------|---------------|
| 1. Users first | Pick a segment, map the journey, find the pain | Product thinking |
| 2. Three pillars | Name model, data, memory explicitly | Architectural thinking |
| 3. Model selection | Argue the tradeoffs, pick, justify | Technical fluency |
| 4. Orchestration | Design the router before the agents | Systems thinking |
| 5. Memory tiers | Session, episodic, semantic -- scoped correctly | Depth |
| 6. Failure modes | Name them, detect them, mitigate them | Production readiness |
| 7. Scale plan | 10x traffic, bottlenecks, load testing | Operational maturity |
| 8. Four-layer metrics | Model, latency, user, business | Business judgment |

---

## What You Now Know

| Lesson | What it covers |
|--------|---------------|
| **Lesson 1** | The three-file structure. program.md directs the researcher, train.py is the experiment, prepare.py is the lab equipment. |
| **Lesson 2** | Data pipeline. Text to tokens to batches to input/target pairs. The model always predicts the next token. |
| **Lesson 3** | Model architecture. Embeddings to 8 layers of attention + MLP to output head. 50M learnable parameters. |
| **Lesson 4** | Training loop. Forward, loss, backward, optimize, repeat for 5 minutes. Two optimizers, scheduled learning rates, time-based stopping. |
| **Lesson 5** | Agent teams. An agent is a loop. A multi-agent system is a DAG of loops sharing a workspace. Six knobs, four patterns, seven failure modes, one debugging trick. |
| **Lesson 6** | AI system design. Users first, three pillars, model selection, orchestration, memory tiers, failure modes, scale planning, four-layer metrics. Framework from Aman Agarwal. |

---

*Framework credit: [Aman Agarwal](https://www.linkedin.com/in/amanagarwal1/), who identified these eight gaps from running live AI PM mock interviews.*
