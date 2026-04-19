# Lesson 5: Managing Agent Teams

How multi-agent systems work, and why they fail.

*[Styled HTML version](html/lesson-5-managing-agent-teams.html) -- download and open in a browser for the interactive version.*

---

## Part 1: What Is an Agent?

An agent is a loop. That's it.

```
Look at the situation -> Think about what to do -> Do it -> Check if done -> If not, repeat
```

A function runs once and returns. An agent keeps going until it decides it's finished. When you give an AI tool a task and it reads files, edits code, runs tests, fixes errors, and tries again -- that's a loop. The agent decides how many times to go around.

A human engineer does the same thing. They look at the ticket, think about the approach, write code, run it, see if it works, adjust. The difference is the loop runs in seconds instead of hours.

**The key word: termination condition.** Every loop needs to know when to stop. "Tests pass" is a termination condition. "All files reviewed" is a termination condition. "I've been going for 50 iterations" is a termination condition. Without one, the agent keeps running forever. This is the equivalent of an engineer who never ships -- they keep polishing forever.

---

## Part 2: What Is a Multi-Agent System?

Multiple loops, connected by their outputs.

Think about how you'd run a feature project with three engineers:

1. **Alice** researches the API design and writes a spec
2. **Bob** reads Alice's spec and implements the backend
3. **Carol** reads Alice's spec and implements the frontend
4. Bob and Carol work at the same time (parallel) because they don't depend on each other
5. **You** review both their outputs at the end

That's a multi-agent system. Four agents (Alice, Bob, Carol, You), connected by data flow (spec flows to Bob and Carol, their code flows to you).

The technical term is a **DAG** -- a directed acyclic graph. Directed means the data flows one way (Alice to Bob, not Bob to Alice). Acyclic means there are no circles (nobody is waiting for someone who's waiting for them). It's just a project plan drawn as a diagram.

```
        Alice (research)
           |
     +-----+-----+
     |           |
  Bob (backend)  Carol (frontend)
     |           |
     +-----+-----+
           |
       You (review)
```

You already think in DAGs. Every project plan is one. Every Gantt chart is one. The only new thing is that the workers are agents instead of humans.

---

## Part 3: How Do Agents Share Work?

Two models. One works, one mostly doesn't.

### The meeting model (mostly doesn't work)

Agents sit in a chat room and talk to each other. Agent A says "I found this." Agent B says "That changes my approach." Agent C says "I disagree." A moderator decides who speaks next.

This sounds intuitive because it's how humans collaborate. But it breaks for agents because:

- Every message fills up every agent's context window with stuff that's irrelevant to their task
- Agents start validating each other's mistakes instead of catching them (the "hallucination loop")
- Nobody can debug what happened because the conversation is a mess
- It scales terribly -- adding more agents means more crosstalk

This is like putting your whole engineering team in one Slack channel and having them do all their work by chatting. It works for 3 people. It's chaos for 10.

### The workspace model (works)

Agents don't talk to each other at all. They read from and write to a shared workspace -- a filesystem, a git repo, a database. Agent A writes a file. Agent B reads that file. They never interact directly.

This is how your engineering team actually works. Alice pushes a spec to the repo. Bob pulls it and starts coding. Bob doesn't need to be in a meeting with Alice. He just needs the spec.

The shared workspace is called the **substrate**. In most agent systems today, the substrate is the filesystem or a git repo. Each agent reads what it needs, does its work, writes its output. The next agent reads that output.

**Why this works better:**
- Each agent only sees what's relevant to its task (no context rot from irrelevant chat)
- You can inspect the workspace at any point to see the state of the project
- Adding agents doesn't create more crosstalk
- It maps directly to how engineers actually work with code

---

## Part 4: Why Split Work Across Agents at All?

Only two reasons. If neither applies, keep it in one agent.

### Reason 1: Context protection

LLMs get worse when their context window fills up with irrelevant stuff. It's not a hard cliff -- it's a gradual degradation. The model starts "losing the thread." Important details from earlier get drowned out by newer, less relevant content.

Splitting into multiple agents means each one gets a clean, focused context. The research agent only sees research materials. The implementation agent only sees the spec and the codebase. Nobody is carrying around context they don't need.

**PM analogy:** You don't invite the whole company to every meeting. You invite the people who need to be there and give them only the context they need. Same principle.

### Reason 2: Parallelism

Independent tasks can run at the same time. Two agents researching two different topics finish in the time it takes one agent to do both.

But parallelism has a subtler benefit: it avoids **path dependence**. If one agent does all the research sequentially, the first result anchors its thinking. It finds one OAuth library and then everything else gets evaluated relative to that first find. Two agents researching independently won't anchor on each other's results.

**PM analogy:** You send two engineers to evaluate two different approaches independently. If you send one engineer to evaluate both sequentially, they'll unconsciously favor whichever they looked at first.

### The filter

Before splitting any task across agents, ask: "Am I buying context protection, parallelism, or neither?" If neither, you've added complexity for nothing.

---

## Part 5: The Six Knobs per Agent

Every agent has exactly six things you configure. When something goes wrong, it's almost always one of these six.

| Knob | What it means | PM equivalent |
|------|--------------|---------------|
| **Prompt** | The instructions and context the agent works with | The brief / ticket / spec you write for an engineer |
| **Tools** | What the agent can do (read files, search web, run code) | What systems the engineer has access to |
| **Termination** | How the agent knows it's done | Definition of done on the ticket |
| **Isolation** | What the agent can see and touch vs what's off-limits | File ownership / code ownership boundaries |
| **Verification** | How the agent's work gets checked | Code review / QA process |
| **Substrate** | Where the agent reads input from and writes output to | The repo / shared drive / project board |

**Debugging with this model:** When an agent produces bad output, check in order:

1. Was the prompt wrong? (bad brief)
2. Did it have the wrong tools? (missing access)
3. Did it stop too early or too late? (unclear definition of done)
4. Did it touch something it shouldn't have? (scope creep)
5. Did anyone check its work? (no review)
6. Did it write its output where the next agent can find it? (wrong handoff)

---

## Part 6: The Four Patterns

Almost every multi-agent system is one of these four, or a combination.

### 1. Pipeline

A then B then C. Each stage does one thing, passes output to the next.

Example: Research then Implement then Test. Simple, linear, easy to debug. Use when each step genuinely needs different context or tools.

### 2. Fan-out / Fan-in

One coordinator splits work to N workers, workers run in parallel, results get merged.

Example: "Research these 5 competitors" -- 5 agents each research one -- coordinator synthesizes findings. This is map-reduce. Use when you have N independent items to process.

### 3. Phased-parallel

Multiple waves of fan-out/fan-in. Each wave's output shapes the next wave.

Example:
- Wave 1: Classify 2000 notes into categories (parallel)
- Wave 2: Within each category, cross-reference and deduplicate (parallel within categories)
- Wave 3: Write coherent articles per section (parallel per section)
- Wave 4: Editorial review (may be sequential)

Between waves, the orchestrator compresses state and plans the next fan-out. This is the most powerful pattern and what serious production systems use.

### 4. Verifier-in-the-loop

After every write, an independent agent checks the work before the next step starts.

Example: Agent writes code, then Verifier runs tests and reviews. If pass, next step. If fail, back to writer. This is the equivalent of requiring code review before merge. It's cheap and catches most regressions.

---

## Part 7: The Failure Modes

Recognize these on sight. Most real-world failures are one of these.

| Failure | What it looks like | PM equivalent |
|---------|-------------------|---------------|
| **Context rot** | Agent has too much irrelevant context, reasoning degrades | Engineer drowning in Slack notifications, can't focus |
| **Path dependence** | Sequential work anchors on first result | Engineer falls in love with first approach, doesn't consider alternatives |
| **Orchestrator bloat** | Parent agent accumulates all children's outputs, goes incoherent | PM who attends every meeting and tries to hold everything in their head |
| **Handoff degradation** | Info lost when passing between agents | Game of telephone -- spec says one thing, implementation does another |
| **Premature decomposition** | Splitting before understanding the problem | Breaking a ticket into subtasks before understanding the actual work |
| **Over-decomposition** | So many agents that coordination cost exceeds actual work | 10 engineers on a 2-person task, spending all their time in standups |
| **Unbounded loops** | Agent never terminates | Engineer who keeps "almost done" forever |

---

## Part 8: The Sequential Reduction Trick

The single most useful debugging and design technique for multi-agent systems.

**Pretend parallelism doesn't exist.** Write out every agent as a sequential step: Step 1, then Step 2, then Step 3. Each step reads the previous step's output.

If the system produces correct output sequentially, you have a correct system. Then go back and ask: "Which of these steps have non-overlapping inputs? Those can run in parallel."

This converts "I have 7 agents and I don't know what's happening" into "I have 7 sequential steps, which ones are independent?" That's tractable.

**PM analogy:** Before assigning work in parallel, write the tasks as a single ordered list. If the list produces the right outcome, then look for tasks that don't depend on each other and assign those simultaneously. You'd never start parallelizing work before you understand the dependencies.

---

## Recap: Multi-Agent Systems in One Paragraph

An agent is a loop that keeps going until it hits a termination condition. A multi-agent system is a DAG of those loops, connected by a shared workspace (not by chatting). You split work across agents for exactly two reasons: context protection and parallelism. Each agent has six knobs (prompt, tools, termination, isolation, verification, substrate). The four patterns are pipeline, fan-out, phased-parallel, and verifier-in-the-loop. When things break, check for context rot, path dependence, orchestrator bloat, handoff degradation, or premature decomposition. When you can't figure out what's happening, use the sequential reduction trick: flatten everything to sequential steps, verify correctness, then re-introduce parallelism only where inputs don't overlap.

---

## What You Now Know

| Lesson | What it covers |
|--------|---------------|
| **Lesson 1** | The three-file structure. program.md directs the researcher, train.py is the experiment, prepare.py is the lab equipment. |
| **Lesson 2** | Data pipeline. Text to tokens to batches to input/target pairs. The model always predicts the next token. |
| **Lesson 3** | Model architecture. Embeddings to 8 layers of attention + MLP to output head. 50M learnable parameters. |
| **Lesson 4** | Training loop. Forward, loss, backward, optimize, repeat for 5 minutes. Two optimizers, scheduled learning rates, time-based stopping. |
| **Lesson 5** | Agent teams. An agent is a loop. A multi-agent system is a DAG of loops sharing a workspace. Six knobs, four patterns, seven failure modes, one debugging trick. |
