# SmolVLA-M Research Notebook

> Living research log. The agenda at the top is the current source of truth. Update the agenda and result tables after every meaningful run.

**Base policy:** Hugging Face LeRobot / SmolVLA  

---

## Current research agenda

**Status: dataset and baseline integration.**

1. Pin the LeRobot revision and verify the SmolVLA forward path.
2. Download and inspect one MIKASA-Robo-VLA LeRobotDataset v3 task, initially `RememberColor3-VLA-v0`.
3. Build a real batch containing images, language, proprioception and an action chunk.
4. Run the memory-free SmolVLA baseline and record loss, tensor shapes, latency and GPU memory.
5. Run a short offline fine-tuning baseline on one memory task.
6. Add the first memory condition: a compact recurrent/global memory token.
7. Add an ELMUR-inspired bounded memory bank with read/write attention and LRU rewrite.
8. Compare memory placement: prefix/VLM, action expert, and layer-local action-expert memory.


---

## Research question

Can a compact, bounded temporal memory improve a small flow-matching VLA on partially observable manipulation tasks without making inference too slow or exceeding consumer GPU memory?

The central comparison is:

```
SmolVLA
SmolVLA + recurrent memory
SmolVLA + bounded external memory
SmolVLA + layer-local external memory
```

The comparison must keep the data split, action horizon, flow-matching objective and evaluation protocol fixed.

---

## Why start with MIKASA-Robo-VLA

MIKASA-Robo-VLA is a language-conditioned extension of the MIKASA memory benchmark. Its tasks show a relevant cue early, hide or distract it later, and require a later manipulation decision. The benchmark provides LeRobotDataset v3 and RLDS exports. It has 90 tasks, 22,500 trajectories and more than six million timesteps.

The first task is intentionally small:

```
RememberColor3-VLA-v0
```

It lets us validate the entire pipeline before downloading the complete collection.

The first offline dataset run is not yet a claim about benchmark success. It validates that:

- the data schema is understood;
- observations and actions are aligned;
- the model receives the intended information;
- the loss and gradient path are correct.


---

## Evaluation ladder

### Level 0 — code and shape checks

No scientific claim. Check imports, tensor shapes and gradients.

### Level 1 — pretrained inference

Load `lerobot/smolvla_base` and run one real observation through the policy. Record:

- image keys and shapes;
- state shape;
- action chunk shape;
- inference latency;
- device and peak memory.

### Level 2 — offline imitation baseline

Fine-tune the memory-free SmolVLA on one task. Use the official flow-matching action loss. Record training and validation curves.

### Level 3 — offline memory ablations

Train the same task with:

1. current observation only;
2. recent state/action history;
3. recurrent memory token;
4. bounded external memory;
5. ELMUR-inspired layer-local memory.

### Level 4 — online benchmark evaluation

Run checkpoints in MIKASA-Robo-VLA. Report:

- per-task success rate;
- mean success rate;
- success by memory type;
- success by horizon split;
- latency and GPU memory.

---

## Baselines

| ID | Policy | Temporal information | Trainable components | Purpose | Status |
|---|---|---|---|---|---|
| B0 | pretrained SmolVLA | current observation | none | verify checkpoint and preprocessing | pending real batch |
| B1 | fine-tuned SmolVLA | current observation | action expert + state projection | offline imitation baseline | pending |
| B2 | history-conditioned SmolVLA | last (K) state/action vectors | history encoder + projection + selected policy layers | test whether short history helps | planned |
| B3 | recurrent memory token | (M_{t-1}) carried across timesteps | memory initialization/update + selected policy layers | minimal recurrent-memory baseline | planned |
| B4 | external memory bank | (M) bounded slots | read/write modules + policy layers | test explicit bounded memory | planned |
| B5 | ELMUR-inspired layer-local memory | separate memory per selected transformer layer | mem2tok, tok2mem, LRU update | test layer-local external memory | planned |

---

# Memory designs

## M0 — no memory

At timestep (t):

[
c_t = [I_t, L, s_t]
]

where:

- (I_t) — current camera tokens;
- (L) — language tokens;
- (s_t) — current proprioceptive state.

SmolVLA predicts an action chunk:

[
hat a_t = pi_	heta(c_t)
]

This is the reference point. Any memory method must beat it on tasks that genuinely require history without regressing badly on reactive tasks.

**Origin:** original SmolVLA formulation.

---

## M1 — recent state/action history

Keep a short explicit window:

[
H_t =
[(s_{t-K+1},a_{t-K+1}),ldots,(s_t,a_t)]
]

Encode it:

[
m_t = g_phi(H_t)
]

and project it to one or more prefix tokens:

[
z_t = W_m m_t
]

The SmolVLA prefix becomes:

[
c_t=[I_t,L,s_t,z_t]
]

This is the simplest temporal baseline. It tests whether the model benefits from history before introducing a persistent memory mechanism.

**Origin:** standard temporal-conditioning idea; our implementation is a project baseline, not a direct reproduction of ELMUR.

**Expected weakness:** information older than (K) steps disappears and the encoder has no explicit write/forget policy.

---

## M2 — recurrent memory token

Maintain a compact state:

[
M_t = f_phi(M_{t-1},I_t,L,s_t)
]

At the next timestep:

[
c_t=[I_t,L,s_t,M_{t-1}]
]

After the transformer pass, the output at memory positions becomes (M_t).

The same action loss trains both the policy and memory:

[
mathcal L =
sum_t
mathcal L_{mathrm{flow}}(hat a_t,a_t)
]

For offline sequences, use truncated backpropagation through time. The memory may be detached between windows to control graph size.

**Origin:** conceptually related to recurrent-memory VLA work, including μVLA; adapted to SmolVLA's prefix/suffix architecture.

**Research value:** clean baseline for comparing one compact recurrent state against a structured external memory.

---

## M3 — bounded external memory bank

Use (M) slots:

[
mathbf M_t in mathbb R^{B	imes M	imes d}
]

Read memory with cross-attention:

[
h_{mathrm{read}}
=
mathrm{CrossAttn}(Q=h,K=mathbf M_t,V=mathbf M_t)
]

Write candidate slots from current tokens:

[
	ilde{mathbf M}_{t+1}
=
mathrm{CrossAttn}(Q=mathbf M_t,K=h,V=h)
]

Then update slots using a controlled rule:

[
mathbf M_{t+1}
=
lambda	ilde{mathbf M}_{t+1}
+
(1-lambda)mathbf M_t
]

or replace selected slots.

**Origin:** directly inspired by ELMUR's external memory track, but initially implemented as a global module rather than inserted into every transformer layer.

**Research value:** separates bounded storage and read/write policy from the details of a particular transformer layer.

---

## M4 — ELMUR-inspired layer-local memory

For selected SmolVLA layers (l), maintain:

[
mathbf M_t^{(l)}
in
mathbb R^{B	imes M	imes d_l}
]

Token track:

[
h_{mathrm{sa}}^{(l)}
=
mathrm{SelfAttn}(h^{(l)})
]

[
h_{mathrm{mem}}^{(l)}
=
mathrm{CrossAttn}
left(
Q=h_{mathrm{sa}}^{(l)},
K=mathbf M_t^{(l)},
V=mathbf M_t^{(l)}
ight)
]

Memory write:

[
	ilde{mathbf M}_{t+1}^{(l)}
=
mathrm{CrossAttn}
left(
Q=mathbf M_t^{(l)},
K=h^{(l)},
V=h^{(l)}
ight)
]

Then apply an LRU replacement or convex blend.

**Origin:** direct architectural continuation of ELMUR, adapted from its offline-RL transformer to SmolVLA's VLM/action-expert structure.

**Important implementation rule:** flow matching has several internal denoising calls for one physical observation. Memory must remain fixed during all denoising steps for (o_t), and update only once after (hat a_t) is produced:

[
o_t
ightarrow
	ext{all flow steps}
ightarrow
a_t
ightarrow
mathbf M_{t+1}
]

Otherwise one observation would incorrectly advance memory multiple times.

**Research question:** does layer-local memory help more than one global memory state when the action expert must retain phase and task information?

---

## M5 — adaptive write gate

Instead of updating memory every timestep:

[
g_t=sigma(W[h_t,M_t])
]

[
M_{t+1}
=
g_todot	ilde M_{t+1}
+
(1-g_t)odot M_t
]

The model learns when an observation contains information worth retaining.

**Origin:** our extension of external-memory update rules; conceptually related to adaptive write/forget mechanisms, but not a direct ELMUR component.

**Hypothesis:** adaptive writing should reduce memory pollution from irrelevant frames and distractors.

---

## M6 — FiLM conditioning of the action expert

Instead of adding memory tokens, use memory to modulate hidden states:

[
gamma_t,eta_t
=
g_phi(M_t)
]

[
h'_t
=
gamma_todot h_t+eta_t
]

where (h_t) is an action-expert hidden state.

**Origin:** standard FiLM/conditional-normalization idea; our adaptation to VLA memory.

**Expected benefit:** memory does not increase attention sequence length.

**Expected risk:** harder to inspect and implement; memory may influence actions weakly if modulation is poorly placed.

---

## What we will not do first

- Reinforcement learning for memory selection.
- Retrieval from a large episodic database.
- Full visual-frame memory.
- Memory in every VLM and action-expert layer simultaneously.
- Full SmolVLA fine-tuning.
- A large simulation setup before the offline data path works.

---

# SmolVLA training objective

For an action chunk (a), sample noise (epsilon) and time (t). SmolVLA constructs:

[
x_t=(1-t)a+tepsilon
]

and predicts a velocity field:

[
v_	heta(x_t,t,c_t)
]

The flow-matching loss is:

[
mathcal L_{mathrm{flow}}
=
left|
v_	heta(x_t,t,c_t)
-
(epsilon-a)
ight|_2^2
]

The memory mechanism changes (c_t), not the basic flow objective.

Initial trainable setup:

- vision encoder frozen;
- most VLM weights frozen;
- action expert trainable;
- state projection trainable;
- memory encoder/read/write modules trainable.

Later we can compare action-expert fine-tuning with LoRA.

---

# Dataset plan

## First dataset

```
mikasa-robo/mikasa-robo-vla-lerobot
```

First task:

```
RememberColor3-VLA-v0
```

The benchmark's canonical signals include:

- RGB observations from two cameras;
- 7D proprioception;
- 7D normalized end-effector delta actions;
- reward and success;
- language instruction.

The LeRobot v3 layout uses:

```
meta/
data/       Parquet shards
videos/     MP4 camera streams
```

The complete dataset is large, so download one task first.

## Dataset checks

The inspection script must report:

- dataset version;
- feature names and shapes;
- FPS;
- number of episodes;
- total frames;
- task instructions;
- camera keys;
- state/action ranges;
- one sample frame;
- one sample action chunk.

## Split policy

Do not split individual frames randomly. Split by episode/task according to the benchmark protocol. Adjacent frames from one episode must not leak across train and validation.

---

# Results tables

## Environment and reproducibility

| Date | Model commit | LeRobot commit | Dataset revision | Notes |
|---|---|---|---|---|

## Dataset inspection

| Dataset/task | Episodes | Frames | Cameras | State shape | Action shape | FPS | Result |
|---|---:|---:|---|---|---|---:|---|
| RememberColor3-VLA-v0 | — | — | — | — | — | — | pending |

## Offline baseline

| Run | Policy | Task | Steps | Batch | Train loss | Val loss | Latency | Peak VRAM | Checkpoint |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| B0-001 | pretrained SmolVLA | — | — | — | — | — | — | — | pending |
| B1-001 | fine-tuned SmolVLA | — | — | — | — | — | — | — | pending |

## Memory ablations

| Run | Memory | Slots/tokens | Window | Update rule | Trainable params | Val loss | Online SR | Notes |
|---|---|---:|---:|---|---:|---:|---:|---|
| M1-001 | recent history | — | — | pooling/MLP | — | — | — | pending |
| M2-001 | recurrent token | — | TBPTT — | recurrent | — | — | — | pending |
| M3-001 | global bank | — | — | LRU/blend | — | — | — | pending |
| M4-001 | layer-local bank | — | — | LRU/blend | — | — | — | pending |

---

# Experiment log

## Entry template

### Run ID

```
YYYYMMDD-model-memory-task-seed
```

### Hypothesis

What should improve and why?

### Configuration

- policy:
- dataset/task:
- data split:
- memory:
- memory size:
- action chunk:
- flow steps:
- batch:
- learning rate:
- frozen modules:
- trainable modules:
- random seed:

### Results

- train loss:
- validation loss:
- success rate:
- latency:
- peak GPU memory:

### Interpretation

What does the result support or contradict?

### Next action

What changes in the next run?

---

# Current decision

Do not implement ELMUR directly yet. First complete:

```
real MIKASA task
→ inspect schema
→ real SmolVLA batch
→ memory-free forward
→ short offline baseline
→ simplest recurrent/global memory
→ ELMUR-inspired memory bank
```

This ordering keeps each experiment interpretable and prevents us from attributing a pipeline/data bug to the memory architecture.
