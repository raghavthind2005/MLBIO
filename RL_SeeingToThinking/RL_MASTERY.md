# RL Mastery — GRPO for VLMs, from absolute scratch

**What this document is.** A from-the-ground-up explanation of how reinforcement learning trains our
vision-language model, written so that *every* technical term is defined before it is used, with
worked numeric examples, and with every claim tied to the **actual code** that runs on the cluster.

**The code it is grounded in.** Our pinned EasyR1 tree at commit `dd71bbd`, located locally at
`RL_SeeingToThinking/EasyR1/`. This is the *exact* code running on Clariden — when this doc says
`core_algos.py:213`, you can open that line and see what is described.

**How to read it.** Top to bottom the first time — each chapter depends on the previous. Later you
can jump to a chapter as reference. Conventions: ✓ = I have read this in the code and cite the line;
🧮 = a worked numeric example; ⚠️ = a subtle point that is easy to get wrong (interview-relevant);
🌐 = a **general principle** that transfers to any RL / LLM-RL setting; 📦 = a detail **specific to
this project / EasyR1** (a choice, swappable — not a law). **Every chapter ends with a 🌐 Generalizes /
📦 Ours box** that separates the transferable idea from our particular instantiation, so you can always
tell universal RL from this repo's conventions. (This dual layering is deliberate: master the general
method *and* know exactly how our code implements it.)

**Roadmap (nothing is skipped — this is the order):**

- **Part I — The learning signal** (Ch. 0–5): what we're doing, what a language model is, what a
reward is, what "token-level" means, and how GRPO turns rewards into a per-token *advantage*.
- **Part II — The weight update** (Ch. 6–8): how that advantage becomes an actual change to the
model's weights — the importance ratio, PPO clipping, KL regularization, and the full loop.
- **Part III — The systems** (Ch. 9+): FSDP, vLLM, the batch hierarchy, and our freeze patch.

This file currently contains **Part I in full**. Part II/III are written at the same depth next.

---

---

# PART I — THE LEARNING SIGNAL

# Chapter 0 — What are we even doing, and why?

We have a vision-language model (VLM): **Qwen3-VL-4B-Instruct**. It takes an image + a text question
and produces a text answer, one word-piece at a time. We want it to answer **perception** questions
more reliably.

There are two broad ways to improve a model like this after it has been pre-trained:

**1. Supervised fine-tuning (SFT).** You have a dataset of (question → *ideal answer*) pairs, and you
train the model to imitate the ideal answer. This requires you to *know and write down* the correct
output, including the correct *reasoning*. It teaches by **imitation**.

**2. Reinforcement learning (RL).** You do **not** give the model a target answer to copy. Instead you
give it a **scorer**: a function that looks at whatever the model produced and returns a number saying
how good it was. The model then **tries things**, sees which attempts scored well, and is nudged to
do more of what scored well and less of what scored poorly. It teaches by **trial, scoring, and
reinforcement**.

Our setting is a special, clean case of RL called **RLVR — Reinforcement Learning from Verifiable
Rewards**. "Verifiable" means the scorer is a *rule*, not a guess: for a perception multiple-choice
question we can check "did the model's final boxed answer equal the correct option? yes → 1.0, no →
0.0." There is no second neural network judging quality (that would be RLHF, used for fuzzy goals like
"helpfulness"); the reward is computed by a short Python function we can read (`math.py`, Chapter 3).

**Why RL instead of SFT here?** Two reasons that matter for our research:

- We often don't *have* gold reasoning traces — we only know the right final answer. RL only needs a
checker, not a worked solution.
- The paper's central claim is that RL *improves perception* in a way SFT on captions does not. The
whole project is about *how* that happens. So we must understand the RL machinery deeply.

The specific RL algorithm we use is **GRPO** (Group Relative Policy Optimization). The rest of Part I
builds up everything you need to understand exactly what GRPO computes and why.

> **🌐 Generalizes:** the SFT vs RL vs RLHF vs RLVR taxonomy; the core idea "RL = generate → score →
> reinforce"; that RL needs only a *scorer*, never target outputs to copy.
> **📦 Ours:** the *choice* of RLVR (a rule-based checker) applied to *perception*; the specific thesis
> that RL repairs perception where caption-SFT does not.

---

# Chapter 1 — A language model is a probability machine

Before any RL, you must understand precisely what the model *is*, mechanically, because RL operates on
these mechanics.

## 1.1 Tokens and the vocabulary

A language model does not read characters or words. Text is first chopped into **tokens** — small
chunks, each of which is an entry in a fixed dictionary called the **vocabulary**. A token might be a
whole word (`" cat"`), a word-piece (`"ing"`), a digit, or punctuation. Qwen3's vocabulary has ~150,000
entries. Each token has an integer **id** (its index in the vocabulary).

So the string `"The cat"` becomes a list of integers like `[785, 9059]`. This is what "token id"
means. In the code you will see tensors of these ids called `input_ids`, `responses`, etc.

🧮 Toy example we'll reuse: pretend the vocabulary has only **5 tokens**:
`["a"=0, "b"=1, "c"=2, "<end>"=3, "d"=4]`. A sequence is just a list of these ids, e.g. `[0, 2, 3]`
means `"a c <end>"`.

## 1.2 From context to "logits"

The model's job: given the tokens **so far** (the "context"), predict the **next** token. Internally,
after all its transformer layers, the model outputs one real number for every token in the
vocabulary. That vector of raw numbers is called the **logits**.

🧮 With our 5-token vocab, after seeing `"a"` the model might output logits:
`z = [2.0, 1.0, 0.1, -1.0, 0.5]` — one number per possible next token (`a, b, c, <end>, d`). A bigger
number means "the model finds this next token more plausible." But logits are unbounded raw scores,
not probabilities yet.

In the code, logits come straight out of the model's forward pass:
✓ `dp_actor.py:150` — `logits: torch.Tensor = output.logits`. Its shape is
`(batch, sequence_length, vocab_size)` — for every position in every sequence, a full vocab-sized
vector of scores.

## 1.3 Softmax: turning logits into a probability distribution

To turn logits into probabilities (non-negative, summing to 1), we apply the **softmax** function:

```
p_i = exp(z_i) / Σ_j exp(z_j)
```

Exponentiate every logit (makes them all positive), then divide by their sum (makes them sum to 1).

🧮 For `z = [2.0, 1.0, 0.1, -1.0, 0.5]`:

- `exp(z) = [7.39, 2.72, 1.11, 0.37, 1.65]`, sum `= 13.24`
- `p = [0.558, 0.205, 0.084, 0.028, 0.124]`

So the model assigns 55.8% probability to `"a"` being next, 20.5% to `"b"`, etc. This vector `p` is a
**probability distribution over the next token** — the model's belief about what comes next.

## 1.4 Generating text: autoregressive sampling

To produce an answer, the model does this in a loop:

1. Feed the context, get logits for the next position, softmax to get `p`.
2. **Sample** a token from `p` (or take the most likely — see temperature below).
3. Append that token to the context.
4. Repeat until a special `<end>`-of-sequence token is produced (or a length limit is hit).

"**Autoregressive**" = each new token is fed back in to predict the next; the model conditions on its
own previous outputs. One full run of this loop = one complete **response**.

⚠️ **Temperature.** Before softmax we can divide the logits by a number `T` called the temperature:
`p_i = softmax(z / T)`. `T = 1.0` leaves the distribution as-is. `T < 1` makes it *sharper* (more
greedy, picks the top token more often). `T > 1` makes it *flatter* (more random/exploratory). Our
training rollouts use `T = 1.0` (config `worker.rollout.temperature: 1.0`) — we want diverse samples
so the model explores. Validation uses `T = 0.6` (config `val_override_config`) — we want the model's
"best guess," less randomness.

## 1.5 Probability of a whole sequence

The probability the model assigns to producing an entire response is the product of the
per-token probabilities, each conditioned on everything before it:

```
P(response) = P(t1 | prompt) · P(t2 | prompt, t1) · P(t3 | prompt, t1, t2) · ...
```

🧮 If the response `"a c <end>"` had per-token probabilities `0.558`, then `0.30`, then `0.80`, the
sequence probability is `0.558 × 0.30 × 0.80 = 0.134`.

## 1.6 Log-probabilities — and why we *always* use them

Multiplying many probabilities (each < 1) gives a vanishingly small number that underflows to zero on
a computer. The fix, used everywhere in RL code, is to work with **log-probabilities**.

A log-probability is just `log(p)`. Key facts:

- Since `0 < p ≤ 1`, `log(p)` is always `≤ 0`. A log-prob of `0` means "probability 1, certain"; a very
negative log-prob means "very unlikely."
- **Products become sums:** `log(a·b) = log(a) + log(b)`. So the log-probability of a whole sequence is
the **sum** of per-token log-probs — numerically stable, no underflow.

```
log P(response) = Σ_t  log P(token_t | everything before it)
```

🧮 For the sequence above: `log(0.134) = log(0.558)+log(0.30)+log(0.80) = -0.583 -1.204 -0.223 = -2.01`.

In the code, the function that computes the log-prob of *the specific token that was actually chosen*
(out of the whole vocab distribution) is `log_probs_from_logits(logits, labels)`:
✓ `dp_actor.py:153` — `log_probs = self.log_probs_from_logits(logits, responses)`. Here `responses`
are the token ids that were actually generated; for each one it returns `log p` of that token under
the model's current distribution. The result has shape `(batch, response_length)` — **one log-prob
per generated token**.

⚠️ **Temperature must match.** Right before computing log-probs, the code divides logits by the same
temperature used at generation: ✓ `dp_actor.py:151` `logits.div_(temperature)`. This is essential: a
log-prob is only meaningful relative to a specific distribution. If the model generated tokens under
`T=1.0`, we must evaluate their log-probs under the `T=1.0` distribution too, or every later
calculation (the "ratio" in Chapter 6) is miscalibrated.

**Hold onto this:** "log-prob of a token" = how much the model liked the token it produced, as a
number ≤ 0. RL works almost entirely by *raising or lowering these log-probs*.

> **🌐 Generalizes:** tokens, logits, softmax, log-probabilities, autoregressive sampling, and
> temperature are true of **every** transformer LM in any framework. The rule "a token's log-prob is
> only meaningful relative to the exact (temperature-scaled) distribution that produced it" is
> universal and underlies the importance ratio in Ch. 6.
> **📦 Ours:** Qwen3's ~150k-entry vocabulary; the specific call sites `dp_actor.py:151` (logits ÷
> temperature) and `:153` (`log_probs_from_logits`) where verl computes these.

---

# Chapter 2 — The model as an RL "policy"

RL has its own vocabulary, borrowed from robots and games. Here is the standard vocabulary and, beside
it, exactly what each term *is* for a language model.


| RL term                  | General meaning              | For our VLM                                                |
| ------------------------ | ---------------------------- | ---------------------------------------------------------- |
| **Agent**                | the thing that acts          | the model                                                  |
| **Environment**          | what the agent acts in       | the text/image context being built up                      |
| **State** `s`            | the current situation        | the prompt + all tokens generated so far                   |
| **Action** `a`           | a choice the agent makes     | the next token it emits                                    |
| **Policy** `π_θ(a        | s)`                          | rule for choosing actions                                  |
| **Episode / trajectory** | one run from start to finish | one full response, from first token to `<end>`             |
| **Reward** `R`           | score for behavior           | the verifiable score for the finished response (Ch. 3)     |
| **Return**               | total reward of an episode   | here, just the single end reward (no intermediate rewards) |


The crucial line: **the model *is* the policy.** "Policy" `π_θ` literally means the function
`(context) → probability distribution over next token`, which is exactly what Chapter 1 described. The
subscript `θ` (theta) is the set of all the model's weights. "Improving the policy" = "changing the
weights θ so the next-token choices lead to higher reward."

Two more terms you'll meet constantly:

- **On-policy data**: responses generated by the *current* version of the model. RL needs fresh
on-policy samples because it's learning from *its own* behavior, not a fixed dataset.
- **Rollout**: the act of generating responses by running the policy forward (Chapter 1's loop). In
our system a fast engine called **vLLM** does rollouts (Part III); the word "rollout" is used both
for the act and for the generated responses.

**The objective, stated plainly.** We want to adjust `θ` to **maximize the average reward of the
responses the model generates**:

```
maximize over θ:   J(θ) = E[ R(response) ]   where response ~ π_θ
```

`E[·]` means "expected value / average." `response ~ π_θ` means "responses sampled by running the
current policy." Everything from here is about how to actually *do* this maximization. Chapters 3–5
build the *signal* (reward → advantage); Chapters 6–8 turn the signal into a weight change.

> **🌐 Generalizes:** the whole RL vocabulary (agent / state / action / policy / episode / return) and
> the central mapping — *an autoregressive LM **is** a policy `π_θ`*, with state = context so far,
> action = next token. On-policy vs off-policy data; "rollout." All universal.
> **📦 Ours:** **vLLM** is our specific rollout engine (a tooling choice, detailed in Part III) — but
> nothing *conceptual* in this chapter is project-specific.

---

# Chapter 3 — Reward, and what "token-level reward" really means

This is the chapter I previously skipped. We go slowly.

## 3.1 The reward function: one number for one response

A **reward function** takes a finished response and returns a single number scoring it. Ours is in
`VLM-CapCurriculum/training/reward_functions/math.py`, function `compute_score`. For each response it
computes two pieces:

**Accuracy reward** ✓ `math.py:54` `accuracy_reward`:

```python
answer = extract_boxed_content(response)   # pull whatever is inside \boxed{...}
... grade_answer(answer, ground_truth) ...  # does it match the correct answer?
return 1.0 if correct else 0.0
```

So accuracy is `1.0` for a correct final answer, `0.0` otherwise. `extract_boxed_content` finds the
text the model wrote inside `\boxed{ }` — that's the convention we force the model to use so the
answer is machine-readable.

**Format reward** ✓ `math.py:63` `format_reward`:

```python
pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
return 1.0 if re.fullmatch(pattern, response) else 0.0
```

This is `1.0` only if the response has the exact shape `<think> ...reasoning... </think> ... \boxed{answer}`. It rewards *following the required output structure*, independent of correctness.

**Combined** ✓ `math.py:85`:

```python
"overall": (1 - 0.1) * accuracy_score + 0.1 * format_score
```

The final scalar is `0.9 × accuracy + 0.1 × format`. So a fully correct, well-formatted answer scores
`0.9·1 + 0.1·1 = 1.0`; correct answer but malformed scores `0.9`; well-formatted but wrong scores
`0.1`; wrong and malformed scores `0.0`. The small `0.1` format weight keeps the model emitting
parseable output without letting "looking right" overwhelm "being right."

⚠️ This is why **Instruct** is the right backbone: the format regex needs the literal text
`<think>...</think>...\boxed{}`. An Instruct model, told to produce that, emits it as plain text. A
"Thinking" model may hide its reasoning in a separate channel, which the regex wouldn't see → near-zero
format reward and distorted training.

## 3.2 The shape mismatch: scalar reward vs. per-token math

Here is the heart of "token-level." The reward function gives **one number per response**. But the
learning math (Chapters 4–6) treats **every generated token as a separate action** that needs its own
slice of credit. So we need the reward represented as **one number per token**, not one per response.

A "**token-level**" quantity is simply *a value for every token position* — a tensor of shape
`(batch_size, response_length)`: for each of the `batch_size` responses, a number for each of its (up
to) `response_length` token positions. "Token-level reward" = a reward laid out this way.

## 3.3 The exact code that converts scalar → token-level

✓ `function.py:96-100` (this is `compute_reward_batch`, the path our config uses):

```python
reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)  # (a) all zeros
for i, score in enumerate(scores):
    cur_response_length = int(response_length[i].item())                         # (b) real length
    reward_tensor[i, cur_response_length - 1] = score["overall"]                 # (c) drop scalar
```

Line by line:

- **(a)** Make a tensor the same shape as `responses` — `(batch_size, response_length)` — filled
entirely with **zeros**.
- **(b)** `response_length[i]` is how many *real* (non-padding — see 3.5) tokens response `i` has. It
comes from summing the response mask: ✓ `function.py:80`
`response_length = torch.sum(data.batch["response_mask"], dim=-1)`.
- **(c)** Place the scalar reward `score["overall"]` at exactly **one** position: index
`cur_response_length - 1`, i.e. the **last real token** of the response. Every other position stays
`0.0`.

🧮 Response `i` is 6 tokens long and scored `0.9`. Its reward row is:
`[0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0, 0, ...]` — zeros everywhere except `0.9` at index 5 (the 6th token),
then zeros for the padded tail.

## 3.4 Why put it on the last token? "Sparse" and "terminal" reward

The response is one **episode**. The model only finds out "was this good?" *after the whole thing is
finished*. There is no per-token feedback ("token 3 was great") — only an end-of-episode verdict. So
the reward naturally lives at the **terminal** step: the last token. This is called a **sparse reward**
(almost all positions are zero) and specifically a **terminal/outcome reward** (the one signal arrives
at the end). Contrast: a "dense" reward would score every token; "process supervision" would score each
reasoning step. We use outcome reward — simplest, and exactly what a verifiable checker gives.

To recover the per-response scalar from the token-level tensor, you just **sum across tokens** (all the
zeros plus the one real value). You will see exactly this in Chapter 5:
✓ `core_algos.py:199` `scores = token_level_rewards.sum(dim=-1)`.

## 3.5 Two supporting concepts you now need: padding and the response mask

**Padding.** A batch is a rectangular tensor — every row must be the same length. But responses have
different lengths (one is 6 tokens, another 200). To make them fit one tensor, short responses are
filled out with a meaningless **padding token** up to the longest length in the batch. These pad
positions are not real model outputs; they must be ignored in every calculation.

**Response mask.** The bookkeeping that marks which positions are real. `response_mask` is a tensor of
the same `(batch, response_length)` shape containing `1` for a real generated token and `0` for
padding (and for any prompt positions). Wherever you must "only count real tokens," you multiply by
this mask. That's why summing it (3.3b) gives the true length, and why later we average losses
*through* it.

## 3.6 `token_level_scores` vs `token_level_rewards` — a naming subtlety

You'll see both names. They differ by whether the KL penalty (Chapter 7) has been folded in yet:

- `**token_level_scores`** = the raw verifiable reward tensor from `function.py` (3.3). ✓ set at
`ray_trainer.py:630`.
- `**token_level_rewards`** = what advantage is actually computed from. In our config (`use_kl_loss: true`) it's just copied from scores unchanged (✓ `ray_trainer.py:640`); in the *other* KL mode it's
`scores − kl_coef × KL` (✓ `ray_trainer.py:127`). We unpack this fully in Chapter 7.

For now: the reward signal entering the next chapters is a sparse `(batch, response_length)` tensor,
one real number per response sitting on its last token.

> **🌐 Generalizes:** reward functions; verifiable (rule) vs learned (RLHF) reward; sparse / terminal /
> outcome reward; the fact that a per-*episode* scalar reward must be spread to per-*token* values for
> policy-gradient math; padding and masking of variable-length sequences.
> **📦 Ours:** the `0.9·accuracy + 0.1·format` weighting, the `\boxed{}` + `<think>` output conventions,
> `math.py`. ⚠️ Also **verl-specific representation choices**, not universal laws: placing the scalar on
> the **last token** and recovering it with `sum` (`function.py:100`, `core_algos.py:199`), and the
> `token_level_scores` vs `token_level_rewards` naming. Other RL frameworks lay reward out differently.

---

# Chapter 4 — How learning works: policy gradients from zero

We can now ask the central question: given rewards, *how do we change the weights θ to get more
reward?* This chapter derives the answer from scratch. No prior RL assumed.

## 4.1 The obstacle: you can't differentiate through sampling

We want to increase `J(θ) = E[R]`, the average reward. The natural ML move is gradient ascent: nudge
`θ` in the direction `∇θ J` (the gradient — the direction of steepest increase). But there's a snag:
the response is produced by **random sampling** (Chapter 1.4). You cannot directly take a derivative
through a coin-flip / dice-roll. So `∇θ E[R]` is not obviously computable.

## 4.2 The policy gradient trick (REINFORCE)

There's a classic identity that rescues us. Using `∇ log f = (∇ f)/f` (so `∇f = f · ∇log f`), one can
show:

```
∇θ E[R]  =  E[ R · ∇θ log π_θ(response) ]
```

In words: **the gradient of the average reward equals the average of (reward × the gradient of the
log-probability of the response).** The right-hand side *is* computable — we can sample responses,
and `log π_θ(response)` is differentiable in `θ` (it's the sum of per-token log-probs from Ch. 1.6).
This estimator is called **REINFORCE**, and the whole family is **policy gradients**.

**Read the formula as an instruction.** `log π_θ(response)` is the sum of the log-probs of the tokens
the model chose. Its gradient points in the direction of `θ` that would make *those tokens more
likely*. Multiplying by `R` scales that push by how good the response was:

- High reward `R` → push hard to make these tokens **more** likely next time.
- Reward near 0 → little push.
- (If reward could be negative → push to make these tokens **less** likely.)

That is the entire intuition of RL on language models: **make the tokens of good responses more
probable, the tokens of bad responses less probable, scaled by how good/bad they were.** Everything
else (baselines, GRPO, clipping) is making this *stable* and *low-variance*.

## 4.3 The problem with raw REINFORCE: variance

Using raw reward `R` as the multiplier works in theory but is **noisy** (high variance), which makes
training unstable and slow. A vivid failure mode: if every response to a prompt gets a *positive*
reward (say all between 0.1 and 1.0), REINFORCE pushes **up** the probability of *every* response —
even the relatively bad ones — just by different amounts. The model gets a confusing signal: "do more
of everything." What we actually want to tell it is "do more of the **better-than-average** responses
and less of the **worse-than-average** ones."

## 4.4 Baselines: subtract "what's typical"

The fix is to subtract a **baseline** `b` from the reward before multiplying:

```
∇θ E[R]  =  E[ (R − b) · ∇θ log π_θ(response) ]
```

⚠️ **Why this is allowed (unbiasedness).** As long as `b` does not depend on the action taken, adding
it changes nothing in expectation: `E[b · ∇log π] = b · Σ π · ∇log π = b · Σ ∇π = b · ∇(Σπ) = b · ∇(1) = 0`. (The probabilities always sum to 1, whose gradient is 0.) So subtracting a baseline leaves the
gradient **unbiased** — it still points the right way on average — but, chosen well, dramatically
**reduces its variance**. This is one of the most important tricks in all of RL.

## 4.5 Advantage = reward − baseline

The quantity `(R − b)` has a name: the **advantage**, written `A`. It measures **how much better this
particular response was than the baseline** — than what's "typical." Positive advantage → better than
typical → make it more likely. Negative advantage → worse than typical → make it less likely. The
learning rule becomes:

```
push each response's tokens by   A = (R − baseline)
```

The only remaining question is: **what baseline?** A good baseline is the *expected reward for this
prompt* — the average over responses. Two famous answers:

- **PPO** trains a *second neural network*, a **value function** `V(s)`, to predict the expected
reward, and uses that as the baseline. Powerful, but it's a whole extra model to train and store.
- **GRPO** uses a beautifully cheap trick instead: sample a **group** of responses to the *same*
prompt and use **their average reward** as the baseline. No second network. That's the next chapter.

## 4.6 The family of policy-gradient methods (where GRPO sits)

GRPO is one point in a well-populated family, and the *only* thing most members disagree on is **what
baseline to subtract** (§4.4). Knowing this lineage is exactly what an interviewer probes — and our
repo conveniently implements several siblings, so this doubles as a map of `core_algos.py`.

- **REINFORCE** (1992): the bare `∇θ E[R] = E[R · ∇log π]` estimator (§4.2), no baseline. High variance.
- **REINFORCE + baseline:** subtract a baseline `b` to cut variance (§4.4). Everything below is just a
different recipe for `b`.
- **Actor–Critic / PPO** (2017): baseline = a **learned value network** `V(s)`; PPO adds the *clipped*
objective (Ch. 6) to safely reuse a batch, and per-token advantages via **GAE**. Powerful but needs a
whole second model.
- **GRPO** (2024, DeepSeekMath): baseline = the **empirical mean reward of a group** of samples for the
same prompt (Ch. 5). No value network. ← *our method.*
- **Siblings, all living in `core_algos.py`** — each just a different baseline/advantage recipe:
**RLOO** (leave-one-out group mean), **REINFORCE++** (global/running baseline), **REMAX**
(greedy-rollout baseline), **GAE** (value-function bootstrap). ✓ enumerated at `core_algos.py:81-86`
and each registered as a `compute_*_advantage` function (e.g. `compute_grpo_outcome_advantage:176`,
`compute_gae_advantage_return:126`, `compute_rloo_outcome_advantage:269`).

So the whole of EasyR1's advantage code is "**pick your baseline**," selected by one config key
`algorithm.adv_estimator` (we set `grpo`). Switching it to `rloo` would change exactly one function
call in the loop (Ch. 8) — nothing else.

> **🌐 Generalizes:** **all of Chapter 4.** REINFORCE, the log-derivative trick, the variance problem,
> baselines + the unbiasedness proof, advantage = reward − baseline, and the family tree above. This is
> textbook policy-gradient RL — it transfers unchanged to robotics, games, and every LLM-RL algorithm.
> **📦 Ours:** essentially nothing — Chapter 4 is pure foundation, identical in any framework. The only
> repo-specific facts are *which* family members `core_algos.py` happens to implement.

---

# Chapter 5 — GRPO: the group is the baseline

**GRPO = Group Relative Policy Optimization.** The whole idea is the baseline from Chapter 4.5: for
each prompt, generate a **group** of `G` responses, and use the group's own average reward as the
baseline. "Relative" because each response is judged *relative to its group-mates*, not an absolute bar.

We use group size `G = 5` (config `worker.rollout.n: 5`). So for every prompt, the model writes 5
different answers (randomness from `T=1.0` sampling makes them differ), all 5 get scored, and each is
pushed up or down depending on whether it beat the average of the 5.

## 5.1 The computation, line by line

✓ `core_algos.py:176` `compute_grpo_outcome_advantage`. Inputs: `token_level_rewards` (the sparse
tensor from Ch. 3), `response_mask` (Ch. 3.5), and `index` (which prompt each response belongs to).

```python
scores = token_level_rewards.sum(dim=-1)          # 199
```

**Recover the per-response scalar.** Summing the token-level tensor over the token axis collapses the
"zeros + one real value at the last token" row back to that single value (Ch. 3.4). `scores` is now one
number per response: shape `(batch,)`.

```python
id2score = defaultdict(list)
for i in range(bsz):
    id2score[index[i]].append(scores[i])           # 204-205
```

**Group the responses by prompt.** `index[i]` is an id telling which prompt response `i` came from
(it's a UUID assigned upstream — see Ch. 8). This builds, for each prompt, the list of its `G=5`
response-scores.

```python
assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."   # 208
id2mean[idx] = torch.mean(...)                                # 209
id2std[idx]  = torch.std(...)                                 # 210
```

**Group statistics.** For each prompt, compute the **mean** and **standard deviation** of its 5
scores. The assertion enforces `G > 1` — with a group of 1 there is no "average to compare against,"
the whole method is meaningless.

(Standard deviation = a measure of spread: how much the 5 scores differ from their average. Small std =
the 5 responses scored similarly; large std = some much better than others.)

```python
scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)   # 213
```

**The advantage.** Each response's score becomes `(its score − its group's mean) / (its group's std)`.

- The `− mean` part is the baseline subtraction from Ch. 4.5: "how far above/below the group average."
- The `/ std` part **normalizes** the spread so advantages are on a consistent scale (roughly unit
variance) regardless of whether a prompt's rewards were tightly or widely spread.
- `eps = 1e-6` is a tiny number added to avoid dividing by zero (Ch. 5.3).

This `(x − mean)/std` operation is called a **z-score** (standardization). So: **a GRPO advantage is
the z-score of a response's reward within its group of 5.**

🧮 Worked example. One prompt, 5 responses, accuracy rewards `[1, 1, 0, 0, 0]` (2 of 5 correct):

- mean `= 0.4`, std `≈ 0.49`.
- Advantages: correct ones `(1−0.4)/0.49 ≈ +1.22`; wrong ones `(0−0.4)/0.49 ≈ −0.82`.
- Interpretation: the 2 correct responses get a **positive** advantage (their tokens will be made more
likely); the 3 wrong ones get a **negative** advantage (their tokens made less likely). The model
learns "be more like the responses that got it right, less like the ones that didn't" — entirely
from *relative* comparison, no external target answer needed.

```python
returns = scores.unsqueeze(-1) * response_mask    # 215
return returns, returns                            # 216
```

**Broadcast back to token-level.** Each response now has one scalar advantage. `unsqueeze(-1)` turns
the `(batch,)` vector into `(batch, 1)`, and multiplying by `response_mask` `(batch, response_length)`
copies that single advantage onto **every real token** of the response (and `0` onto padding). So every
token in a response carries the **same** advantage.

⚠️ **Why every token gets the *same* advantage.** Because the reward is an *outcome* reward (Ch. 3.4) —
we only know the whole response was good or bad, not which specific token deserves credit. With no
finer signal, every action (token) in the episode shares the credit equally. This is the defining
characteristic of outcome-supervised RL. (A value function, as in full PPO with GAE, *could* assign
different credit per token — GRPO deliberately trades that resolution away for not needing a value
network.) The function returns `returns, returns` — the same tensor twice — because for GRPO the
"advantage" and the "return" are identical; the two-value signature exists to match other estimators
like GAE that distinguish them.

## 5.2 Why this is clever (the one-paragraph summary for an interview)

PPO needs a second neural network (a value function) to estimate the baseline "expected reward,"
doubling the models in memory and adding its own training instabilities. GRPO observes that if you
simply sample several responses per prompt, the **empirical mean of those rewards is already an
unbiased estimate of the expected reward** — a free baseline. You pay for it with extra generations
(5× the rollouts) instead of extra parameters. On top, dividing by the group std standardizes the
advantage scale across easy and hard prompts.

## 5.3 The subtleties that bite (interview gold)

⚠️ **"Dead groups."** If all 5 responses get the *same* reward (all correct, or all wrong), the group
std is `0`, mean equals each score, so every advantage is `0/(0+eps) = 0`. Zero advantage → zero
gradient (Ch. 4.5) → **that prompt teaches the model nothing this step.** This is wasted compute. It's
exactly why the optional **online filtering** (DAPO) exists — config `algorithm.online_filtering`,
`filter_low: 0.01`, `filter_high: 0.99`: before training, drop any group whose *mean* reward is ~0 (all
wrong) or ~1 (all right), keeping only groups with a mix to learn from. (We'll see where this happens
in the loop in Ch. 8.)

⚠️ **The std-normalization is contested.** Dividing by the group std is a *choice*, not a law. The
"Dr. GRPO" critique argues it subtly **biases** learning: prompts where responses happen to have low
std get their advantages blown up (divided by a small number), so easy-ish prompts with little spread
can dominate the gradient. Some variants drop the `/std`. DeepSeek's original GRPO keeps it. Worth
knowing we *use* it (it's hard-coded in `core_algos.py:213`) and that it's debatable.

⚠️ `**G = 5` is small.** DeepSeek's GRPO used groups of ~64. A smaller group means the mean and std are
estimated from only 5 numbers — noisier baseline, higher-variance advantages. The paper chose 5 to keep
generation cost down (each prompt costs 5 full generations). This is a real accuracy-vs-compute knob.

⚠️ **The reward must have spread to learn.** With binary accuracy and `G=5`, the *most informative*
groups are the mixed ones (e.g. 2/5 or 3/5 correct). All-correct or all-wrong groups are dead (above).
So GRPO implicitly relies on the model being at the "edge of its ability" for each prompt — which is
why **difficulty curricula** (the paper sorts data by pass-rate) matter: they keep groups in the
informative middle.

> **🌐 Generalizes:** GRPO itself; "the group mean is a free, unbiased baseline"; the z-score advantage;
> outcome-reward ⇒ the *same* advantage on every token; dead groups (zero-variance ⇒ zero gradient); the
> std-normalization debate; the group-size ↔ variance/compute tradeoff; and the family tree (§4.6).
> **📦 Ours:** `G=5` (`worker.rollout.n`); the `online_filtering` thresholds `0.01 / 0.99`;
> difficulty-curriculum data ordering; the exact implementation at `core_algos.py:176-216`, including
> the verl-specific `return returns, returns` signature.

---

## End of Part I

You now understand the **learning signal** end to end:

1. The model is a policy: context → distribution over next token, parameterized by weights `θ` (Ch. 1–2).
2. A verifiable reward scores each finished response with one scalar (Ch. 3.1), laid into a sparse
  token-level tensor at the last token (Ch. 3.3–3.4).
3. Policy gradients say: push the tokens of a response by `(reward − baseline)` = the **advantage**
  (Ch. 4).
4. GRPO computes that advantage as the **z-score of the reward within a group of 5 responses** to the
  same prompt, then copies it onto every token (Ch. 5).

The output of Part I is, for every generated token, **one advantage number** telling the optimizer
which direction to push that token's probability and how hard.

The output of Part I is, for every generated token, **one advantage number** telling the optimizer
which direction to push that token's probability and how hard. Part II turns that number into an actual
change to the weights `θ`, safely.

---

---

# PART II — THE WEIGHT UPDATE

# Chapter 6 — Reusing data safely: the importance ratio and PPO clipping

Part I ended with: for each token, an advantage `A`. The naive thing (REINFORCE, §4.2) is to push each
token's log-prob by `A` once and throw the data away. This chapter explains why we *don't* do that, and
what we do instead.

## 6.1 The motivation: generation is expensive, so reuse the batch

Producing the rollouts is the costly part of RL: for every prompt we ran the model forward hundreds of
times (one forward per generated token × 5 responses), on a slow autoregressive loop, via vLLM. Having
paid that price, we'd like to extract **several gradient steps** from each batch of rollouts, not just
one. That is far more sample-efficient.

But there's a catch that is the entire subject of this chapter. The policy-gradient formula (§4.2) is
**on-policy**: it is only valid for data generated by the *current* policy `π_θ`. The instant we take
one gradient step, `θ` changes, so the model that *would* generate now differs from the model that
*did* generate the batch. The data is now **off-policy** — generated by an older policy `π_old` than the
one we're updating, `π_new`. Using it naively gives a *wrong* gradient. We need a correction.

(Terminology: **on-policy** = learn only from data the current policy produced; **off-policy** = learn
from data some other policy produced. Pure REINFORCE is strictly on-policy. PPO/GRPO make limited
off-policy reuse safe.)

## 6.2 Importance sampling — the general statistical tool

The correction comes from a classic trick called **importance sampling**, which answers: *how do I
estimate an average under one distribution `P` using samples drawn from a different distribution `Q`?*

The identity:

```
E_{x∼P}[ f(x) ]  =  E_{x∼Q}[ f(x) · P(x)/Q(x) ]
```

**Why it's true** (one line): `E_{x∼Q}[f(x)·P(x)/Q(x)] = Σ_x Q(x)·f(x)·P(x)/Q(x) = Σ_x P(x)·f(x) = E_{x∼P}[f(x)]`. The `Q(x)` cancels. The factor `P(x)/Q(x)` is the **importance ratio**: it *re-weights*
each sample from `Q` to count as if it came from `P` — samples that `P` finds more likely than `Q` get
up-weighted, and vice versa.

🧮 If `Q` gave you a sample that `P` considers twice as likely (`P/Q = 2`), you count that sample
double; if `P` considers it half as likely (`P/Q = 0.5`), you count it half. That reweighting makes
`Q`-samples represent a `P`-average.

## 6.3 The ratio in our setting

Here `P = π_new` (the policy we want the gradient for) and `Q = π_old` (the policy that generated the
tokens). For a single token, the importance ratio is:

```
ratio = π_new(token | context) / π_old(token | context)
```

Because we store **log**-probabilities (§1.6), this is computed as a difference-then-exponentiate:

```
ratio = exp( log π_new − log π_old )
```

✓ In code, `core_algos.py:464` `negative_approx_kl = log_probs - old_log_probs`, then
`core_algos.py:478` `ratio = torch.exp(torch.clamp(log_importance_ratio, -20.0, 20.0))`. The clamp to
`[-20, 20]` before `exp` is pure numerical safety (it stops `exp` from overflowing to infinity on a
wild log-ratio).

Where do the two log-probs come from?

- `**old_log_probs**`: computed **once**, right after generation and *before* any weight update.
✓ `ray_trainer.py:611` `old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)`. This is
`log π_old` — frozen for the whole update phase.
- `**log_probs`** (the "new" ones): recomputed **inside the update loop**, with the *current* (already
partly-updated) weights, every micro-step. ✓ `dp_actor.py:257`
`log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)`.

So at the very first update, `π_new == π_old` and `ratio == 1` everywhere (the surrogate equals plain
policy gradient). As updates proceed and weights move, `ratio` drifts from 1, and that drift is exactly
what we must keep under control.

## 6.4 Why an unconstrained ratio is dangerous

The importance-weighted objective for one token is `ratio · A`. Suppose a token had a large positive
advantage `A`. The gradient pushes to increase `π_new(token)`, which increases `ratio`, which increases
the objective, which pushes harder... A single batch could send `ratio` to 5, 10, 50 — a colossal,
confident update built on a *finite, noisy* batch of 5 samples per prompt. This is how RL training
"falls off a cliff": one over-large step wrecks the policy, and because the policy generates its own
future data, it may never recover. We need a **trust region**: a rule that says "don't move the policy
too far from `π_old` in any single round of updates."

## 6.5 PPO's clipped objective — the trust region made cheap

**PPO (Proximal Policy Optimization)** enforces the trust region with a brilliantly simple device:
**clip the ratio** so that moving it beyond a band around 1 yields *no further benefit*. The standard
PPO per-token objective (the thing we *maximize*) is:

```
L = min( ratio · A ,  clip(ratio, 1−ε, 1+ε) · A )
```

where `clip(ratio, 1−ε, 1+ε)` just forces the ratio to stay within `[1−ε, 1+ε]` (e.g. `ε=0.2` →
`[0.8, 1.2]`). Read the `min` through the two advantage cases — this is the whole idea:

**Case `A > 0`** (good token, want it *more* likely → want `ratio > 1`):

- If `ratio ≤ 1+ε`: both terms equal `ratio·A`; normal gradient, push it up.
- If `ratio > 1+ε`: the clipped term `(1+ε)·A` is **smaller** than `ratio·A`, so `min` picks it. But
`(1+ε)·A` is a **constant** in `ratio` → its gradient w.r.t. the policy is **zero**. The incentive to
push this token higher **switches off** once you've moved it `+ε`. You got your reward; stop.

**Case `A < 0*`* (bad token, want it *less* likely → want `ratio < 1`):

- If `ratio ≥ 1−ε`: both terms equal `ratio·A`; normal gradient, push it down.
- If `ratio < 1−ε`: the clipped term `(1−ε)·A` is **larger** (less negative) than `ratio·A`, so `min`
picks `ratio·A`... wait — for `A<0`, `min` of the two picks the **more negative**, which is the
unclipped `ratio·A` only while `ratio ≥ 1−ε`; once `ratio < 1−ε` the clipped constant is selected and
again the gradient is **zero**. The incentive to push this token lower **switches off** after `−ε`.

**Net effect:** the policy is free to adjust each token's probability by up to about `±ε` per update
round; beyond that, the gradient flatlines, so no single batch can yank the policy far. The trust region
is enforced *per token*, with no extra computation — just a `min` and a `clip`.

🧮 `ε = 0.2`, a token with `A = +2`. If an update has driven `ratio = 1.5` (policy now makes this token
1.5× as likely as `π_old` did), the objective uses `min(1.5·2, 1.2·2) = min(3.0, 2.4) = 2.4`, the
clipped value — and since 2.4 is constant in the policy, that token contributes **no gradient** this
step. Had `ratio` been `1.1` (within band), it'd use `1.1·2 = 2.2` and contribute a normal gradient.

## 6.6 How the code implements it (with two extensions)

✓ `core_algos.py:497-504`, `loss_type="default"` (our setting). The code works with **negated** losses
(it minimizes `loss = −objective`), so PPO's `min` becomes a `max` of negated terms:

```python
pg_loss  = -advantages * ratio            # −(ratio·A)            : the unclipped term
pg_loss2 = -advantages * clipped_ratio    # −(clip(ratio)·A)      : the clipped term
pg_loss3 = -advantages * clip_ratio_dual  # −(C·A)                : the dual-clip floor (extension 2)

clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)                     # == PPO's min on the objective
clipped_pg_loss_lower  = torch.min(clipped_pg_loss_higher, pg_loss3)
final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
```

- `torch.max(pg_loss, pg_loss2)` on the *negated* losses is identical to `min(ratio·A, clip·A)` on the
objective — that's the standard PPO clip from §6.5.
- **Extension 1 — asymmetric clip (DAPO).** Notice `clipped_ratio` uses *two different* bounds:
✓ `core_algos.py:479-481` clamps the log-ratio to `[log(1−clip_ratio_low), log(1+clip_ratio_high)]`.
Our config: `clip_ratio_low=0.2`, `clip_ratio_high=0.3`. So the band is **asymmetric**: `[0.8, 1.3]`.
Allowing a *wider upper* bound (`+0.3` vs `−0.2`) lets the policy raise the probability of good tokens
a bit more aggressively than it lowers bad ones — DAPO found this helps exploration/entropy. Plain PPO
uses `low = high = 0.2`.
- **Extension 2 — dual clip.** For `A < 0`, the extra `torch.min(..., pg_loss3)` puts a **floor** on how
large the (negated) loss can get when both `A` is very negative *and* `ratio` has exploded. Without it,
a hugely off-policy bad token could dominate the batch with an enormous gradient. `clip_ratio_dual=3.0`
(a constant `C`) caps that. This only activates for negative advantages — hence the `where(A < 0, …)`.

⚠️ The `loss_type` config also offers `gspo`, `gspo_token`, `cispo`, `sapo` (✓ `core_algos.py:465-495`)
— these compute the ratio at the *sequence* level instead of per-token, or use sigmoid gates. We use
`default` (per-token PPO clip). Knowing they exist (and that they're swappable via one config key) is
enough for now.

## 6.7 The diagnostics this produces (read these during training)

✓ `core_algos.py:484-505` builds metrics you will watch in the logs:

- `**ppo_kl`** `= mean(old_log − new_log)` (`:484`): a cheap measure of *how far the update moved the
policy* from `π_old` this step. Should stay small/positive. A spike means a big, possibly unstable
update.
- `**pg_clipfrac_higher`** (`:502`): the *fraction of tokens that hit the upper clip*. If this is high,
many updates are being capped — the policy is trying to move further than the trust region allows
(often fine early, concerning if persistently huge).
- `**pg_clipfrac_lower`** (`:505`): fraction hitting the dual-clip floor (negative-advantage side).
- `**entropy_loss*`* `= mean(−log_probs)` (`:486`): a proxy for the policy's **entropy** (randomness).  
Falling entropy = the policy is becoming more    ) — a key thing to monitor for our runs.

## 6.8 Where this sits in the batch structure (preview of Part III)

✓ `dp_actor.py:219 update_policy` wraps all of the above in three nested loops, which we'll fully unpack
in Part III but should name now because `old_log_probs` and the ratio depend on them:

- **epoch** (`ppo_epochs`, we use **1**): how many full passes over the rollout batch. `old_log_probs`
are frozen across all of them; with >1 epoch, later epochs are increasingly off-policy (the clip is
what makes that safe). GRPO typically uses 1.
- **mini-batch** (`global_batch_size`): one **optimizer step** per mini-batch (✓ `dp_actor.py:294`). So
weights *do* change within an epoch, across mini-batches — which is why even at `ppo_epochs=1` the
ratio matters.
- **micro-batch** (`micro_batch_size_per_device_for_update`): gradients are **accumulated** over
micro-batches (✓ `dp_actor.py:288 loss.backward()` with no step until `:294`). Micro-batches exist
purely to fit a large mini-batch through limited GPU memory — they don't change the math, only the
memory footprint.

> **🌐 Generalizes:** importance sampling and the importance ratio; on-policy vs off-policy; the
> trust-region idea; PPO's clipped surrogate and *why* clipping zeroes the gradient past the band; the
> entropy-collapse failure mode; gradient accumulation (micro-batches) as a memory tool. All of this is
> standard across modern RL (PPO is the workhorse of RLHF too).
> **📦 Ours:** `clip_ratio_low=0.2`, `clip_ratio_high=0.3` (DAPO asymmetry), `clip_ratio_dual=3.0`,
> `loss_type="default"`, `ppo_epochs=1`; the negated-loss `max`/`min` implementation and the exact
> metric names in `core_algos.py`. The `gspo/cispo/sapo` alternatives are repo features we don't use.

---

# Chapter 7 — KL regularization: don't wander away from the start

PPO clipping (Ch. 6) limits how far the policy moves *per update*. KL regularization limits how far it
moves *in total* from where it began. These are different jobs and the algorithm uses both.

## 7.1 The motivation: reward hacking and forgetting

We are optimizing a *proxy* — the reward function — not "being a good model." If we let the policy
chase reward without restraint, it finds degenerate shortcuts:

- **Reward hacking:** exploiting quirks of the checker (e.g. always emitting a `\boxed{}` with a
guessed letter to farm format/partial reward) rather than truly solving the task.
- **Catastrophic forgetting:** drifting so far that it loses general capabilities it had at the
start (fluency, broad knowledge) that the narrow reward doesn't measure.
- **Mode collapse:** the entropy collapse from §6.7 — converging to one rigid output style.

The guard is to **anchor the policy to a frozen copy of the model it started from**, called the
**reference policy** `π_ref`, and penalize drifting away from it. `π_ref` is the initial
Qwen3-VL-4B-Instruct, never updated.

## 7.2 KL divergence — what "how far apart are two distributions" means

The standard measure of how different one probability distribution is from another is the
**Kullback–Leibler (KL) divergence**:

```
KL(P ‖ Q)  =  Σ_x P(x) · log( P(x) / Q(x) )
```

Read it as: the average (under `P`) of the log-ratio between `P` and `Q`. Properties to know:

- `KL(P‖Q) ≥ 0` always, and `= 0` **iff** `P` and `Q` are identical. So it's a "distance-like" score
(0 = same, bigger = more different).
- It is **not symmetric**: `KL(P‖Q) ≠ KL(Q‖P)` in general. (So you must be careful which way round.)
- For our policies it's evaluated **per token**: at each generated position, `P = π_new(·|context)` and
`Q = π_ref(·|context)` are full next-token distributions, and we want them to stay close.

## 7.3 Estimating KL from one sample — and why the k3 form

Computing the full sum over the ~150k-vocab at every token would be expensive, and we only have the
*one* token that was actually sampled, not the whole distribution. So we **estimate** the KL from that
single token's log-probs. There are several estimators (Schulman's well-known note catalogs them):

- **Naive ("k1"):** `KL ≈ log π_new − log π_ref`. Unbiased, but **high variance** and, for a single
token, frequently **negative** — which is nonsensical for a quantity that should be ≥ 0, and worse,
it creates a perverse incentive (the optimizer could *lower* an already-low estimate by diverging).
- **k3 (what we use):** define `x = log π_ref − log π_new`. Then `KL ≈ exp(x) − x − 1`.
✓ `core_algos.py:590-594`:
  ```python
  kl  = (ref_log_probs - log_probs).clamp(-20.0, 20.0)   # x = log π_ref − log π_new
  kld = (kl.exp() - kl - 1)                               # e^x − x − 1
  return torch.clamp(kld, min=-10.0, max=10.0)
  ```
  Why this is better: the function `e^x − x − 1` is **≥ 0 for every `x`** (it's 0 only at `x=0`, i.e.
  when the two policies agree on this token), it's an **unbiased** estimator of the true KL, and it has
  **much lower variance** than the naive difference. The `clamp`s are numerical guards.

🧮 If `π_new == π_ref` on a token, `x=0` → `kld = e^0 − 0 − 1 = 0`. If the new policy made the token
slightly *more* likely than the reference (`log π_new > log π_ref`, say `x = −0.1`):
`kld = e^{-0.1} + 0.1 − 1 = 0.9048 + 0.1 − 1 = 0.0048` — a small **positive** penalty. Notice the naive
estimator would have given `log π_new − log π_ref = +0.1`'s negative, i.e. `−0.1` (negative!), showing
exactly the pathology k3 fixes.

This estimator name in our config is `**low_var_kl`** (`algorithm.kl_penalty: low_var_kl`) — "low
variance KL," precisely the property above. The code also supports `kl`, `abs`, `mse`, `full`
(✓ `core_algos.py:579-597`).

## 7.4 The reference policy in code

`π_ref` is a **frozen** copy of the starting model. ✓ When a worker is built as the reference, all its
gradients are turned off: `fsdp_workers.py:233` `model.requires_grad_(False)` (under `if role == "ref"`).
It is never optimized. Its log-probs are computed **once per step**: ✓ `ray_trainer.py:617`
`ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)`. So every step measures drift
against the *same fixed anchor* — not against the previous step (that's what `old_log_probs` is for, a
different role: `old` = pre-update *this step* for the ratio; `ref` = frozen *initial* model for KL).

⚠️ `**old_log_probs` vs `ref_log_probs` — do not confuse them.** Both are log-probs of the same
responses, but: `old_log_probs` come from the policy *as it was at the start of this step's update*
(used for the PPO **ratio**, Ch. 6); `ref_log_probs` come from the *original* model and never change
(used for the KL **penalty**, this chapter). One controls per-step step-size; the other controls total
drift from the origin.

## 7.5 Two places KL can be applied (and which we use)

EasyR1 can inject the KL penalty in **either** of two spots, switched by `algorithm.use_kl_loss`:

**Mode A — inside the reward** (`use_kl_loss = false`). ✓ `ray_trainer.py:127` (inside
`apply_kl_penalty`):

```python
token_level_rewards = token_level_scores - kl_ctrl.kl_coef * kld
```

KL is subtracted from each token's reward **before** advantages are computed. Consequence: the KL
penalty flows *through* the GRPO group-normalization (Ch. 5) — it changes the per-response rewards, and
thus the group mean/std and the z-scores. The penalty and the task reward are blended into one signal.

**Mode B — as a separate loss term** (`use_kl_loss = true`, **our setting**). ✓ `dp_actor.py:272-281`:

```python
kld = compute_kl(log_probs, ref_log_probs, kl_penalty=self.config.kl_penalty)  # k3, per token
kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)       # mask out padding, average
loss = pg_loss + kl_loss * self.config.kl_coef                                   # add to the gradient
```

KL is computed **after** advantages and added directly to the loss as a regularizer. Consequence: the
advantage signal (Ch. 5) stays **uncontaminated** by KL; the KL acts as a clean, separate gradient that
pulls the policy back toward `π_ref`. This is the more common modern choice and what our config uses,
with `kl_coef = 0.01` (a small weight — the KL is a gentle leash, not a straitjacket).

(`average_loss` with `mode="token"` is just `masked_mean`: sum the per-token KL over real tokens and
divide by the real-token count, ignoring padding — ✓ `core_algos.py:402-403`.)

## 7.6 Fixed vs adaptive KL coefficient

`kl_coef` can be held constant or auto-tuned to hit a target KL:

- `**FixedKLController`** (✓ `core_algos.py:64`): `kl_coef` never changes. **This is what we use** —
constant `0.01` throughout training.
- `**AdaptiveKLController`** (✓ `core_algos.py:47`): nudges `kl_coef` up when measured KL exceeds a
target and down when below, to keep drift near a set point. Useful for long RLHF runs; unnecessary for
our short, well-behaved perception runs.

> **🌐 Generalizes:** *why* we regularize toward a reference (reward hacking, forgetting, mode collapse);
> KL divergence as the measure of distributional drift, its non-negativity and asymmetry; estimating KL
> from samples and why a low-variance non-negative estimator (k3 / `e^x−x−1`) beats the naive
> log-ratio; reward-side vs loss-side KL; fixed vs adaptive coefficients. All standard in RLHF/RLVR.
> **📦 Ours:** `use_kl_loss=true` (loss-side, Mode B), `kl_penalty=low_var_kl`, `kl_coef=0.01`, fixed
> controller; `π_ref` = frozen Qwen3-VL-4B-Instruct; the exact lines in `core_algos.py`/`dp_actor.py`.
> Our `freeze_language_model`/`freeze_vision_tower` conditions don't touch `π_ref` — it's always the full
> frozen init.

---

# Chapter 8 — The full training loop: one step, end to end

Everything in Parts I–II is one line somewhere in `fit()`. Now we read the whole loop with all
ingredients understood. ✓ `ray_trainer.py:561 fit`, with `:585` `while self.global_step < self.training_steps:`.

## 8.1 The anatomy of a single step

```
ONE STEP (ray_trainer.py:585-688):

PHASE 1 — GENERATE  (timer "gen", :591-594)               [the expensive part]
  prepare_rollout_engine()                # hand FSDP weights to vLLM (Part III)
  batch = _make_batch_data(...)           # see 8.2 — produce rollout_batch_size prompts × n responses
  release_rollout_engine()                # give GPUs back to training

PHASE 2 — INGREDIENTS  (:599-618)
  _balance_batch(batch)                   # reorder so each GPU gets ~equal tokens (load balance)
  reward_fn.compute_reward(batch)         # Ch.3: scalar→token-level reward tensor (async, :607)
  compute_log_probs(batch)  → old_log_probs   # Ch.6: π_old, frozen for the ratio (:611)
  compute_ref_log_probs(batch) → ref_log_probs # Ch.7: π_ref, frozen anchor (:617, if KL on)

PHASE 3 — ADVANTAGE  (timer "adv", :626-648)
  if use_kl_loss:  token_level_rewards = token_level_scores     # KL stays out of reward (:640)
  else:            apply_kl_penalty(...)                        # Mode A (:637)
  compute_advantage(batch, adv_estimator="grpo", …)            # Ch.5: GRPO z-score (:643)

PHASE 4 — UPDATE  (timer "update_actor", :659-664)
  update_actor(batch)                      # Ch.6: epochs→mini→micro, clip + KL loss, optimizer steps

THEN  (:666-688)
  if val_freq hit:  _validate()            # generate on held-out set, score (no weight update)
  if save_freq hit: _save_checkpoint()     # write actor weights (+ dataloader state)
  logger.log(metrics)                      # everything from Ch.6/7 diagnostics + reward/length/timing
```

Note the order is exactly the data-dependency order: you can't score before generating, can't compute
advantages before scoring + log-probs, can't update before advantages.

## 8.2 How a batch is actually assembled (`_make_batch_data`)

✓ `ray_trainer.py:466-559`. This sub-loop is subtle and contains two details that matter for GRPO:

1. **Unique prompt ids.** ✓ `:485` `new_batch.non_tensor_batch["uid"] = [uuid4() … ]`. Every prompt
  gets a UUID. This `uid` is the `index` that GRPO groups by (Ch. 5.1) — *not* the row position.
2. **Generate, then replicate by `n`.** ✓ `:497` `generate_sequences(gen_batch)` produces the responses;
  ✓ `:514` `new_batch.repeat(repeat_times=n, interleave=True)` makes `n=5` copies of each prompt row so
   each lines up with one of its 5 responses. **Interleaved** (`p1 p1 p1 p1 p1 p2 …`) so that after
   `_balance_batch` later scrambles row order, the `uid` still correctly reunites each prompt's 5
   responses. *(This is why GRPO groups by `uid`, not position — `_balance_batch` at `:599` deliberately
   breaks position order for GPU load-balancing.)*
3. **Online filtering (optional, Ch. 5.3).** ✓ `:518-540` if `algorithm.online_filtering`, it scores
  responses immediately and **drops whole  groups** whose mean reward is ≤ `filter_low` or ≥
   `filter_high` (the all-wrong / all-right *dead groups*). It then keeps generating until it has
   `rollout_batch_size` surviving prompts (✓ `:543-559`, capped by `max_try_make_batch` to avoid
   infinite loops on bad data).

## 8.3 The roles and the "hybrid engine"

✓ `ray_trainer.py:64-75` defines worker roles (Actor, Rollout, RefPolicy, …). In our setup the actor,
its rollout (vLLM), and the reference all live in **one** colocated worker group — `hybrid_engine`
(✓ `:256` `if self.hybrid_engine:` → `Role.ActorRolloutRef`). "Hybrid" = the *same GPUs* are time-shared
between **training** (FSDP) and **generation** (vLLM), which is why each step explicitly hands weights
back and forth (`prepare_rollout_engine` / `release_rollout_engine`). This weight handoff is the single
trickiest systems piece and gets its own treatment in Part III. There is **no critic worker** for us
(GRPO has no value network, §4.5/§5.2) — ✓ `use_critic` is false, so the `Critic` phases (`:620-624`,
`:650-656`) are skipped entirely.

## 8.4 Reading a step's logs (tie it all back)

Each step logs (✓ `:681-687`): the **reward** breakdown (overall/accuracy/format, Ch. 3.1), **response
length** stats, and the actor diagnostics from §6.7 (`ppo_kl`, `pg_clipfrac_`*, `entropy_loss`,
`grad_norm`, `kl_loss`, `kl_coef`). A healthy GRPO run, in plain terms: **mean reward trends up**;
**accuracy-reward rises** (the part we care about for perception); **entropy falls gently** (not a
cliff); **ppo_kl and clipfrac stay modest**; **kl_loss stays bounded** (the leash holds). When we launch
Stage 1, we'll read these live and map each curve back to the chapter that explains it — that is the
capstone exercise.

## 8.5 Validation and checkpointing

✓ `_validate` (`:392-447`) periodically (`val_freq`) generates on a held-out set with the *evaluation*
sampling settings (`temperature 0.6, n=1`, Ch. 1.4) and logs reward — **no gradient update**. ✓
`_save_checkpoint` (`:308-340`) every `save_freq` writes the actor weights, tracks the best-so-far by
val reward, and prunes old checkpoints to `save_limit`. (Recall our experiment design wants **frequent,
full** checkpoints — these are the knobs that give us the trajectory for the offline analyses in the
proposal.)

> **🌐 Generalizes:** the canonical RL post-training loop — *generate → score → estimate advantage →
> constrained policy update → repeat* — and the strict data-dependency order; the idea of time-sharing
> GPUs between a fast inference engine and a training engine; validation-without-update; checkpoint
> selection by held-out reward.
> **📦 Ours:** `hybrid_engine` with no critic; `adv_estimator="grpo"`; the `uid`+`repeat(interleave)`
> grouping mechanics; `online_filtering` thresholds; `val_freq`/`save_freq`/`save_limit` cadences; the
> exact `fit()` line numbers. Swapping `adv_estimator` (§4.6) or toggling `online_filtering`/critic would
> change only the marked lines.

---

## End of Part II

You can now trace a single training step from raw rollouts to a weight update, and explain every term:

1. The policy generates `n=5` responses per prompt (`_make_batch_data`), each scored into a sparse
  token-level reward (Ch. 3).
2. GRPO turns rewards into per-token advantages via group z-scores (Ch. 5).
3. The update reuses that batch safely: the **importance ratio** corrects for off-policy drift and
  **PPO clipping** keeps each step inside a trust region (Ch. 6).
4. A **KL penalty** against the frozen reference keeps the policy from wandering off or reward-hacking
  (Ch. 7).
5. `fit()` sequences all of this and logs the diagnostics that tell you whether learning is healthy
  (Ch. 8).

**Part III (next, same depth)** — the systems that make this run on 4×GH200:

- FSDP: how a 4B model + optimizer is sharded across GPUs, and `offload`.
- The FSDP↔vLLM weight handoff (`sharding_manager/`) — the "hybrid engine" boundary from §8.3.
- The batch-size hierarchy in full: `rollout_batch_size` vs `global_batch_size` vs
`micro_batch_size_per_device`_*, plus `padding_free`, `dynamic_batching`, `ulysses_size`.
- Our `freeze_language_model` patch (Ch. 7-adjacent): exactly what gradient flow the three conditions
produce, and how `requires_grad` + FSDP `use_orig_params` + the optimizer's param filter interact.

