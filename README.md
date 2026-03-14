# nanoGPT-Understanding
Replicating Andrej Karpathy's nanoGPT for learning purposes
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a4a7a,100:00d4aa&height=180&section=header&text=nanoGPT%20%E2%80%94%20From%20Scratch&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=My%20notes%20as%20a%20biotech%20undergrad%20learning%20transformers&descAlignY=56&descSize=16" width="100%"/>

</div>

# nanoGPT — My Understanding from Scratch

> Replicated Andrej Karpathy's nanoGPT as part of my deep learning journey.
> I come from a **Biotechnology + Data Science** background — not a CS degree.
> This README is written the way I wish someone had explained it to me.

---

## Why I did this

I kept seeing "self-attention", "residual connections", "causal masking" in every ML paper I read for my bioinformatics work. I used HuggingFace pipelines and fine-tuned BioBERT — but I realised I didn't actually understand what was happening inside. I could use transformers but couldn't explain them.

So I went through Karpathy's nanoGPT line by line, broke it down, rebuilt it, and wrote down everything that confused me and eventually clicked. This README is that document.

---

## What the model does in one sentence

It reads Shakespeare text **one character at a time**, learns the statistical patterns of how characters follow each other, and generates new text that sounds Shakespearean.

The same fundamental idea powers ChatGPT — GPT-4 just predicts the next *token* over hundreds of billions of parameters. This model predicts the next *character* over ~200K parameters. The architecture is identical.

---

## The Data Pipeline
```python
chars = sorted(list(set(text)))         # every unique character in Shakespeare
stoi = { ch:i for i,ch in enumerate(chars) }   # 'A' → 0, 'B' → 1 ...
encode = lambda s: [stoi[c] for c in s]         # "hi" → [35, 46]
decode = lambda l: ''.join([itos[i] for i in l]) # [35, 46] → "hi"
```

The vocabulary is just every character that appears — 65 total (a–z, A–Z, punctuation, space). Each character maps to an integer. The model never sees actual letters — only numbers.

Training uses 90% of the text, validation uses the remaining 10%.

### Batches
```python
batch_size = 16   # 16 sequences processed in parallel
block_size = 32   # each sequence is 32 characters long
```

Instead of feeding one character at a time, the model trains on 16 random 32-character chunks simultaneously. For each chunk, the **target** (`y`) is the same chunk shifted one position forward — the model's job is to predict the next character at every single position.

---

## The Part That Confused Me Most — the `tril` Mask

This is the single most important line to understand. Let me explain it slowly.

When the model trains, it sees a sequence of 32 characters at once:
```
T  o     b  e     o  r     n  o  t     t  o     b  e
0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
```

The rule is: **a character can only look backwards, never forwards.**

- Position 0 (`T`) can only see position 0 when predicting position 1
- Position 3 (`b`) can see positions 0, 1, 2, 3 when predicting position 4
- Position 9 (`n`) can see positions 0–9 when predicting position 10

This makes sense because at **generation time**, you don't know what comes next — you're building the sequence one character at a time. If the model was allowed to see future characters during training, it would just cheat by copying them. It would learn nothing useful and fail completely when generating.

The `tril` mask (lower triangular matrix) enforces this rule:
```python
tril = torch.tril(torch.ones(T, T))

# For T = 5, this looks like:
# 1 0 0 0 0
# 1 1 0 0 0
# 1 1 1 0 0
# 1 1 1 1 0
# 1 1 1 1 1
#
# Row i = position i
# 1 = "I am allowed to look at this position"
# 0 = "This is in the future — blocked"
```

In the attention code, the `0` positions get set to `-inf` before softmax:
```python
wei = wei.masked_fill(tril[:T, :T] == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
```

When softmax sees `-inf`, it outputs exactly `0` probability for that position. The model literally cannot attend to future characters — they are completely invisible. This is called **causal masking** or **autoregressive masking**.

> **Biological analogy that helped me:** Think of reading a DNA strand 5'→3'. Each base can only "see" the bases that came before it in the synthesis direction. It has zero information about what bases will be added downstream. The `tril` mask is exactly this constraint — written as a matrix.

---

## How Attention Works — Q, K, V in Plain Terms

Before I understood attention I thought of it as magic. It is actually asking three questions for every character:

- **Query (Q):** *"What kind of information am I looking for?"*
- **Key (K):** *"What kind of information do I contain?"*
- **Value (V):** *"What will I actually share if I'm selected?"*

For every position, you measure how well its Query matches every other position's Key — this gives a score. Softmax converts scores into probabilities (attention weights). Then you take a weighted average of all Values using those weights.
```python
k = self.key(x)    # (B, T, head_size) — what each position contains
q = self.query(x)  # (B, T, head_size) — what each position is looking for

wei = q @ k.transpose(-2,-1) * C**-0.5  # scores: how well Q matches K
wei = wei.masked_fill(tril == 0, float('-inf'))  # block the future
wei = F.softmax(wei, dim=-1)             # turn scores into probabilities

v = self.value(x)  # (B, T, head_size) — what to share
out = wei @ v      # weighted average of values
```

The `* C**-0.5` scaling (dividing by √head_size) prevents scores from getting too large before softmax. Very large values push softmax into near-zero gradient regions, which kills learning. Scaling keeps it stable.

### Why Multiple Heads?

One attention head might learn to focus on the immediately preceding character. Another might track punctuation. Another might pick up on repeated words. Running 4 heads in parallel and concatenating their outputs lets the model learn several types of relationships simultaneously — each head specialises in something different.
```python
out = torch.cat([h(x) for h in self.heads], dim=-1)  # concat all heads
out = self.proj(out)  # project back to original embedding dimension
```

---

## Feedforward Neural Network — What It Does and Why It's There

After attention, every position passes through a small feedforward network **independently**:
```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),   # expand: 64 → 256
    nn.ReLU(),                         # non-linearity
    nn.Linear(4 * n_embd, n_embd),   # compress: 256 → 64
    nn.Dropout(dropout),
)
```

A natural question is: **why do we need this if we already have attention?**

Attention is purely about **communication** — it lets positions look at each other and gather information from across the sequence. But attention itself is just a weighted average — it is linear. It cannot learn complex non-linear relationships on its own.

The feedforward network is about **computation** — once a position has gathered information via attention, the feedforward layer is where it actually *thinks* about what it collected. The ReLU non-linearity is what gives it the ability to learn complex patterns.

The 4× expansion (`64 → 256 → 64`) gives the network more room to represent intermediate computations before compressing back. This ratio of 4 was found empirically in the original transformer paper and has been kept ever since.

> **Simple analogy:** Attention is like a meeting where everyone shares their notes. The feedforward network is each person going back to their desk and actually *processing* what they heard before the next meeting.

---

## LayerNorm vs BatchNorm — The Difference That Actually Matters

This tripped me up badly. Both normalize activations — but they do it across different dimensions, and that difference is crucial.

### BatchNorm — normalizes across the batch
```
Input shape: (Batch, Features)

BatchNorm asks: "For feature #5, what is the mean and std
                 across ALL samples in this batch?"

Normalizes DOWN the batch dimension ↓
```
```
Batch:  [sample_1_feat5, sample_2_feat5, sample_3_feat5 ...]
         ←————————— compute mean & std across these ————————→
```

**Problem for language models:** The statistics depend on other samples in the batch. During inference, if you generate one character at a time (batch size = 1), BatchNorm behaves completely differently than during training. You have to track running averages of mean/std across all training batches, which introduces instability. It also doesn't work well with variable-length sequences.

### LayerNorm — normalizes across the features
```
Input shape: (Batch, Time, Features)

LayerNorm asks: "For THIS specific token at THIS specific position,
                 what is the mean and std across all its features?"

Normalizes ACROSS the feature dimension →
```
```
Token at position 3:  [feat_1, feat_2, feat_3, ... feat_64]
                       ←—— compute mean & std across these ——→
```

**Why this works better for transformers:**
- Each token is normalized independently — batch size doesn't matter
- Works identically during training and inference
- No need to track running statistics
- Stable with variable-length sequences
```python
self.ln1 = nn.LayerNorm(n_embd)  # normalizes each token's 64 features independently
self.ln2 = nn.LayerNorm(n_embd)

# Used as Pre-LN (normalize BEFORE attention/feedforward)
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

This is called **Pre-LN** (normalise before the sub-layer). The original 2017 transformer paper used Post-LN (normalise after), but Pre-LN was found to train more stably — especially for deeper networks.

> **One-line summary:**
> BatchNorm = normalise across samples (bad for text, great for images/CNNs)
> LayerNorm = normalise across features (great for text, standard in all transformers)

---

## Residual Connections — Why `x = x + layer(x)`

When I first saw this pattern I thought it was a bug:
```python
x = x + self.sa(self.ln1(x))   # why add x to itself??
x = x + self.ffwd(self.ln2(x))
```

The reason is **vanishing gradients**. In a deep network, gradients flow backwards through every layer during training. Without residual connections, gradients shrink exponentially — by the time they reach the early layers, they are essentially zero. Those layers stop learning.

The residual (skip) connection gives gradients a **direct highway** back to early layers:
```
Input x ──────────────────────────────────────────► (+) ► output
         └──► LayerNorm ──► Attention/FFN ──► ─────────►
```

Even if the attention or feedforward layer learns nothing useful early in training, the original `x` still flows through unchanged. The model always falls back to what it already knew. Each layer only needs to learn the *difference* (the residual) from the current representation — which is a much easier optimisation problem.

This is why very deep networks (GPT-3 has 96 layers) are even trainable at all.

---

## The Full Forward Pass Step by Step

Here is what happens to the character `'T'` (position 0) as it flows through the model:
```
Step 1 │ 'T' → integer index (e.g. 39)
       │
Step 2 │ token_embedding_table[39] → vector of size 64
       │ "What is this character semantically?"
       │
Step 3 │ position_embedding_table[0] → vector of size 64
       │ "Where does it sit in the sequence?"
       │
Step 4 │ x = tok_emb + pos_emb
       │ "Combined: what it is + where it is"
       │
Step 5 │ Block 1: LayerNorm → MultiHeadAttention → residual add
       │          LayerNorm → FeedForward → residual add
Step 6 │ Block 2: (same)
Step 7 │ Block 3: (same)
Step 8 │ Block 4: (same)
       │
Step 9 │ Final LayerNorm
       │
Step10 │ Linear layer → 65 scores (one per vocabulary character)
       │
Step11 │ Cross-entropy loss vs. actual next character
       │ "How wrong were we? Update weights accordingly."
```

After 5000 training steps, the score for the correct next character becomes reliably higher than all others.

---

## Generating Text — the Autoregressive Loop
```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -block_size:]         # only look at last 32 chars
    logits, _ = model(idx_cond)             # get scores for next character
    logits = logits[:, -1, :]               # focus on the LAST position only
    probs = F.softmax(logits, dim=-1)       # convert scores → probabilities
    idx_next = torch.multinomial(probs, 1)  # SAMPLE from distribution
    idx = torch.cat((idx, idx_next), dim=1) # append and repeat
```

One thing I noticed: it **samples** from the distribution rather than always picking the highest-probability character (argmax). Always picking the top character makes output repetitive and boring. Sampling introduces controlled randomness — this is the same reason ChatGPT has a temperature parameter. Higher temperature = more random. Lower = more conservative.

---

## Summary of Key Concepts

| Concept | What it does | Why it matters |
|--------|-------------|---------------|
| `tril` mask | Blocks future positions in attention | Makes training honest — model can't cheat |
| Q, K, V attention | Lets positions look at each other | Captures long-range dependencies in sequence |
| Multi-head attention | Runs attention in parallel with different projections | Each head learns different relationship types |
| Feedforward network | Processes each token independently after attention | Adds non-linear computation after linear attention |
| LayerNorm | Normalises each token's features independently | Stable training regardless of batch size |
| Residual connections | Adds input back after each sub-layer | Solves vanishing gradient in deep networks |
| Autoregressive generation | Predicts one token, appends, repeats | How all GPT-style models generate text |

---

## What I Want to Explore Next

- [ ] Visualise the attention weights — what does each head actually focus on?
- [ ] Add temperature scaling to control generation creativity
- [ ] Train on a biological sequence dataset (promoter sequences / protein domains)
- [ ] Understand how BPE tokenization differs from character-level

---

## References

- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [Attention Is All You Need — Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Let's build GPT — Karpathy YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00d4aa,100:1a4a7a&height=100&section=footer" width="100%"/>
</div>
