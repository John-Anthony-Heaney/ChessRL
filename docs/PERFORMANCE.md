# PERFORMANCE — where ChessRL's throughput actually goes

**Status of this document.** The method, the harness and the *structural* findings
(sections 2, 3a, 3c-i, 3d, 3e, 4) are done and reproducible. The *absolute*
throughput numbers in it are **not trustworthy** and are labelled where they
appear, for the reason in section 1. Re-run the commands in section 2 on a
mains-powered machine and the absolutes will be right; the ratios should hold.

---

## 1. READ THIS FIRST: the machine was throttled

Every measurement below was taken on 2026-09-11 between 23:19 and 23:30 local
time on:

| | |
|---|---|
| Machine | Apple M3, 4 performance + 4 efficiency cores, 8 GB |
| OS | Darwin 23.6.0 |
| Compiler | Apple `cc`, `-O3 -std=c11 -funroll-loops -fno-math-errno -ffp-contract=fast` |
| `-mcpu=native` | supported and applied (`ARCH` resolves to `-mcpu=native`; verified with `cc -mcpu=native -E -x c /dev/null`) |
| Compiler version | Apple clang 16.0.0 (clang-1600.0.26.3) |
| Source snapshot | `git rev-parse --short HEAD` = `37ff263`; `md5 src/net.c` = `683151af06688f0cb3b76650daba3c72`, `src/mcts.c` = `818c935e7b045050e704b5c45de258af`, `src/chess.c` = `956b6ea6d3103ff4ea399fc08987b89a` |

**The machine was running on battery at 3% charge with a macOS low-battery
warning active, and was therefore being clock-throttled.** Measured, not assumed:

```
$ pmset -g ps
Now drawing from 'Battery Power'
 -InternalBattery-0   3%; discharging; 0:11 remaining present: true
	Battery Warning: Early
```

A single dependent `add` chain — one instruction per cycle by construction, so
iterations/second *is* the core clock — measured **1.43 to 1.91 GHz** across the
session, against 4.05 GHz nominal for an M3 performance core. Source:
`tools/` scratch probe, reproduced in section 2.6.

Two consequences, both important:

1. **Absolute rates are roughly 2x low** and drifted *during* the session. The
   thread-scaling sweep in section 3b swung by 4x between adjacent runs as the
   battery collapsed (4 threads: 57.2 games/s; 6 threads: 14.8 games/s; 8
   threads: 27.5 games/s). Those numbers are noise, not scaling, and are
   presented only to show why they were discarded.
2. **Ratios measured inside one run are still sound**, because both sides of the
   ratio were throttled identically. The phase split (section 3a) and the
   generation split (section 3e) are single-run ratios and are the load-bearing
   results here.

The brief's starting figures — ~80 games/sec and ~720,000 evaluations/second —
**were not reproduced**, and I cannot tell from this session whether that is the
throttle, a different configuration, or drift in `src/` (another agent was
editing `src/net.c` and `src/mcts.c` *while these runs were in flight*;
`src/net.c` changed on disk at 23:21:54, mid-session). The `evals/move` figure
is stable at **63-65** across every run, so the *shape* of the workload is
settled even though its speed is not.

Other load: `ps -A -o %cpu` was checked before every quoted run. The runs in
sections 3a, 3c and 3d ran with the machine otherwise idle (background total
< 10% of one core). Earlier attempts at 23:19 were discarded because two other
agents' jobs (`build/profile` at 270%, `build/chessrl` at 102%) were running.

---

## 2. How to reproduce

### 2.1 Build

```sh
make -j8 bench-throughput
```

builds two binaries from one source and runs the first:

* `build/bench_throughput` — the headline numbers. **No timers are compiled in
  at all**: the probe macros expand to `((void)0)`, so there is no branch, no
  counter and no clock read anywhere on the measured path.
* `build/bench_throughput_probe` — the same source with `-DBENCH_PROBE`, linked
  against a *second* copy of `src/mcts.c` compiled into `build/probe/mcts.o`
  with `-include tools/bench_probe.h`.

`src/` is not modified by any of this. The instrumentation is a compiler flag:
`tools/bench_probe.h` defines function-like macros that rename `nn_eval`,
`gen_legal` and friends **at their call sites inside that one translation
unit**, so each becomes a static inline that reads `clock_gettime(CLOCK_MONOTONIC)`,
calls the real function, and accumulates into a thread-local counter. The
header includes `chess.h` and `net.h` first, so the real declarations are parsed
before the macros exist and are not themselves mangled.

### 2.2 The runs quoted in this document

```sh
# section 3a — the phase split (this is the important one)
build/bench_throughput_probe --mode gen --threads 4 --games 48 --sims 160 \
                             --steps 40 --agents 32

# section 3e — generation split at 8 threads, az.c's real step count
build/bench_throughput --mode gen --threads 8 --games 256 --sims 160 \
                       --cap-frac 0.25 --steps 200 --batch 256 --agents 32

# section 3b — thread scaling (DISCARDED this session; re-run on mains power)
for T in 1 2 3 4 6 8; do
  build/bench_throughput --mode selfplay --threads $T --games $((40*T)) \
                         --sims 160 --agents 32 --quiet --csv
done

# section 3c — QoS / core-affinity A/B
for Q in off ui utility background; do
  build/bench_throughput --mode selfplay --threads 8 --games 320 \
                         --sims 160 --qos $Q --csv
done

# section 3d — primitive costs and the pure-forward-pass scaling curve
build/bench_throughput --mode micro --secs 0.6

# section 3c-i — what the evaluation cache and subtree reuse are worth
build/bench_throughput --mode selfplay --threads 4 --games 64 --sims 160 --agents 1
build/bench_throughput --mode selfplay --threads 4 --games 64 --sims 160 --agents 32
build/bench_throughput --mode selfplay --threads 4 --games 64 --sims 160 --agents 1 \
                       --no-cache --no-reuse
```

Use **at least 40 games per thread**; the short runs in this session were the
proximate cause of the unusable scaling table. `build/bench_throughput --help`
lists every knob.

### 2.3 Before you trust a number

```sh
pmset -g ps | head -2                 # must say 'AC Power'
ps -A -o %cpu,comm -r | head -5       # nothing else above a few percent
```

and check the measured clock (section 2.6) is near 4.0 GHz, not 1.9.

### 2.4 What the harness is

`tools/bench_throughput.c` is a **replica** of the self-play and learning loop in
`src/az.c`, not a wrapper around it: `az.c`'s worker pool, replay buffer and
learning job are all `static`, so there is no seam to hook. Every primitive it
calls — `mcts_search`, `gen_legal`, `nn_eval`, `nn_backward`, `adam_step` — is
the real one, so the per-call costs *are* the trainer's costs; only the loop
around them is re-implemented. It matches `az_play_one()` on start-position
policy, playout-cap randomisation, resignation, temperature schedule and
recording, and `az_learn_job()` on batch construction, per-agent gradient
ownership and optimiser order.

**This is the harness's one real caveat**: if `az.c`'s loop changes, the replica
must be re-checked against it. It is deliberately a small file for that reason.

### 2.5 Timed vs counted, and why

`clock_gettime(CLOCK_MONOTONIC)` measured **19.8 to 27.0 ns/call** on this
machine (the harness prints it at startup). That is a rounding error against
`nn_eval` (tens of microseconds) and a catastrophe against `make_move` (tens of
nanoseconds). So:

* **timed** — `nn_eval`, `nn_logits`, `nn_features`, `gen_legal`, `nn_backward`.
  Two clock reads each; the harness subtracts `2 x clock_cost x calls` from every
  bucket before reporting.
* **counted only** — `make_move`, `unmake_move`, `in_check`,
  `insufficient_material`, `nn_move_key`, `softmax_t`. One thread-local
  increment, which is free. Their cost lands inside "bookkeeping" and is
  attributed by multiplying the exact call count by the `--mode micro` cost.
  `-DBENCH_PROBE_FINE` times them directly as a cross-check, at the price of
  distorting the total.

MCTS bookkeeping — selection, backup, node allocation, the evaluation cache, the
subtree-reuse checks, the weight stamp — is **by subtraction**:
`mcts_search - nn_eval - nn_logits - nn_features - gen_legal`.

### 2.6 The clock probe

Not part of the harness; a five-line check worth keeping to hand.

```c
uint64_t x = 1; long N = 2000000000L;
/* one dependent integer add per iteration: exactly 1 instruction per cycle */
for (long i = 0; i < N; i++) __asm__ volatile("add %0, %0, #1" : "+r"(x));
/* iterations per second == core clock */
```

---

## 3. The questions, answered

### 3a. What fraction of self-play is the network? — **92%, and that decides everything**

From `build/bench_throughput_probe --mode gen --threads 4 --games 48 --sims 160
--steps 40 --agents 32`. Thread-seconds, summed over 4 threads, corrected for
probe overhead:

| phase | calls | thread-sec | ns/call | share of `mcts_search` |
|---|---:|---:|---:|---:|
| `nn_eval` | 497,607 | 15.710 | 31,572 | **86.4 %** |
| `nn_logits` | 499,311 | 1.090 | 2,184 | **6.0 %** |
| `gen_legal` | 499,512 | 0.220 | 441 | 1.2 % |
| `nn_features` | 497,607 | 0.012 | 24 | 0.07 % |
| MCTS bookkeeping (by subtraction) | — | 1.147 | — | 6.3 % |
| **`mcts_search` total** | 7,840 | **18.179** | 2,318,766 | 100 % |
| driver, outside `mcts_search` | 7,840 | 0.014 | 1,746 | 0.08 % |

> The `mcts_search` total exceeds `4 x self-play wall` because the probe counters
> also accumulate during the harness's warmup round, which is not included in the
> reported wall clock. This inflates the "% wall" column the tool prints, and does
> **not** affect the shares above, which are all ratios within `mcts_search`.

**Network (`nn_eval` + `nn_logits`) = 92.4 % of self-play search time.**
Move generation is 1.2 %. Feature encoding is 0.07 %. Everything MCTS itself
does — selection, backup, allocation, cache, reuse verification, and the 13.0
million `nn_move_key` calls it makes — is 6.3 %. The driver loop outside the
search is 0.08 % and can be ignored entirely.

Folding in the learning phase (section 3e: 91.35 % self-play / 8.65 % learning
by wall clock, and 25.2 % of the learning phase is network math):

> **The network is ~86.6 % of a whole generation.**

Amdahl, with 86.6 % network:

| network speedup | generation speedup |
|---|---|
| 2x | 1.76x |
| 3x | 2.31x |
| 5x | **3.26x** |
| 10x | 4.65x |
| infinite | 7.5x (the ceiling) |

**We are firmly in the world where kernels matter enormously.** A 5x faster
network is worth 3.3x overall. Nothing else in this document is worth more than
a few percent. Section 4 is ordered accordingly.

### 3b. Thread scaling — **NOT MEASURED; the data is unusable**

| threads | games/s | speedup |
|---|---:|---:|
| 1 | 16.53 | 1.00x |
| 2 | 35.79 | 2.17x |
| 3 | 50.39 | 3.05x |
| 4 | 57.21 | 3.46x |
| 6 | 14.84 | 0.90x |
| 8 | 27.54 | 1.67x |

**Discard this table.** 2 threads showing 2.17x is superlinear, which is
impossible and proves the clock was ramping *during* the sweep; 6 threads below
1 thread proves it collapsed. The runs were also far too short (0.56-4.0 s).
Re-run with the loop in section 2.2 on mains power, 40 games per thread minimum.

What *can* be said, and it is not nothing: the two independent probes in
section 3d both show a knee at 4 threads with the same shape, which is what the
4P+4E topology predicts. Expect the real curve to be near-linear to 4 and to
gain roughly 25-35 % more from threads 5-8, not 100 %.

### 3c. Are the efficiency cores helping or hurting? — **inconclusive on QoS; yes on self-balancing**

**QoS had no discriminating power under the battery throttle.** The measured
core clock under each class:

| requested class | effective class | measured clock |
|---|---|---:|
| (inherited) | 33 = `USER_INTERACTIVE` | 1.91 GHz |
| `QOS_CLASS_BACKGROUND` | 9 | 2.13 GHz |
| `QOS_CLASS_UTILITY` | 17 | 1.90 GHz |
| `QOS_CLASS_DEFAULT` | 21 | 1.89 GHz |
| `QOS_CLASS_USER_INITIATED` | 25 | 1.87 GHz |
| `QOS_CLASS_USER_INTERACTIVE` | 33 | 1.88 GHz |

Two things are worth keeping from this even so:

1. **The process already inherits `USER_INTERACTIVE` (class 33)** from the shell.
   So `--qos ui` is a no-op for the default configuration, and any P-core-affinity
   experiment has to be framed as *"does dropping to `UTILITY` hurt?"* rather
   than *"does raising to `USER_INTERACTIVE` help?"*. The harness supports both
   directions (`--qos utility`, `--qos background`).
2. Under throttle, `BACKGROUND` was *fastest*. That is noise, and it is the
   clearest single indication that nothing in this table means anything yet.

`build/bench_throughput --mode selfplay --threads 8 --qos {off,ui,utility}` is
wired and ready; it needs a mains-powered machine, not more code.

**Do slow threads self-balance? Yes — verified.** At 8 threads over 256 games:

```
    thread 0   29 games   busy 9.914 s ( 99.8% of wall)
    thread 1   37 games   busy 9.851 s ( 99.2% of wall)
    thread 2   26 games   busy 9.823 s ( 98.9% of wall)
    thread 3   30 games   busy 9.843 s ( 99.1% of wall)
    thread 4   34 games   busy 9.924 s ( 99.9% of wall)
    thread 5   33 games   busy 9.854 s ( 99.2% of wall)
    thread 6   35 games   busy 9.931 s (100.0% of wall)
    thread 7   32 games   busy 9.867 s ( 99.4% of wall)
    spread: min 26  max 37  ratio 1.42x
```

The 1.42x spread in *game counts* is the expected consequence of the shared
atomic index plus variable game length — it is not imbalance. The metric that
matters is **busy time, which is 98.9-100 % for every thread**. No thread
starved and no thread waited. The shared-index work distribution is working
exactly as intended and needs no change.

### 3c-i. What the evaluation cache and subtree reuse are actually worth — **~9 %, and self-play throws it away**

`az.c` pairs **two different agents** in every self-play game. `mcts_search()`
fingerprints `(trunk, head)` on entry and drops both the evaluation cache and the
standing tree whenever that stamp changes — so with different heads on the two
sides, **both are invalidated on every single move**. Measured at 4 threads,
sims 160:

| configuration | evals/move | eval-cache hits | subtree reuse |
|---|---:|---:|---:|
| 1 agent (same head both sides) | **58.90** | 1.8 % | **98.5 %** |
| 32 agents (as `az.c` plays) | **62.03** | 0.4 % | **0.0 %** |
| 1 agent, `--no-cache --no-reuse` | **64.81** | — | — |

So: subtree reuse plus the cache are worth **9.1 % of evaluations** when they
work (64.81 → 58.90), and the two-agent pairing forfeits most of that, costing
**5.3 %** (58.90 → 62.03).

**This is real, and it is not worth doing first.** 5 % of evaluations against a
term that is 92 % of the time is 5 % overall, versus 226 % from a 5x kernel. It
also is not free: restoring *tree* reuse changes search order and therefore
changes results, which the brief forbids without gating. Restoring the *cache*
alone (key it on the head, or give each side its own `Mcts`) is bit-exact and
safe — a cache hit is defined to be what `nn_eval` would have returned — but
it is worth only the ~2 % the cache column shows. See section 4, item 4.

### 3d. Is it memory-bandwidth bound? — **No. It is core-heterogeneity bound, plus a slow kernel.**

Primitive costs, single thread, `--mode micro` (throttled; ratios valid):

| primitive | ns/call |
|---|---:|
| `nn_eval` | 19,090 |
| `nn_logits` (35 moves) | 1,753 |
| `nn_backward` (35 moves) | 20,224 |
| `gen_legal` | 139 |
| `make_move` + `unmake_move` | 35 |
| `nn_features` | 59 |

`nn_eval` at 19.1 µs for **27,136 MACs** is **2.84 GFLOP/s**. The `W1` 128x128
`nn_matvec` is 16,384 of those 27,136 MACs (60 %), so that single matrix-vector
product is about two thirds of the entire forward pass. (A standalone copy of
the kernel measured 12.4 µs, but it was compiled *without* the project's
`-ffp-contract=fast -mcpu=native`, so treat it as indicative only; the
shipped-build figure is the 2.84 GFLOP/s above, measured against `build/net.o`.)

**The kernel is not failing to vectorise.** Compiling `src/net.c` with the exact
project flags emits **475 `fmla`** and **1,660 `.4s`** instructions — it is
using 128-bit NEON fused multiply-add throughout:

```sh
cc -O3 -std=c11 -D_DARWIN_C_SOURCE -funroll-loops -fno-math-errno \
   -ffp-contract=fast -mcpu=native -Isrc -S src/net.c -o /tmp/net.s
grep -c 'fmla' /tmp/net.s     # 475
grep -c '\.4s'  /tmp/net.s     # 1660
```

So the 2.84 GFLOP/s against a 19 GFLOP/s NEON FMA rate is **not** a missing
vectorisation. See section 4 item 1 for what it is instead.

Now the arithmetic-intensity argument. One `nn_eval` touches:

| tensor | bytes |
|---|---:|
| 35 gathered rows of `W0` (35 x 128 x 4) | 17.9 KB |
| `W1` (128 x 128 x 4) | 65.5 KB |
| 35 gathered rows of `W0v` | 4.5 KB |
| `Wvh`, `Wp` | 20.5 KB |
| **total weights touched** | **~108 KB** |

108 KB for 27,136 MACs is **0.25 MAC/byte** — very low intensity, which is
exactly why the matrix-*vector* shape is the problem. But low intensity only
becomes a *bandwidth* limit if the bytes come from DRAM, and they do not: the
trunk is 560 KB, read-only, and shared by every thread, so all 8 threads' working
sets coalesce into one 560 KB footprint that sits comfortably in the M3's shared
L2. At the 8-thread rate, total weight traffic would be 190,678 x 108 KB =
**20.7 GB/s if nothing were cached at all** — and it is nearly all cached.

The decisive evidence that bandwidth is not the limit is that a workload with
**zero** memory traffic shows the same knee. Two scaling curves, same machine,
same session:

| threads | `nn_eval` evals/s | per thread | pure NEON FMA GFLOP/s | per thread |
|---|---:|---:|---:|---:|
| 1 | 57,157 | 57,157 | 18.99 | 18.99 |
| 2 | 106,848 | 53,424 | 39.17 | 19.58 |
| 3 | 131,644 | 43,881 | — | — |
| 4 | 147,652 | 36,913 | 70.94 | 17.73 |
| 6 | 189,256 | 31,543 | 98.79 | 16.47 |
| 8 | 190,678 | 23,835 | 112.61 | 14.08 |

The right-hand pair is eight register-resident `vfmaq_f32` chains — no loads, no
stores, no cache, no memory system at all — and it still loses **26 %** per
thread from 1 to 8 threads and flattens after 4. That loss is the E-cores and
nothing else.

`nn_eval` loses **58 %** per thread over the same range. Dividing out the
core-heterogeneity component (0.42 / 0.74) leaves roughly **44 % unexplained**,
which is the honest upper bound on everything memory-related: shared-L2
pressure, the per-thread 322 KB node pool, the per-thread 295 KB evaluation
cache. It is an upper bound and not a measurement, because the two curves ran
minutes apart under a drifting throttle. **Re-measure both back to back before
anyone optimises for memory.**

On the specific question asked — read-only shared trunk versus written
per-thread gradient buffers — they are in **different phases** and do not
compete. Self-play writes only the 2.7 KB `Fwd` struct per thread; the 560 KB
`TrunkGrad` per thread is written only during the learning phase, which is 15 %
of a generation and is a separate barrier-synchronised section. There is no
overlap to contend for.

### 3e. What does the learning step cost, and does it parallelise? — **~15 % of a generation; yes, but three quarters of it is not network math**

Measured two ways, at different thread counts, agreeing closely.

| | 4 threads | 8 threads |
|---|---:|---:|
| ms per optimiser step | 9.93 | 4.53 |
| ms per self-play game | 87.5 | 38.8 |
| positions/s through the learner | 25,772 | 56,540 |

Normalised to `az.c`'s actual defaults — `games_per_agent 4 x n_agents 32` = 128
games and `steps_per_gen 200`:

| | 4 threads | 8 threads |
|---|---:|---:|
| self-play | 11.2 s | 4.97 s |
| learning | 1.99 s | 0.91 s |
| **learning share of the generation** | **15.1 %** | **15.4 %** |

> The raw `--mode gen` runs report 8.65 % and 8.36 %, because the harness's
> default games:steps ratio is not `az.c`'s. Use the normalised 15 %.

**It parallelises well at the level of per-position gradients** — every thread
reported 99.8-99.9 % busy during the learning wall clock, so the barriers and
the serial reduction are not currently costing much wall time.

**But three quarters of it is not network math.** Of 1.586 learning
thread-seconds:

| | thread-sec | share |
|---|---:|---:|
| `nn_eval` + `nn_logits` + `softmax_t` | 0.225 | 14.2 % |
| `nn_backward` | 0.174 | 11.0 % |
| everything else | **1.187** | **74.8 %** |

"Everything else" is optimiser and gradient bookkeeping: `grad_zero` over 143,328
floats per thread per step, the serial reduction of 4 (or 8) trunk gradients on
thread 0, the trunk `adam_step` over 143,328 floats, and the per-agent head
`adam_step`s. **All of it is O(trunk size) per step and none of it scales with
batch size** — at batch 256 across 8 threads that is 32 positions per thread of
real work against a fixed 143,328-float sweep, several times over.

That is the one structural inefficiency outside the kernel worth naming. It is
still only ~11 % of a generation (75 % of 15 %), so see section 4, item 3.

---

## 4. What to change, in priority order

### 1. Make `nn_eval` faster without changing what it computes — **worth up to 3.3x**

This is 86 % of a generation and everything else is rounding error. `nn_eval`
runs at **2.84 GFLOP/s** against **19 GFLOP/s** from a trivially-vectorised NEON
FMA loop on the same machine in the same session — **15 % of what this core
demonstrably does** — and it is *already* emitting NEON FMAs (section 3d).

**The most likely cause is reduction latency, not width.** `nn_matvec` uses four
scalar accumulators specifically so clang's SLP vectoriser folds them into one
NEON vector accumulator — and `src/net.c`'s own header says so. But that fold
produces exactly **one dependent chain of 32 vector FMAs per output row**:

```c
for (int i = 0; i < nin; i += 4) {        /* nin = 128 -> 32 iterations */
    s0 += w[i+0] * x[i+0];  s1 += w[i+1] * x[i+1];
    s2 += w[i+2] * x[i+2];  s3 += w[i+3] * x[i+3];
}   /* -> one v0.4s accumulator, each fmla waiting on the previous one */
```

FMA latency on an M-series core is ~4 cycles, so a single chain retires one FMA
per 4 cycles instead of the ~4 per cycle the core can issue — a ~16x shortfall
that matches the observed gap. Predicted cost at 1.9 GHz: 32 FMAs x 4 cycles x
128 rows = 16,384 cycles = **8.6 µs**, against ~12 µs measured for that matvec.
**Verify this before acting on it** — count cycles with `--mode micro` while
varying only the accumulator count.

The fix is more *independent* accumulators (4 vector accumulators = 16 partial
sums), which shortens the dependency chain 4x. Three routes, and the contract
distinction matters:

* **More accumulators — the likely 3-4x, but NOT bit-exact.** Regrouping which
  elements land in which partial sum changes the association order of a float
  reduction, so the last bits move. The finite-difference gradient check in
  `tests/test_net.c` will almost certainly still pass; the MCTS determinism
  tests compare exact visit counts and may not. **Run both before and after**,
  and if the determinism tests move, treat it as a gated change rather than a
  free one. Expected: 2-4x on the kernel, **1.8-2.9x overall**.
* **Bit-exact alternative.** Software-pipelining the *same* grouping — keeping
  lane k accumulating exactly the elements it accumulates today, in the same
  order — preserves results exactly but cannot break the chain, so it buys much
  less. Worth trying first only to establish a safe baseline.
* **Accelerate `cblas_sgemv`.** Reassociates too, with the same caveat, plus it
  needs the portable scalar fallback the brief requires. Measure it against the
  hand-written NEON path rather than assuming the vendor kernel wins at n=128.

### 2. Batch the evaluations — **the only path past ~3x, and it changes results**

A matrix-*vector* product loads each weight once and uses it once; that is why
2.64 GFLOP/s is not a bug in the kernel so much as a property of the shape.
Batching B leaves into a matrix-*matrix* product amortises every weight load over
B and is the only way to reach the machine's width.

**This changes what is computed.** MCTS is sequential by construction — it needs
each leaf value before choosing the next path — so batching requires virtual
loss or a similar mechanism, which changes the search order and therefore the
visit distribution. That is a behaviour change, not an optimisation, and it must
be gated behind a flag and validated against the MCTS determinism tests with the
flag off. Sequence it *after* item 1, so the cheap bit-exact win is banked first
and the expensive semantic change is measured against a fast baseline rather
than a slow one.

`src/net.c`'s header already refers to an `nn_eval_batch()` and an Accelerate
path, so this may already be in flight in the parallel `src/` work.

### 3. Stop sweeping the whole trunk gradient per optimiser step — **worth ~5 % of a generation**

74.8 % of the learning phase (≈ 11 % of a generation) is `grad_zero` + serial
trunk reduction + `adam_step`, all O(143,328 floats) per step and independent of
batch size. Three cheap fixes, none of which change results:

* Parallelise the trunk gradient reduction across threads instead of running it
  serially on thread 0 (it is a plain elementwise sum over disjoint ranges).
* `adam_step` already zeroes its gradient on the way out; check whether the
  separate per-thread `grad_zero` at the top of each step is redundant.
* Raise `batch_size` or lower `steps_per_gen` at constant `batch x steps` — the
  fixed per-step cost is amortised over more positions. **This changes learning
  dynamics**, so it is a training decision, not a performance one; flag it to
  whoever owns the pipeline rather than doing it unilaterally.

Expected: 30-50 % off the learning phase, so ~5-7 % of a generation. Do it after
item 1; it is a rounding error until the kernel is fixed.

### 4. Give each side its own evaluation cache — **worth ~2 %, bit-exact; do not bother yet**

Section 3c-i: the two-agent pairing invalidates the cache and the tree every
move, costing 5.3 % of evaluations. Only the cache half can be recovered
bit-exactly (key the cache on the head, or hold one `Mcts` per side); the tree
half changes search results. 2 % against a kernel worth 226 % — **park it.**

### 5. Re-measure thread scaling and QoS on mains power — **required before anyone tunes threading**

The harness is built and the commands are in section 2.2. This is 10 minutes of
machine time and zero code. Until it is done, nobody should change the thread
count, the QoS, or the work-distribution strategy — and on the evidence in
section 3c, the work distribution is the one thing that is already provably fine.

### Explicitly not worth doing

* **Faster move generation.** `gen_legal` is **1.2 %** of self-play. Making it
  infinitely fast buys 1.2 %.
* **Faster feature encoding.** `nn_features` is **0.07 %**.
* **A faster driver loop.** Everything outside `mcts_search` is **0.08 %**.
* **Chasing compiler flags.** `-mcpu=native` is already applied and the hot
  kernels already emit NEON FMA. There is no free win hiding in the Makefile.
* **An incremental NNUE-style accumulator.** The sparse gather is already only
  17.9 KB of the ~108 KB touched, and `nn_eval`'s cost is the dense `W1` matvec,
  not the gather. `src/net.h` already says as much; the measurements agree.
