# Design

This document explains *why* `qmodel_7_onyx` is built the way it is: the
reasoning behind each design choice, and the failure modes each one exists
to avoid. For file layout, module responsibilities, and data flow, see
[ARCHITECTURE.md](../../../ARCHITECTURE.md) instead - this document assumes
that map and focuses on intuition, not structure.

## The problem

A QATCH run is a dissipation / resonance-frequency time series recorded
while a liquid sample fills a microfluidic chip. The chip fills in a fixed
physical sequence: an initialization phase, then three channels (ch1, ch2,
ch3) fill one after another. Each fill event perturbs the signal - a
transition the model has to locate in time. The five points of interest
(POI1...POI5 in chain space: two initialization landmarks followed by the
three channel-fill transitions) are the run's "ground truth" - and the
whole system exists to predict their times from the raw signal, plus how
many of the three channels actually filled (a run can end after any
channel, and does not always complete all three).

Two properties of the signal make this harder than it sounds:

- **Transitions vary by orders of magnitude in scale.** A fast, low-viscosity
  fill produces a sharp step; a slow, viscous one produces a barely-visible
  slope change over tens of seconds.
- **Later events are systematically fainter, relative to the run.** Whatever
  normalization is applied per-run, the *earliest* large transition tends to
  dominate the visible dynamic range, and everything after it gets
  visually compressed.

Most of the design decisions below trace back to one or the other of these
two facts.

## Detection on rendered images, not sequence labeling

The signal is turned into an image and handed to a YOLO object detector,
rather than treated as a 1-D sequence for a classifier or segmentation head
to label directly. The payoff is what YOLO already gives you for free: a set
of scored, localized *candidates* per class, not just a single best guess.
That matters downstream in two concrete ways used elsewhere in this system -
the decode layer (below) needs multiple candidates per POI to arbitrate
between, not just one point estimate, and confidence is a first-class,
calibrated-by-training output rather than something that has to be derived
post hoc from logits. Framing localization as detection over an image also
means the mature augmentation, training, and inference tooling around YOLO
(letterboxing, rect batching, hyperparameter search, export) is reusable
almost as-is - see the rect-training section below for one place that
tooling still had to be corrected for this problem's shape.

## Salience-strip rendering: fighting late-event flattening

The render is not a plot of raw dissipation/resonance values. Each image has
three horizontal strips: two are the raw dissipation and resonance-frequency
traces (plain percentile normalization, preserved so the detectors keep
global fill-context cues - "how far into the run am I"), and the third is a
purpose-built **salience strip** that is *not* a transform of amplitude at
all.

The reason is the "later events are fainter" problem above. Per-strip
percentile normalization make the first, usually-largest step own the
dynamic range; on a long, viscous run, everything after it flattens into a
near-featureless plateau - exactly where earlier training runs measured the
worst recall (ch2/ch3 detectors specifically). No global amplitude rescaling
fixes this, because it is still scaling one shared range across events that
don't share a scale. The fix is to stop rendering amplitude and render what
the detector is actually looking for instead:

- The **detector cascade** (`rendering/detector_render.py`) renders a
  **derivative-energy** strip: the robustly-scaled, log-compressed `|d/dt|`
  of both signals, combined across a few timescales. POIs are transitions,
  not levels, so this strip turns every transition into a bright ridge with
  roughly uniform salience regardless of where in the run it falls or how
  large the absolute value change is.
- The **fill classifier** (`rendering/fill_render.py`) needs something even
  more sensitive to late events, because its job is literally to count how
  many transitions have happened - the class boundary between "2 channels
  filled" and "3 channels filled" is exactly the boundary a flattened plateau
  erases. It renders a **step-coincidence energy** strip instead: a matched
  step filter combined across signals via a geometric mean, which stays
  sensitive to transitions too slow for a simple derivative to register.

Both renderers are shared, byte-identical, between dataset building and
production inference (the same function builds the image whether it's
labeling training data or serving a prediction) specifically so there is no
train/inference skew in what "the same input" looks like.

## Fill-state gating: deciding which cascade stages even run

Before any channel detector runs, a 5-class ordinal classifier
(`no_fill` / `initial_fill` / `1ch` / `2ch` / `3ch`) looks at the whole run
and decides how many channels actually filled. That count then gates which
cascade stages execute at all (`if num_channels >= 3: ...`, `>= 2`, `>= 1`).

This exists because a channel detector has no way to distinguish "the
event isn't in this window" from "the event hasn't happened yet, ever." A
run that stopped after channel 1 has no ch2 or ch3 transition anywhere in
the signal - running those detectors on it either produces a low-confidence
false positive on noise, or (worse) a confident one on some incidental
slope change. Gating on a count decided up front turns an open-ended search
problem ("is this transition present anywhere?") into a bounded one ("find
this transition, which the classifier says exists").

## Reverse Cascading Detection: why last-to-first

The cascade runs **ch3 -> ch2 -> ch1 -> init**, the opposite of the
physical fill order. After each stage detects a transition at time `t`, the
signal is cut to `< t` before the next stage runs - so each earlier stage
searches a strictly shrinking prefix of the run, bounded on the right by the
event just found ahead of it.

Running in reverse, rather than physical order, is what makes that cut
useful as a *disambiguating* signal instead of just a speed optimization.
The ch3 transition - the last one to occur - gets the entire run as its
search window, which is exactly where a detection cascade wants the least
ambiguity: with no later look-alike event anywhere in its window, whatever
ridge the ch3 detector locks onto is very likely the real one. Once that cut
lands, ch2's detector never even sees the ch3 region - so it cannot latch
onto a transition that visually resembles what it's trained for but is
actually the wrong event downstream. Every subsequent (earlier) stage
inherits a tighter, already-disambiguated window than the one before it.
Running the cascade the other way around - init first - would give the
*first* detector the whole run too, but every stage after it would still
have every later transition sitting uncut in its search window, so nothing
about the later stages would get any easier.

## Candidate harvesting: keeping the decoder's options open

The cascade above is a *greedy* pipeline: each stage keeps only its single
best pick, and the next stage's cut is anchored to it. That's fine as a
production default, but it means a single wrong greedy pick permanently
excises the true event before any later mechanism - the decode layer below -
ever gets a chance to reconsider it.

`predict()` optionally runs a second, parallel bookkeeping pass
(`harvest_candidates`) that records *every* candidate each detector saw at
each stage, not just the winner. Critically, this harvest runs on its own
slice chain (`harvest_df`), cut at the **latest** candidate any stage
proposed rather than the greedy pick - deliberately wider than what
production ever sees. If the greedy cut had been used to drive the harvest
too, an early greedy mistake would delete the true downstream event from the
harvest slice before its own detector ever ran on it, and no amount of
downstream cleverness could recover a candidate that was never generated.
The wider harvest chain guarantees the true event survives into the
candidate pool whenever *any* stage's search window covered it - which is
what the decode layer below needs to have a real chance at fixing a bad
greedy pick instead of just rationalizing it.

## Configuration-prior decode: one joint choice instead of five independent ones

The cascade's greedy picks are five independent decisions, each made with
no knowledge of the others. But the POIs are not independent - the physical
process that produces them enforces a rough, learnable rhythm: consecutive
fill events tend to be separated by gaps in a fairly predictable (though
viscosity-dependent) range. `SpacingPrior` learns a log-normal model of
those gaps from complete runs, and `dp_decode` uses it as the objective
function for an exact dynamic-program search that picks *one candidate per
present POI, jointly*, maximizing a combination of detector confidence and
how well the resulting gaps fit the learned prior - instead of just taking
each stage's local best.

This is where the harvested candidate pools earn their cost: the decoder
can only correct a bad greedy pick if the true event is present as *some*
candidate in the pool, which is exactly what the wider harvest chain above
guarantees.

Two safety properties keep this from being a net risk over the plain
cascade:

- **Accept-margin (hysteresis).** The decoder doesn't unconditionally
  replace the cascade's picks - the cascade's own configuration is scored
  under the same objective, and the decoded configuration only wins if it
  beats the cascade's score by `DECODE_MIN_MARGIN`. A configuration that is
  merely *different*, not *better*, is left alone. This check is skipped
  when the decode places a POI the cascade missed entirely - scores over
  different POI sets aren't comparable, and recovering a missed POI is
  always worth taking regardless of margin.
- **Feasibility fallback.** If the DP search can't find a feasible joint
  assignment (or errors), it falls back to the same greedy picks the
  cascade already made. The decode layer can only make things strictly
  better or leave them unchanged in production - it cannot make the
  no-decode baseline worse.

## Crosscheck: a conservative second opinion, spent only when needed

The fill-count classifier is a single forward pass over the whole run - fast,
but it's still just a prior on how many channels filled, not a verdict.
`inference/crosscheck.py` runs immediately after the cascade (using
detectors already loaded for zoom refinement, so it costs nothing extra to
have them available) and spends them on two checks framed with deliberately
asymmetric confidence:

- **Under-count rescue.** If the classifier says fewer than 3 channels
  filled, slide the next channel's zoom detector over the tail of the run
  after the last confirmed POI. A confident detection there is strong,
  specific, local evidence a transition exists that the whole-run classifier
  missed - so the verdict is upgraded and the cascade is re-run with the
  extra channel enabled. This is safe-by-construction: a confident positive
  detection in a narrow, targeted window is hard to get by accident.
- **Over-count veto.** The mirror check - re-render a zoom window around the
  cascade's own placement for the claimed last channel and ask whether
  anything is actually there. This one is *advisory only*, reported but not
  auto-applied, because silence at zoom scale is ambiguous: it could mean
  the fill classifier hallucinated a channel that never filled, or it could
  just mean the zoom detector itself failed to recall a real event. Treating
  ambiguous evidence as a veto would trade one class of false negative for
  another; leaving it to the caller keeps the ambiguity visible instead of
  silently resolving it in one direction.

Both checks exploit the same asymmetry the salience renders were built to
correct: a slow, late transition that's a near-invisible slope change at
full-run scale can fill a large fraction of a tightly zoomed window around
it - so the zoom detectors, trained on exactly that tighter distribution,
see evidence the full-run view doesn't.

## Zoom refinement: sharpening within a basin, not searching for one

After decode has picked a globally-coherent configuration, each placed
channel POI gets one more optional pass: a narrow window
(`REFINE_WINDOW_S`) around the decoded time is re-rendered at full image
width and re-detected with a zoom-trained detector. The refined position is
accepted only if it's confident *and* within a trust region - a fraction of
the window width (`REFINE_MAX_SHIFT_FRAC`) - of the decoded anchor; anything
further is treated as latching onto a different event rather than sharpening
the right one, and the decode's pick is kept instead.

This has to run *after* decode, not instead of it or before it. Decode's job
is choosing the right *basin* - which of several plausible candidates,
consistent with the others, is the real configuration. Zoom refinement's job
is narrowing the localization *within* whatever basin was chosen, using
the extra pixel resolution a tight window buys over a full-run render. Doing
it first would mean sharpening a placement that global decode might still
overturn; the ordering lets each stage do only the one thing it's actually
good at.

## Viscosity tiering and upsampling: keeping the long tail visible

Run viscosity spans several orders of magnitude, and the corpus is
right-skewed - most runs are low-viscosity, with a long, sparse tail of
slow, viscous ones. Those are exactly the runs the salience renders and
zoom refinement exist for, so a training split that underrepresents them
would quietly undermine the rest of the design.

`tiers.py` defaults to `log_uniform` binning - equal-width bins in
`log10(cP)` space - specifically because the alternatives collapse that
tail: equal-*count* binning (`quantile`) or BIC-selected clusters (`gmm`)
both optimize for representing the dense, low-viscosity mass well, at the
cost of merging the sparse high-viscosity runs into one indistinct bucket.
Equal-width log bins don't optimize for count at all, so a tier's width in
`cP` terms grows with position, but the same *number* of tiers stays
resolvable across the full span the corpus actually has, including the
tail.

`dataset/splitting.py` builds on that in two ways: `stratified_group_split`
groups whole runs by `(tier, POI count)` before splitting, so tail-viscosity
and partial-fill runs land in both train and val instead of leaking or
disappearing from one; `repeat_factor` then upsamples each tier by roughly
`sqrt(n_max / n_tier)` (clipped) rather than to full parity, which corrects
under-representation without fully drowning the dense low-viscosity tiers in
duplicated tail runs.

## Signal-domain augmentation: augmenting before the pixels exist

Augmentation (`augmentation.py`) - time warping, noise injection, amplitude
jitter - happens to the raw signal and its POI labels *before* rendering,
not to the rendered pixels afterward. Pixel-space augmentation (the usual
YOLO defaults: mosaic, flips, perspective, scale/shear) is explicitly turned
off (see the training section below).

The reason is that the render's geometry carries physical meaning a generic
image augmentation would break. A horizontal flip would reverse time. A
perspective warp or shear would distort the derivative-energy strip in a way
that doesn't correspond to any real transition shape. Time-domain warping,
by contrast, is exactly what a real run's variability looks like - the same
fill sequence happening somewhat faster or slower - so it's applied where it
stays physically meaningful: to the time axis of the signal itself, with the
POI labels carried through the same warp so they stay correct by
construction, before the render (and its label boxes) is ever produced.
`dynamic_box_width_sec` follows the same logic: rather than a fixed label
box width, it measures each transition's actual duration from the smoothed
derivative magnitude and sizes the box to match, so the label reflects how
gradual or sharp that specific event really is.

## The rect-training workaround

`training/train_detectors.py` overrides Ultralytics' `DetectionTrainer` with
a `RectDetectionTrainer`. This isn't a performance tweak; it works around a
real correctness bug: Ultralytics hard-codes `rect=(mode == "val")` inside
`build_dataset`, so the `rect=True` argument passed to `model.train()` is
silently honored for validation but ignored for the training loader. Since
these renders are wide, short strips, letterboxing them into a square at the
long edge's size means most of every training image is black padding -
inflating memory usage and slowing training through the resulting
host/PCIe thrashing. The override forces genuinely rectangular batches at
the render's native aspect ratio for both loaders, so what the model trains
on is close to what it's actually shaped like.

## Two copies of the controller, on purpose

`inference/controller.py` and `deployment/onyx.py` are separate,
near-identical implementations of the same controller, not a shared class
with a thin wrapper. That duplication is deliberate: `deployment/` is
shipped and imported under the `QATCH.QModel.models.qmodel_onyx.*` dotted
path a downstream consumer (the QATCH nanovis app) actually uses, so the
release/eval pipeline (`scripts/build_and_release_qmodel_onyx.py`) can load
and exercise it exactly the way that consumer will - not a proxy for it. A
shared base class importable only from inside this repo's own package
layout wouldn't be loaded the same way, so it wouldn't actually validate the
one thing this split exists to validate: that the package works when
imported the way it will really be imported.

## Recap

Nearly every choice above traces back to one of two things: rendering *what
the detector needs to see* instead of *what the value happens to be*
(salience strips, dynamic box widths), or giving every stage a narrower,
better-disambiguated problem than the one before it (reverse cascade, greedy
cut, decode's joint search, zoom's trust region). Where a stage could make
things worse instead of only better, it's built to fail toward the
established baseline rather than past it: decode falls back to greedy,
crosscheck's veto is advisory only, zoom refinement keeps the decode pick
outside its trust region. The result is a pipeline where each added stage is
optional and additive - correctness under `predict()`'s defaults never
depends on every optional stage succeeding.
