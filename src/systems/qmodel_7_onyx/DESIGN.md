# Design

This document explains *why* QModel Onyx is designed the way it is, the
reasoning behind the major design choices, the assumptions they rely on, and
the failure modes each mechanism is intended to reduce. It is deliberately
concerned with behavior and intuition rather than implementation structure.

For file layout, module responsibilities, and end-to-end data flow, see
[ARCHITECTURE.md](../../../ARCHITECTURE.md). This document assumes that
overview and focuses on the design rationale behind the pipeline.

## The problem

A NanovisQ run is a time series of dissipation and resonance-frequency
measurements recorded while a liquid sample fills a microfluidic QCM-D sensor.
The physical process follows a fixed event order: an initialization phase
containing two landmarks, followed by the fill transitions for channel 1,
channel 2, and channel 3.

These five ordered events are the run's points of interest (POIs). The system
must estimate the time of each POI that is actually present and determine how
many channel-fill events occurred. A run may terminate after any channel, so
not every run contains all three channel transitions.

The task is therefore not simply "find five peaks." It is an ordered,
variable-cardinality event-localization problem in which the visual evidence
for an event can vary substantially from run to run.

Several properties of the signal make this harder than the physical sequence
suggests:

- **Inconsistent event shape.** A fill transition can appear as a sharp step,
  a broad slope, or a more gradual change. Its visual signature depends on
  the sample, sensor response, and where the event occurs in the run.
- **Inconsistent event timing.** The elapsed time between corresponding events
  is not fixed. Fill duration varies with the run conditions, so absolute
  timestamps are not reliable event templates.
- **Variable number of fill events.** A run can contain zero, one, two, or
  three channel-fill transitions. The absence of a later event is therefore a
  valid outcome.
- **Extreme localization requirements.** Small timing errors can correspond
  to large errors in the underlying viscosity measurment when the signal is rendered
  at full-run scale. Localization therefore has to be treated separately from
  coarse event discovery.
- **Variable sensor characteristics.** Absolute dissipation and
  resonance-frequency levels vary across sensors and runs. A detector that
  relies primarily on absolute amplitude is therefore less transferable than
  one that emphasizes transition structure.
- **Subjective event boundaries.** Some POI locations are determined from the
  shape and visual interpretation of a transition rather than from a single
  mathematically unambiguous sample. Training labels consequently have
  finite localization uncertainty.

Most of the design decisions below trace back to these properties.

## Why use object detection?

The system treats POI localization as an object-detection problem over a
rendered representation of the run rather than as direct sequence labeling.

This choice separates two concerns that are difficult to satisfy
simultaneously with a per-sample classifier:

1. **Finding the event.** The detector learns the visual structure of a
   transition and predicts a localized region around it.
2. **Localizing the event.** The predicted region provides a continuous
   spatial representation of the event boundary instead of requiring one
   exact sample to be treated as the uniquely correct label.

A full run is encoded as an image whose horizontal axis
represents time and whose vertical bands contain different signal
representations. Each POI is represented during training by a bounding box
centered on its annotated event location. The box therefore encodes an
interval around the event rather than pretending that a subjective transition
has a single indisputable pixel or sample.

Object detection is also a practical fit for the highly variable run lengths
and rendering resolutions. The same detector interface can operate on
full-run views and on zoomed views while the rendering pipeline controls how
much temporal detail is represented by the available pixels.

## Lossy signals

The rendered image is not simply a plot of the raw signal. It contains three
horizontal strips, each serving a different purpose:

- a dissipation strip,
- a resonance-frequency strip, and
- a purpose-built salience strip.

The first two strips use percentile normalization. They retain global
signal-level and fill-context information, including cues about where the
model is within the overall run. The third strip deliberately changes the
representation.  This strip emphasizes transition evidence rather than absolute
signal amplitude.

This distinction addresses a common failure mode in long, viscous runs.
When the first fill transition is substantially larger than later transitions,
normalizing all events into one shared amplitude range can give the first
event most of the available dynamic range. Subsequent transitions can then
collapse visually toward a flat plateau. This was particularly harmful to
recall for later channel events.

Simply rescaling the original amplitudes does not solve the underlying
problem. The detector is interested primarily in *changes* in the signal, not
in the absolute level of the signal. The salience representation therefore
makes transition evidence explicit.

### Detector salience

The detector renderer
(`rendering/detector_render.py`) produces a **derivative-energy** strip. It
uses a robustly scaled, log-compressed magnitude of the time derivative of
both input signals, combined across multiple temporal scales.

POIs correspond to transitions rather than steady-state levels. The
derivative-energy representation consequently converts both sharp and
relatively broad transitions into localized regions of elevated evidence,
while reducing the dependence on their absolute amplitude.

### Fill type salience

The fill classifier
(`rendering/fill_render.py`) has a different objective.  This classifier must determine how
many channel transitions occurred. Its difficult cases are often precisely
the slow, late transitions that can disappear in a derivative-only
representation.

It therefore uses a **step-coincidence energy** representation.  For this strip, a matched-step
response is computed and combined across the input signals using a geometric
mean. This preserves evidence for transitions that are broad enough to be
weak in a simple derivative representation.

### Train/inference consistency

The rendering functions used to construct training examples are also used
for production inference. Keeping these renderers shared and identical
prevents train/inference skew and the model sees the same definition of its
input representation during training and deployment.

## Detector orchestration

Before channel-specific detection, a five-class ordinal classifier
(`no_fill`, `initial_fill`, `1ch`, `2ch`, `3ch`) estimates how many channel
fills occurred in the run. That estimate is then used to gate the channel
detectors so that only stages corresponding to expected events are executed.

This is necessary because a channel detector cannot, by itself, reliably
distinguish between:

- "the event is not in this search window," and
- "the event never occurred in this run."

For example, a run that ended after channel 1 contains no channel 2 or
channel 3 transition. Searching for those events anyway creates an open-ended
false-positive problem.  In other words, an incidental slope or noise feature may look
sufficiently similar to a channel transition to produce a confident prediction.

Fill-state gating changes the task from "search for this event and determine
whether it exists" to "search for this event because independent evidence
indicates that it should exist." The classifier is therefore a structural
prior, not a mathematical guarantee. Later stages retain mechanisms for
checking or recovering from an incorrect count.

## Why use a backwards cascade?

The channel cascade runs in reverse physical order:

**Channel 3 -> Channel 2-> Channel 1 -> Initial**

After a stage detects an event at time `t`, the signal is restricted to the
prefix preceding `t` before the next stage runs. Each earlier detector
therefore receives a strictly smaller search region bounded by an event that
has already been localized.

Running the cascade backward makes that restriction useful for
disambiguation, not merely for reducing computation.

The latest expected event receives the full run as its search domain. Once it
has been localized, the channel 2 detector cannot see the channel 3 region.
Likewise, after channel 2 is localized, channel 1 cannot select a visually
similar feature downstream of channel 2. The cascade therefore incorporates
the known event ordering directly into the search space.

The shrinking prefix also improves the effective temporal resolution of the
earlier stages. A fixed-width image represents a shorter time interval after
each cut, so the same number of horizontal pixels corresponds to fewer
seconds of signal. This is particularly useful for early POIs, where precise
localization is important.

Running in physical order would give the initial detector the entire run and
would leave later detectors exposed to all downstream transitions. It would
also provide the early detector with relatively poor full-run temporal
resolution. Reverse cascading addresses both issues with the same structural
constraint.

## Candidate evaluation

The production cascade is intentionally greedy. Each detector stage keeps its
best candidate and uses that placement to define the next search region. This
is efficient and provides a strong production baseline, but it has an
important failure mode. A single incorrect greedy choice can remove the true
event from every later search window.

`predict()` can therefore run a second bookkeeping pass
(calling `harvest_candidates`) that retains the candidate set produced by each
detector stage rather than only its winner.

The harvest pass uses its own slice chain. Each successive slice is bounded
using the latest candidate proposed by the preceding stage, rather than the
greedy winner used by production inference. The harvest window is
intentionally wider.

This distinction is important. If the greedy cut also drove candidate
harvesting, an early greedy mistake could remove the true downstream event
before the detector responsible for that event had an opportunity to propose
it. A later decoder cannot recover a candidate that was never generated.

The wider harvest chain therefore preserves alternative hypotheses for the
global decode stage whenever the relevant event was visible to a detector's
search window.

## Configuration-prior decoding

The greedy cascade makes local decisions. The POIs, however, are not
independent but are generated by a physical process with a consistent
ordering and a characteristic distribution of inter-event gaps.

`SpacingPrior` models those gaps from complete runs using a log-normal
distribution. `dp_decode` then uses dynamic programming to select one
candidate for each POI that is expected to be present, jointly optimizing:

- detector confidence, and
- compatibility of the resulting event spacings with the learned prior.

The result is a globally coherent configuration rather than five unrelated
local maxima.

Candidate harvesting is essential here. The decoder can correct a greedy
mistake only when the correct event remains represented somewhere in the
candidate pool. The wider harvest chain increases the probability that a
useful alternative survives long enough to be considered by the decoder.

Two safeguards keep joint decoding from becoming a new source of production
regressions:

- **Accept margin** The cascade's own configuration is scored
  using the same objective as the decoded configuration. The decoded result
  replaces the cascade only when it improves the objective by at least
  `DECODE_MIN_MARGIN`. A merely different configuration is therefore not
  sufficient. If decoding introduces a POI that the cascade missed entirely,
  the configurations are not directly comparable, so the margin requirement
  is not applied; recovering a missing expected event is treated separately.
- **Feasibility fallback** If dynamic programming cannot find a feasible
  assignment, or the decode operation fails, the system returns the original
  cascade result. The decoder is consequently an enhancement to the baseline,
  not a dependency whose failure can invalidate an otherwise usable
  prediction.

## Crosschecks (curently not deployed!)

The fill-count classifier is a fast whole-run estimate of the number of
channel fills, but it remains a classifier and can be wrong. The crosscheck
stage in `inference/crosscheck.py` uses the already-loaded zoom detectors to
test the classifier's decision in targeted regions.

The checks are intentionally asymmetric because positive and negative
evidence are not equally reliable.

- **Under-count rescue** If the classifier predicts fewer than three
  channels, the next expected channel detector is applied to a targeted tail
  region after the last confirmed POI. A sufficiently confident detection is
  treated as strong local evidence that an event exists and was missed by the
  whole-run classifier. The channel count is upgraded and the cascade can be
  rerun with the additional stage enabled.
- **Over-count advisory check** If the classifier/cascade indicates a final
  channel, the system can re-render a zoomed region around the proposed event
  and ask whether the detector sees supporting evidence. This result is
  advisory rather than an automatic veto. Failure to detect an event in the
  zoomed view is ambiguous: it may indicate a false positive, but it may also
  indicate a detector miss. Automatically vetoing the event would therefore
  risk converting one failure mode into another.

Both checks exploit the same scale effect addressed by the salience renders:
a slow late transition can be visually weak in a full-run image but occupy a
substantial portion of a tightly zoomed window. Zoom-trained detectors can
therefore operate on a representation with substantially more temporal detail
around the candidate event.

## Zoom refinement

After the global configuration has been selected, each placed channel POI can
receive an optional refinement pass. A narrow window
(`REFINE_WINDOW_S`) centered on the selected time is rendered at full image
width and evaluated by a zoom-trained detector.

The refined position is accepted only when:

1. the zoom detector is sufficiently confident, and
2. the proposed position remains within a trust region defined by
   `REFINE_MAX_SHIFT_FRAC` of the refinement window width.

A candidate outside that trust region is treated as evidence that the zoom
detector may have latched onto a different event. In that case, the globally
decoded position is retained.

This stage deliberately runs **after** global decoding. The decoder chooses
the correct *basin* among competing event hypotheses using information from
the whole configuration. Zoom refinement performs a different task: it
sharpens the location *within the selected basin* using the higher temporal
resolution of a narrow render.

Running refinement first would risk spending high-resolution capacity on a
candidate that the global decoder would later reject. The ordering ensures
that each stage performs the task for which it has the strongest information.

## Viscosity tiering and upsampling

Run viscosity spans several orders of magnitude, and the corpus is
right-skewed. Most runs are concentrated at lower viscosity while a smaller
number occupy a long high-viscosity tail. Those high-viscosity runs are
especially important because their late transitions are among the hardest
cases for the detection pipeline.

`tiers.py` therefore defaults to `log_uniform` binning equal-width bins in
`log10(cP)` space. This preserves resolution across orders of magnitude
without allowing the dense low-viscosity region to consume all available
tiers.

Two alternatives have different failure modes:

- Equal-count (`quantile`) bins devote approximately equal numbers of samples
  to each tier, which can compress a wide high-viscosity range into a small
  number of broad bins.
- BIC-selected Gaussian-mixture (`gmm`) clusters optimize a statistical
  description of the observed distribution and can similarly devote most
  of their resolution to the dense low-viscosity population.

Log-uniform bins instead make the tier boundaries scale with the physical
range of viscosity. As a result, the sparse tail remains represented as
distinct regions even when it contains relatively few runs.

`dataset/splitting.py` builds on this in two ways:

1. `stratified_group_split` groups complete runs by `(tier, POI count)` before
   splitting. This keeps related observations from being separated in a way
   that could cause leakage and ensures that both viscosity extremes and
   partial-fill cases are represented in the relevant splits when the corpus
   permits.
2. `repeat_factor` upsamples underrepresented tiers by approximately
   `sqrt(n_max / n_tier)`, with clipping. Partial rather than full balancing
   provides additional exposure to rare cases without allowing a small number
   of tail runs to dominate training.

## Signal-domain augmentation

Augmentation (`augmentation.py`) is applied to the raw signal and its POI
labels before rendering. Current transformations include time warping, noise
injection, and amplitude jitter.

Generic pixel-space augmentation is deliberately disabled for these training
images, including the usual YOLO transformations such as mosaic, flips,
perspective, scale, and shear.

The reason is that the image geometry has physical meaning:

- A horizontal flip reverses time and therefore reverses the physical event
  sequence.
- Perspective and shear change the apparent timing geometry of transitions in
  ways that do not correspond to the intended signal-domain variability.
- Pixel transformations can also alter the relationship between the rendered
  signal and the labels in ways that are difficult to interpret physically.

Time-domain augmentation, by contrast, models a genuine source of variation i.e.,
the same fill sequence occurring somewhat faster or slower. Applying the
transformation to both the signal and its POI labels preserves their
relationship by construction.

`dynamic_box_width_sec` follows the same principle. Instead of assigning every
POI the same label-box width, it estimates the transition's duration from the
smoothed derivative magnitude and sizes the box according to the observed
shape of that event. Therefore, the label represents the temporal extent of
the transition rather than imposing a universal width on events with
different dynamics.

## Recap

The design can be reduced to two recurring principles.

First, **represent the evidence the detector needs rather than relying only on
the raw value of the signal**. Salience strips emphasize transitions,
percentile-normalized signal strips preserve global context, and dynamic box
widths reflect the observed duration of individual events.

Second, **give each stage a more constrained and better-informed problem than
the stage before it**. Reverse cascading uses event order to shrink the search
space; candidate harvesting preserves alternatives that greedy inference
might discard; joint decoding uses relationships between events; crosschecks
test structural assumptions with targeted evidence; and zoom refinement
improves localization only after the global event configuration has been
selected.

Where an optional stage could otherwise make a baseline prediction worse,
the design favors conservative failure behavior. Decoding falls back to the
greedy cascade when it cannot produce a valid improvement, over-count
crosschecking remains advisory, and zoom refinement rejects candidates that
move outside its trust region.

The result is a layered inference pipeline in which each mechanism addresses
a specific failure mode while preserving the established baseline when its
own evidence is insufficient.
