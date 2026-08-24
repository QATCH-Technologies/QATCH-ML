"""Joint dynamic-programming decode over candidate POI detections.

Provides a global alternative to independent per-POI confidence selection by
choosing a temporally ordered configuration of candidate detections under
learned spacing constraints and a spacing prior. The decoder supports partial
fills, non-adjacent present POIs, span-conditioned likelihoods, and a
production-safe greedy fallback when no complete admissible path exists.

The module is intentionally independent of the surrounding predictor: inputs
and outputs are represented by plain dictionaries and lightweight data
classes, allowing the decoder to be integrated into different inference
pipelines.

The primary entry point is :func:`dp_decode`. Supporting helpers prepare
candidate lattices, perform dynamic-programming passes, condition the
fractional spacing objective on the selected span, and provide compatible
greedy scoring and baseline behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .spacing_prior import POI_ORDER, SpacingPrior


@dataclass
class Candidate:
    """Represent a candidate detection for a point of interest.

    Attributes:
        time (float): Detected timestamp in seconds.
        conf (float): Detection confidence, expected to lie in the range
            `[0, 1]`.
    """

    time: float  # detected timestamp (seconds)
    conf: float  # confidence in [0,1]


@dataclass
class DecodeResult:
    """Contain the selected POI configuration and its decode metrics.

    Attributes:
        chosen (Dict[str, Candidate]): Mapping from POI name to the selected
            candidate.
        total_score (float): Combined confidence and spacing-prior score of
            the selected configuration.
        spacing_loglik (float): Unweighted spacing log-likelihood accumulated
            across the selected configuration.
        conf_sum (float): Confidence contribution to the total score.
        feasible (bool): Whether the selected configuration satisfies the
            requested feasibility constraints.
        fallback_used (bool): Whether relaxed decoding or greedy fallback was
            required.
    """

    chosen: Dict[str, Candidate]  # poi_name -> chosen candidate
    total_score: float
    spacing_loglik: float
    conf_sum: float
    feasible: bool  # False if no fully-feasible path existed
    fallback_used: bool  # True if we relaxed to greedy/partial


def _clip01(x: float) -> float:
    """Clamp a numeric value to the unit interval.

    Args:
        x (float): Value to constrain.

    Returns:
        float: `x` limited to the range `[0.0, 1.0]`.
    """
    return max(0.0, min(1.0, x))


def _lam_between(lam, i: int, j: int) -> float:
    """Resolve the spacing-prior weight for a POI pair.

    Supports either a single scalar weight applied uniformly to all spacing
    edges or a mapping containing individual weights for global POI pairs.
    When a pair spans intervening POIs, the weights of the constituent
    adjacent pairs are averaged.

    Args:
        lam (float | dict): Global spacing-prior weight or mapping of pair
            names to individual weights.
        i (int): Global index of the first POI in the edge.
        j (int): Global index of the second POI in the edge.

    Returns:
        float: Effective spacing-prior weight for the requested edge.
    """
    if not isinstance(lam, dict):
        return float(lam)
    names = [f"{POI_ORDER[k]}->{POI_ORDER[k + 1]}" for k in range(i, j)]
    return float(np.mean([lam.get(n, 1.0) for n in names]))


def _prep_candidates(
    candidates: Dict[str, List[Candidate]],
    present: List[str],
    min_conf: float,
    max_candidates: int,
) -> Dict[str, List[Candidate]]:
    """Prepare candidate detections for dynamic-programming decode.

    Filters candidates by confidence when possible, preserves a POI when
    filtering would otherwise remove all of its candidates, deduplicates
    timestamps, limits each POI to the highest-confidence candidates, and
    finally sorts candidates chronologically for lattice traversal.

    Args:
        candidates (Dict[str, List[Candidate]]): Candidate detections grouped
            by POI name.
        present (List[str]): Ordered POIs that should participate in decoding.
        min_conf (float): Minimum preferred confidence threshold.
        max_candidates (int): Maximum number of candidates retained per POI.

    Returns:
        Dict[str, List[Candidate]]: Prepared, deduplicated, confidence-capped,
        time-sorted candidate lists.
    """
    cand: Dict[str, List[Candidate]] = {}
    for p in present:
        cs = [c for c in candidates.get(p, []) if c.conf >= min_conf]
        if not cs:
            cs = list(candidates.get(p, []))  # don't drop a POI entirely
        # dedupe identical timestamps (coarse+fine stages can re-emit the same
        # box); keep the max-confidence instance.
        by_t: Dict[float, Candidate] = {}
        for c in cs:
            key = round(c.time, 6)
            if key not in by_t or c.conf > by_t[key].conf:
                by_t[key] = c
        cs = list(by_t.values())
        # cap to top-K by confidence (keeps the lattice small and prunes
        # noise-floor boxes), then sort by time for the DP.
        cs.sort(key=lambda c: c.conf, reverse=True)
        cs = cs[: max(1, max_candidates)]
        cs.sort(key=lambda c: c.time)
        cand[p] = cs
    return cand


def _score_config(
    chosen: Dict[str, Candidate],
    placeable: List[str],
    prior: SpacingPrior,
    lam: float,
    conf_weight: float,
    use_frac: bool,
) -> Tuple[float, float, float]:
    """Score a complete candidate configuration under the decode objective.

    Combines detection confidence with pairwise spacing log-likelihood while
    respecting the global POI ordering and any intervening POIs. When enabled,
    the spacing prior uses the span of the selected configuration for its
    fractional component.

    Args:
        chosen (Dict[str, Candidate]): Selected candidate for each placed POI.
        placeable (List[str]): Ordered POIs participating in the configuration.
        prior (SpacingPrior): Learned spacing model used to evaluate gaps.
        lam (float | dict): Global or per-edge spacing-prior weight.
        conf_weight (float): Weight applied to summed detection confidence.
        use_frac (bool): Whether the span-dependent fractional likelihood is
            enabled.

    Returns:
        Tuple[float, float, float]: Combined objective, unweighted spacing
        log-likelihood, and weighted confidence sum.
    """
    g_index = {POI_ORDER[i]: i for i in range(len(POI_ORDER))}
    placed = [p for p in placeable if p in chosen]
    conf_sum = conf_weight * sum(_clip01(chosen[p].conf) for p in placed)
    times = [chosen[p].time for p in placed]
    span = (max(times) - min(times)) if (use_frac and len(times) > 1) else 0.0
    span_lo = g_index[placed[0]] if placed else 0
    span_hi = g_index[placed[-1]] if placed else 0
    spacing_ll = 0.0
    weighted_ll = 0.0
    for a, b in zip(placed[:-1], placed[1:], strict=True):
        gi, gj = g_index[a], g_index[b]
        ll = prior.gap_loglik_scoped(
            gi, gj, chosen[b].time - chosen[a].time, span, span_lo, span_hi
        )
        spacing_ll += ll
        weighted_ll += _lam_between(lam, gi, gj) * ll
    return conf_sum + weighted_ll, spacing_ll, conf_sum


def _dp_pass(
    cand: Dict[str, List[Candidate]],
    placeable: List[str],
    prior: SpacingPrior,
    lam: float,
    conf_weight: float,
    feas_slack: float,
    require_feasible: bool,
    span_for_frac: float,
) -> Optional[Dict[str, Candidate]]:
    """Run one exact dynamic-programming pass over the candidate lattice.

    Evaluates temporally ordered candidate paths while optionally enforcing the
    learned gap-feasibility constraints. Each transition contributes its
    spacing-prior score and the confidence of the newly selected candidate.

    Args:
        cand (Dict[str, List[Candidate]]): Prepared candidate lists keyed by
            POI name.
        placeable (List[str]): Ordered POIs to decode.
        prior (SpacingPrior): Learned spacing model.
        lam (float | dict): Global or per-edge spacing-prior weight.
        conf_weight (float): Weight applied to candidate confidence.
        feas_slack (float): Multiplicative tolerance applied to learned gap
            bounds.
        require_feasible (bool): Whether infeasible gap transitions should be
            rejected.
        span_for_frac (float): Fixed span used by the fractional prior
            component. Non-positive values disable that component.

    Returns:
        Dict[str, Candidate] | None: Highest-scoring admissible configuration,
        or `None` when no complete path exists.
    """
    g_index = {POI_ORDER[i]: i for i in range(len(POI_ORDER))}
    span_lo, span_hi = g_index[placeable[0]], g_index[placeable[-1]]
    P = len(placeable)
    dp: List[List[float]] = [[-1e18] * len(cand[placeable[j]]) for j in range(P)]
    back: List[List[int]] = [[-1] * len(cand[placeable[j]]) for j in range(P)]

    for k, c in enumerate(cand[placeable[0]]):
        dp[0][k] = conf_weight * _clip01(c.conf)

    for j in range(1, P):
        a, b = placeable[j - 1], placeable[j]
        gi, gj = g_index[a], g_index[b]
        for k, cb in enumerate(cand[b]):
            best = -1e18
            best_prev = -1
            for kp, ca in enumerate(cand[a]):
                if dp[j - 1][kp] <= -1e17:
                    continue
                gap = cb.time - ca.time
                if gap <= 0:
                    continue  # hard: strict ordering
                if require_feasible and not prior.gap_feasible_between(gi, gj, gap, feas_slack):
                    continue  # hard: gap bounds
                ll = prior.gap_loglik_scoped(gi, gj, gap, span_for_frac, span_lo, span_hi)
                score = (
                    dp[j - 1][kp] + _lam_between(lam, gi, gj) * ll + conf_weight * _clip01(cb.conf)
                )
                if score > best:
                    best = score
                    best_prev = kp
            dp[j][k] = best
            back[j][k] = best_prev

    last = P - 1
    if not dp[last]:
        return None
    best_k = int(np.argmax(dp[last]))
    if dp[last][best_k] <= -1e17:
        return None

    chosen: Dict[str, Candidate] = {}
    k = best_k
    for j in range(last, -1, -1):
        if k < 0:
            return None  # defensive: broken backtrack chain
        chosen[placeable[j]] = cand[placeable[j]][k]
        k = back[j][k] if j > 0 else -1
    return chosen


def _span_conditioned_decode(
    cand: Dict[str, List[Candidate]],
    placeable: List[str],
    prior: SpacingPrior,
    lam: float,
    conf_weight: float,
    feas_slack: float,
    require_feasible: bool,
) -> Optional[Dict[str, Candidate]]:
    """Perform an exact span-conditioned decode.

    Enumerates possible first and last candidates, fixes their resulting span,
    and runs a dynamic-programming pass for each endpoint pair. The resulting
    configurations are consistently rescored under the complete
    span-dependent objective so the best valid configuration can be selected.

    Args:
        cand (Dict[str, List[Candidate]]): Prepared candidate lists keyed by
            POI name.
        placeable (List[str]): Ordered POIs participating in decoding.
        prior (SpacingPrior): Learned spacing model.
        lam (float | dict): Global or per-edge spacing-prior weight.
        conf_weight (float): Weight applied to candidate confidence.
        feas_slack (float): Multiplicative tolerance applied to learned gap
            bounds.
        require_feasible (bool): Whether gap-feasibility constraints must be
            satisfied.

    Returns:
        Dict[str, Candidate] | None: Highest-scoring span-consistent
        configuration, or `None` when no admissible configuration exists.
    """
    first, last = placeable[0], placeable[-1]
    best_chosen: Optional[Dict[str, Candidate]] = None
    best_score = -1e18
    for f in cand[first]:
        for last_c in cand[last]:
            span = last_c.time - f.time
            if span <= 0:
                continue
            sub = dict(cand)
            sub[first] = [f]
            sub[last] = [last_c]
            ch = _dp_pass(
                sub,
                placeable,
                prior,
                lam,
                conf_weight,
                feas_slack,
                require_feasible,
                span_for_frac=span,
            )
            if ch is None:
                continue
            s = _score_config(ch, placeable, prior, lam, conf_weight, True)[0]
            if s > best_score:
                best_score = s
                best_chosen = ch
    return best_chosen


def dp_decode(
    candidates: Dict[str, List[Candidate]],
    present_pois: Sequence[str],
    prior: SpacingPrior,
    lam: float = 1.0,
    conf_weight: float = 1.0,
    feas_slack: float = 1.5,
    min_conf: float = 0.0,
    require_feasible: bool = True,
    max_candidates: int = 10,
) -> DecodeResult:
    """Decode the best globally consistent POI configuration.

    Considers only POIs identified as present, prepares their candidate
    detections, and selects a configuration that maximizes the combined
    detection-confidence and learned-spacing objective. Consecutive placed
    POIs are required to be strictly time-ordered, while learned gap
    constraints can be enforced as hard constraints.

    When enough POIs are present for meaningful ratio information, the
    span-dependent spacing component is handled through exact
    span-conditioned dynamic programming. Non-adjacent present POIs are scored
    using the composition of the intervening fitted gaps.

    If no fully feasible configuration exists, the decoder first attempts a
    relaxed ordered decode. If that also fails, it falls back to independent
    highest-confidence selection for each available POI.

    Args:
        candidates (Dict[str, List[Candidate]]): Candidate detections grouped
            by POI name.
        present_pois (Sequence[str]): POIs determined to be present by the
            upstream fill/type classification stage.
        prior (SpacingPrior): Learned model used for gap feasibility and
            spacing likelihood.
        lam (float | dict, optional): Weight applied to spacing likelihood,
            either globally or per POI pair. Defaults to 1.0.
        conf_weight (float, optional): Weight applied to summed detection
            confidence. Defaults to 1.0.
        feas_slack (float, optional): Multiplicative slack applied to learned
            gap-feasibility bounds. Defaults to 1.5.
        min_conf (float, optional): Preferred minimum confidence for candidate
            filtering. Defaults to 0.0.
        require_feasible (bool, optional): Whether the initial decode must
            satisfy learned gap constraints. Defaults to True.
        max_candidates (int, optional): Maximum candidates retained per POI.
            Defaults to 10.

    Returns:
        DecodeResult: Selected configuration and its objective components,
        together with feasibility and fallback status.
    """
    present = [p for p in POI_ORDER if p in present_pois]
    if not present:
        return DecodeResult({}, -1e18, 0.0, 0.0, False, True)

    cand = _prep_candidates(candidates, present, min_conf, max_candidates)
    placeable = [p for p in present if cand[p]]
    if not placeable:
        return DecodeResult({}, -1e18, 0.0, 0.0, False, True)

    # The frac component needs at least two gaps to carry ratio information
    # (with a single gap, gap/span == 1 identically). Prefix-fill span
    # semantics are handled by gap_loglik_scoped, so partial fills are
    # eligible too.
    use_frac = len(placeable) >= 3 and prior.frac_blend > 0

    # decode: exact under the full objective. With frac active the
    # decode is span-conditioned (see _span_conditioned_decode); otherwise a
    # single seconds-only DP is already exact.
    if use_frac:
        chosen1 = _span_conditioned_decode(
            cand, placeable, prior, lam, conf_weight, feas_slack, require_feasible
        )
    else:
        chosen1 = _dp_pass(
            cand,
            placeable,
            prior,
            lam,
            conf_weight,
            feas_slack,
            require_feasible,
            span_for_frac=0.0,
        )

    if chosen1 is None and require_feasible:
        # Relax feasibility (keep only strict ordering) and decode once more.
        relaxed = _dp_pass(cand, placeable, prior, lam, conf_weight, 1e9, False, span_for_frac=0.0)
        if relaxed is None or len(relaxed) < len(placeable):
            # Even strict ordering has no complete path -> production-safe
            # floor: per-POI greedy, never worse than current behaviour.
            return _greedy_result(cand, placeable, prior, lam, conf_weight)
        total, sll, csum = _score_config(relaxed, placeable, prior, lam, conf_weight, use_frac)
        return DecodeResult(relaxed, total, sll, csum, False, True)

    if chosen1 is None:
        return DecodeResult({}, -1e18, 0.0, 0.0, False, True)

    total, sll, csum = _score_config(chosen1, placeable, prior, lam, conf_weight, use_frac)
    return DecodeResult(chosen1, total, sll, csum, True, False)


def _greedy_result(
    cand: Dict[str, List[Candidate]],
    placeable: List[str],
    prior: SpacingPrior,
    lam: float,
    conf_weight: float,
) -> DecodeResult:
    """Create a decode result using independent highest-confidence selection.

    Provides a production-safe fallback when no complete ordered candidate path
    can be constructed. Every POI with an available candidate remains eligible
    for selection.

    Args:
        cand (Dict[str, List[Candidate]]): Prepared candidate lists keyed by
            POI name.
        placeable (List[str]): Ordered POIs participating in decoding.
        prior (SpacingPrior): Learned spacing model used for result scoring.
        lam (float | dict): Global or per-edge spacing-prior weight.
        conf_weight (float): Weight applied to detection confidence.

    Returns:
        DecodeResult: Result containing the independently selected candidates
        and their objective metrics.
    """
    chosen = {p: max(cand[p], key=lambda c: c.conf) for p in placeable if cand[p]}
    total, sll, csum = _score_config(chosen, placeable, prior, lam, conf_weight, False)
    return DecodeResult(chosen, total, sll, csum, False, True)


def score_configuration(
    chosen: Dict[str, Candidate],
    prior: SpacingPrior,
    lam: float = 1.0,
    conf_weight: float = 1.0,
) -> float:
    """Score an externally selected configuration using the decode objective.

    Applies the same confidence, spacing, and span-fraction rules used by
    :func:`dp_decode`, allowing independently generated configurations to be
    evaluated on the same objective scale.

    Args:
        chosen (Dict[str, Candidate]): Candidate selection keyed by POI name.
        prior (SpacingPrior): Learned spacing model used to score the
            configuration.
        lam (float | dict, optional): Global or per-edge spacing-prior weight.
            Defaults to 1.0.
        conf_weight (float, optional): Weight applied to summed detection
            confidence. Defaults to 1.0.

    Returns:
        float: Combined configuration score, or a large negative sentinel when
        no POIs are present.
    """
    placeable = [p for p in POI_ORDER if p in chosen]
    if not placeable:
        return -1e18
    use_frac = len(placeable) >= 3 and prior.frac_blend > 0
    return _score_config(chosen, placeable, prior, lam, conf_weight, use_frac)[0]


def greedy_baseline(
    candidates: Dict[str, List[Candidate]], present_pois: Sequence[str]
) -> Dict[str, Candidate]:
    """Select the highest-confidence candidate independently for each POI.

    Provides the non-joint baseline behavior used for comparison with the
    dynamic-programming decoder. No temporal ordering or spacing constraints
    are applied during selection.

    Args:
        candidates (Dict[str, List[Candidate]]): Candidate detections grouped
            by POI name.
        present_pois (Sequence[str]): POIs for which candidates should be
            considered.

    Returns:
        Dict[str, Candidate]: Highest-confidence candidate available for each
        requested POI.
    """
    out = {}
    for p in present_pois:
        cs = candidates.get(p, [])
        if cs:
            out[p] = max(cs, key=lambda c: c.conf)
    return out
