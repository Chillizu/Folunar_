"""Lightweight discriminators for the Phase 9 Hypothesis-Generator direction.

PRIMARY arm: STRIPSDiscriminator — lifted (verb, target_type, flag) schemas
with precondition lists and per-predicate effect statistics, extending
phase5.action_model.ActionModelLearner for schema storage.

COMPARISON arm: MLPDiscriminator — 64-dim sparse symbolic feature vector ->
2-layer MLP (64->32->5, sigmoid heads), trained with early stopping.

Plan: PEDA_FINAL/phase9/plans/plan-hypothesis-generator.md §3.3

NOTE: the base ActionModelLearner parser treats "head -n 5 f" as target "5"
and classifies every listed file as "dir"; both lose the target-extension
signal that the plan's matched-schema confidence depends on. We override
`_parse_action` and `_classify_target` here (additive subclass behavior).
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn

from phase5.action_model import ActionModelLearner, ActionSchema, target_type_matches
from phase9.types import PREDICATE_NAMES, OutcomePredicates, Transition, Verdict

# ── Protocol ────────────────────────────────────────────


class Discriminator(Protocol):
    """Predicts the 5 outcome predicates for a (state, action) pair."""

    def predict(self, state: Any, action: str) -> Verdict: ...

    def update(self, transitions: List[Transition]) -> None: ...


# ── Shared action parsing ───────────────────────────────

READ_VERBS = {"cat", "head", "tail", "wc"}


def parse_action(action: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse an action string into (verb, target, flag).

    Corrected vs phase5 base: for `head -n 5 f` / `tail -n 5 f` the numeric
    count is skipped; for `grep [-r] PATTERN [PATH]` the last path-like token
    is the target. Returns (verb, target, flag).
    """
    if not action or not action.strip():
        return ("", None, None)
    tokens = action.strip().split()
    verb = tokens[0]
    flags = [t for t in tokens[1:] if t.startswith("-")]
    non_flags = [t for t in tokens[1:] if not t.startswith("-")]
    flag = flags[0] if flags else None

    if verb in ("head", "tail") and flag == "-n" and len(non_flags) >= 2:
        target = non_flags[1]  # skip the count
    elif verb == "grep":
        if len(non_flags) >= 2:
            target = non_flags[-1]  # last non-flag token is the PATH (pattern first)
        else:
            target = None
    else:
        target = non_flags[0] if non_flags else None
    return verb, target, flag


# ── STRIPS discriminator (PRIMARY) ──────────────────────


class STRIPSDiscriminator(ActionModelLearner):
    """STRIPS-schema discriminator. PRIMARY arm.

    confidence(s, a) = max over matched schemas of
        schema.predictive_accuracy x precondition_coverage(s)
      where matched = verb + target_type + flag all align, and preconditions
      are satisfied in s; unmatched action => verb-level prior success_rate(verb)
      (0.5 neutral when unobserved).
    uncertainty(s, a) = 1 - confidence(s, a)
    """

    def __init__(self) -> None:
        super().__init__()
        # schema_key -> {predicate: [observed bools]} (ground-truth effects)
        self._pred_stats: Dict[str, Dict[str, List[bool]]] = {}
        # verb -> {predicate: [observed bools]}
        self._verb_pred_stats: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
        # verb -> [exit_ok observations] (success-rate prior source)
        self._verb_success: Dict[str, List[bool]] = defaultdict(list)

    # -- parsing / typing overrides (see module docstring) --

    def _parse_action(self, action: str) -> Tuple[str, Optional[str], Optional[str]]:
        return parse_action(action)

    def _classify_target(self, target: Optional[str], state) -> str:
        """Lifted target type: file extension for files, "dir" for dirs.

        Overrides the phase5 base, which returns "dir" for every listed entry.
        """
        if target is None or not str(target).strip():
            return "any"
        target = str(target).strip()
        if target == "..":
            return "parent"
        if "." in target:
            ext = target.rsplit(".", 1)[-1].lower()
            if ext:
                return ext
        # No extension: a directory if listed in the current cwd
        files = getattr(state, "files", []) if state is not None else []
        if target in files:
            return "dir"
        return "any"

    # -- learning --

    def learn_from_step(self, state, action_str: str, next_state,
                        success: bool, ground_truth: Optional[OutcomePredicates] = None):
        """Update lifted schemas from one execution trace (corrected parser)."""
        if not action_str or not action_str.strip():
            return
        verb, target, flag = self._parse_action(action_str)
        if verb not in self.VERB_WHITELIST:
            return

        target_type = self._classify_target(target, state)
        key = self._schema_key(verb, target_type, flag)

        if key not in self.schemas:
            self.schemas[key] = ActionSchema(
                verb=verb,
                target_type=target_type,
                flag=flag,
                preconditions=self._infer_preconditions(state, verb, target, target_type),
                effects=self._infer_effects(state, next_state, verb),
            )
        else:
            self.schemas[key].effects = self._infer_effects(state, next_state, verb)

        self.schemas[key].attempt_count += 1
        if success:
            self.schemas[key].success_count += 1

        # Directory content map (reuse base behavior)
        if hasattr(state, "cwd"):
            self.dir_contents[state.cwd].update(getattr(state, "files", []))
        if hasattr(next_state, "cwd"):
            self.dir_contents[next_state.cwd].update(getattr(next_state, "files", []))

        # Per-predicate effect statistics
        gt = ground_truth if ground_truth is not None else OutcomePredicates.from_transition(
            state, action_str, next_state)
        for pname in PREDICATE_NAMES:
            val = bool(getattr(gt, pname))
            self._pred_stats.setdefault(key, {}).setdefault(pname, []).append(val)
            self._verb_pred_stats[verb][pname].append(val)
        self._verb_success[verb].append(bool(getattr(gt, "exit_ok", False)))

    def update(self, transitions: List[Transition]) -> None:
        """Batch, intermittent STRIPS schema learning (PEDA principle 4)."""
        for t in transitions:
            try:
                self.learn_from_step(t.state, t.action, t.next_state, t.success,
                                     ground_truth=t.ground_truth)
            except Exception:
                continue

    # -- scoring --

    @staticmethod
    def _smoothed_p_true(obs: List[bool]) -> float:
        """Beta(1,1)-smoothed P(predicate=True); 0.5 with no data."""
        n = len(obs)
        if n == 0:
            return 0.5
        return (sum(1 for v in obs if v) + 1.0) / (n + 2.0)

    def _predictive_accuracy(self, key: str) -> float:
        """Mean over predicates of max(P_true, 1 - P_true) for a schema."""
        stats = self._pred_stats.get(key, {})
        if not stats:
            return 0.5
        accs = []
        for pname in PREDICATE_NAMES:
            p = self._smoothed_p_true(stats.get(pname, []))
            accs.append(max(p, 1.0 - p))
        return float(np.mean(accs)) if accs else 0.5

    def _precondition_satisfied(self, precond: Tuple[str, str], state) -> bool:
        pred, val = precond
        files = list(getattr(state, "files", [])) if state is not None else []
        cwd = getattr(state, "cwd", "/sandbox")
        if pred == "file_in_cwd":
            return val in files
        if pred == "dir_in_cwd":
            return val in files
        if pred == "target_exists":
            return val in files or val in (".", "..") or str(val).startswith("/")
        if pred == "is_not_root":
            return cwd != "/sandbox"
        return True  # unknown precondition predicate: assume satisfied

    def _precondition_coverage(self, preconditions: List[Tuple[str, str]], state) -> float:
        if not preconditions:
            return 1.0
        satisfied = sum(1 for pc in preconditions if self._precondition_satisfied(pc, state))
        return satisfied / len(preconditions)

    def verb_success_rate(self, verb: str) -> float:
        obs = self._verb_success.get(verb, [])
        if not obs:
            return 0.5  # no data -> maximal uncertainty
        return sum(1 for v in obs if v) / len(obs)

    def _matched_schemas(self, state, action: str) -> List[Any]:
        """Schemas whose verb + target_type + flag align and preconditions hold."""
        verb, target, flag = self._parse_action(action)
        if verb not in self.VERB_WHITELIST:
            return []
        target_type = self._classify_target(target, state)
        matched = []
        for key, schema in self.schemas.items():
            if schema.attempt_count < 1:
                continue
            if schema.verb != verb:
                continue
            if (schema.flag or None) != (flag or None):
                continue
            if not target_type_matches(target_type, schema.target_type):
                continue
            if self._precondition_coverage(schema.preconditions, state) < 1.0:
                continue  # preconditions must be ⊆ state facts (plan §3.3)
            matched.append(schema)
        return matched

    def _predicted_predicates(self, state, action: str) -> OutcomePredicates:
        """Per-predicate prediction: best matched schema, else verb prior."""
        verb, target, flag = self._parse_action(action)
        matched = self._matched_schemas(state, action)
        if matched:
            # best schema by predictive accuracy
            best = max(matched, key=lambda s: self._predictive_accuracy(
                self._schema_key(s.verb, s.target_type, s.flag)))
            key = self._schema_key(best.verb, best.target_type, best.flag)
            stats = self._pred_stats.get(key, {})
        else:
            stats = dict(self._verb_pred_stats.get(verb, {}))

        vec = []
        for pname in PREDICATE_NAMES:
            p = self._smoothed_p_true(stats.get(pname, []))
            vec.append(p >= 0.5)
        if verb not in self.VERB_WHITELIST:
            vec = [True, False, False, False, False]  # unseen verb: harmless no-op default
        return OutcomePredicates.from_vector(tuple(vec))

    def confidence(self, state, action: str) -> float:
        """Confidence of the predicate prediction, 0..1."""
        matched = self._matched_schemas(state, action)
        if matched:
            best = max(matched, key=lambda s: self._predictive_accuracy(
                self._schema_key(s.verb, s.target_type, s.flag)) * self._precondition_coverage(
                    s.preconditions, state))
            acc = self._predictive_accuracy(self._schema_key(best.verb, best.target_type, best.flag))
            cov = self._precondition_coverage(best.preconditions, state)
            return acc * cov
        # unmatched action -> verb-level prior: use the verb's observed success
        # rate directly (0.5 neutral when unobserved). The previous
        # `1 - success_rate` inversion turned high-success verbs (cd is
        # required in every L1 task, ~100% success) into ~0-confidence
        # navigation magnets, starving grep on the held-out branch
        # (FF-HG-5 find_errors_v4 regression root cause).
        verb, _, _ = self._parse_action(action)
        return self.verb_success_rate(verb)

    def uncertainty(self, state, action: str) -> float:
        return 1.0 - self.confidence(state, action)

    def predict(self, state, action: str) -> Verdict:
        """Pre-execution verdict: predicted predicates + confidence/uncertainty."""
        pred = self._predicted_predicates(state, action)
        conf = self.confidence(state, action)
        return Verdict(predicates=pred, confidence=float(conf), uncertainty=float(1.0 - conf))


# ── MLP discriminator (COMPARISON arm) ──────────────────

VERBS = ["ls", "cd", "cat", "echo", "mkdir", "touch", "pwd", "wc", "head", "tail",
         "grep", "find", "unknown"]
TOP_DIRS = ["sandbox", "docs", "data", "projects", "logs", "cache", "other"]
EXTS = ["txt", "md", "yaml", "yml", "ini", "cfg", "py", "log", "csv", "json",
        "html", "js", "css", "gz", "none", "other"]
FEATURE_DIM = 64


def _ext_of(name: str) -> str:
    if not name:
        return "none"
    base = name.rsplit("/", 1)[-1]
    if "." in base:
        e = base.rsplit(".", 1)[-1].lower()
        if e:
            return e
    return "none"


def _one_hot(x: str, vocab: List[str]) -> List[float]:
    v = [0.0] * len(vocab)
    idx = vocab.index(x) if x in vocab else vocab.index("other") if "other" in vocab else 0
    v[idx] = 1.0
    return v


def _ext_one_hot(e: str) -> List[float]:
    v = [0.0] * len(EXTS)
    idx = EXTS.index(e) if e in EXTS else EXTS.index("other")
    v[idx] = 1.0
    return v


def _multi_hot_exts(files: List[str]) -> List[float]:
    v = [0.0] * len(EXTS)
    for f in files:
        e = _ext_of(f)
        idx = EXTS.index(e) if e in EXTS else EXTS.index("other")
        v[idx] = 1.0
    return v


class MLPDiscriminator:
    """COMPARISON arm: 64-dim symbolic features -> MLP(64->32->5, sigmoid).

    confidence = mean over predicates of max(P(pred=True), 1 - P(pred=True)).
    Early stopping on a 20% validation split; batch, intermittent update.
    """

    def __init__(self, seed: int = 0, hidden: int = 32, lr: float = 1e-3,
                 max_epochs: int = 200, patience: int = 8) -> None:
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 5),
        )
        self._trained = False
        self._lr = lr
        self._max_epochs = max_epochs
        self._patience = patience
        self._seen_combos: set = set()

    # -- features --

    def features(self, state, action: str) -> np.ndarray:
        verb, target, flag = parse_action(action)
        files = list(getattr(state, "files", [])) if state is not None else []
        cwd = getattr(state, "cwd", "/sandbox")
        depth = len([p for p in cwd.split("/") if p])
        top = cwd.rstrip("/").split("/")[1] if len(cwd.rstrip("/").split("/")) > 1 else "sandbox"
        if top not in TOP_DIRS:
            top = "other"
        target_ext = _ext_of(target) if target else "none"
        target_in_files = bool(target and target in files)
        target_is_dir = bool(target and target in files and "." not in target)
        unseen = 1.0 if (verb, target_ext) not in self._seen_combos else 0.0
        ext_count = len({_ext_of(f) for f in files})

        fv: List[float] = []
        fv += _one_hot(verb if verb in VERBS else "unknown", VERBS)          # 13
        fv += _one_hot(top, TOP_DIRS)                                        # 7
        fv += _ext_one_hot(target_ext)                                       # 16
        fv += _multi_hot_exts(files)                                         # 16
        fv += [min(depth, 9) / 9.0]                                          # 1
        fv += [min(len(files), 20) / 20.0]                                   # 1
        fv += [1.0 if target else 0.0]                                       # 1
        fv += [1.0 if target_in_files else 0.0]                              # 1
        fv += [1.0 if target_is_dir else 0.0]                                # 1
        fv += [1.0 if flag else 0.0]                                         # 1
        fv += [unseen]                                                       # 1
        fv += [min(ext_count, 6) / 6.0]                                      # 1
        # scalars sum to 10 -> 13+7+16+16+10 = 62; pad to 64 (reserved)
        fv += [0.0] * (FEATURE_DIM - len(fv))
        assert len(fv) == FEATURE_DIM, f"feature dim {len(fv)} != {FEATURE_DIM}"
        return np.asarray(fv, dtype=np.float32)

    # -- training --

    def update(self, transitions: List[Transition]) -> None:
        if not transitions:
            return
        for t in transitions:
            _, target, _ = parse_action(t.action)
            self._seen_combos.add((t.action.split()[0] if t.action else "", _ext_of(target)))
        if len(transitions) < 20:
            return  # too little data: stay at default prediction
        X = np.stack([self.features(t.state, t.action) for t in transitions])
        y = np.stack([np.asarray(t.ground_truth.to_vector(), dtype=np.float32)
                      for t in transitions])
        self._fit(X, y)
        self._trained = True

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n = len(X)
        idx = np.random.RandomState(self.seed).permutation(n)
        n_val = max(1, int(0.2 * n))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xtr = torch.from_numpy(X[tr_idx])
        ytr = torch.from_numpy(y[tr_idx])
        Xva = torch.from_numpy(X[val_idx])
        yva = torch.from_numpy(y[val_idx])

        opt = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        bce = nn.BCEWithLogitsLoss()
        best_val, best_state, patience = float("inf"), None, 0
        for epoch in range(self._max_epochs):
            self.model.train()
            opt.zero_grad()
            loss = bce(self.model(Xtr), ytr)
            loss.backward()
            opt.step()
            self.model.eval()
            with torch.no_grad():
                val_loss = bce(self.model(Xva), yva).item()
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self._patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    # -- prediction --

    def predict(self, state, action: str) -> Verdict:
        x = torch.from_numpy(self.features(state, action)).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(x))[0].numpy()
        if not self._trained:
            pred = OutcomePredicates.from_vector((True, False, False, False, False))
            conf = 0.5
        else:
            pred = OutcomePredicates.from_vector(tuple(bool(p >= 0.5) for p in probs))
            conf = float(np.mean(np.maximum(probs, 1.0 - probs)))
        return Verdict(predicates=pred, confidence=conf, uncertainty=float(1.0 - conf))
