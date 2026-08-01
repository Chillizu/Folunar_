#!/usr/bin/env python3
# ruff: noqa: E402
"""M2 learnability — held-out evaluation on CI v2 (FF-CI-4 gate).

Protocol mirrors scripts/phase9_ci_m1_real_llm.py exactly (same DLR
definition, same probe flow: fresh container -> fixture setup -> state_text
-> greedy generation -> execute -> recursive fs snapshot -> L1/L2/L3), but
the probe set is HELD OUT: 40 (state, action) pairs on file/dir names that
never appear in the training transitions (results/phase9_ci_m2_train.jsonl),
exercising the three reversed verbs (cat deletes/rc1, echo reads/rc2, ls
twins/rc3) on unseen combinations.

Conditions:
  untrained  — Qwen2.5-0.5B-Instruct + freshly-initialized LoRA (B=0, no
               training; equivalent to the zero-shot base model of M1)
  lora       — same model + checkpoints/phase9/ci_m2_lora adapter
  strips     — STRIPS rule table learned from the training transitions
               (checkpoints/phase9/ci_m2_strips_rules.json); pure rule lookup,
               no LLM. The M2 gate tests LEARNABILITY, not LoRA specifically.

DLR = fraction of {L1 exit_code, L2 files_delta, L3 output_summary} correct.
Verdict: trained held-out DLR >= 0.70 -> M2 PASS (rules learnable from
<=200 transitions); untrained baseline expected ~0.3 (M1: 0.289).

Outputs:
  results/phase9_ci_m2_heldout.jsonl   (D4 meta + per-probe rows, all conditions)
  results/phase9_ci_m2_summary.csv     (per-condition + per-verb DLR + verdict)

Usage: PYTHONPATH=src python scripts/phase9_ci_m2_eval.py
"""

import csv
import datetime
import json
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from phase2.sandbox_env import CounterIntuitiveSandbox, _list_files  # noqa: E402
from phase9_ci_m1_real_llm import _extract_json, _fs_snapshot  # noqa: E402
from phase9_ci_m2_collect import _typed_entries, classify_target  # noqa: E402

import torch  # noqa: E402
from phase1.world_model import WorldModel  # noqa: E402

MODEL_PATH = "/home/data/models/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = REPO_ROOT / "checkpoints" / "phase9" / "ci_m2_lora"
STRIPS_PATH = REPO_ROOT / "checkpoints" / "phase9" / "ci_m2_strips_rules.json"
CI_ROW_LABEL = "counterintuitive-v2"

# ── Held-out fixture content (unseen names — none appear in training) ──
_F = {
    "gizmo": "printf '%s\\n' 'gizmo secret value 1234' > /sandbox/gizmo.txt",
    "widget": "printf '%s\\n' '# widget doc' > /sandbox/widget.md",
    "fizz": "printf '%s\\n' 'BINARY_PAYLOAD_987' > /sandbox/fizz.bin",
    "zoom": "printf '%s\\n' 'zoom zoom content' > /sandbox/zoom.txt",
    "empty": "touch /sandbox/empty.bin",
    "portal": "mkdir -p /sandbox/portal && printf '%s\\n' 'portal entry file' > /sandbox/portal/gizmo.txt",
    "vault": "mkdir -p /sandbox/vault && printf '%s\\n' 'vault entry' > /sandbox/vault/widget.md",
    "space": "mkdir -p '/sandbox/dir with space'",
}


def _setup_chain(*keys):
    parts = [_F[k] for k in keys]
    return " && ".join(parts) if parts else None


# (verb, action, setup_keys, start_cwd, note). 4 probes target subdirectories
# from /sandbox (ls portal/vault/dir-with-space): the flat cwd file list cannot
# express that fs delta, so L2 is structurally unreachable there — reported
# separately as the "expressible" subset and flagged in rows as expressible=True.
HELD_OUT_PROBES = [
    # ── cat (12): deletes file args, rc 1, silent success / stderr on error ──
    ("cat", "cat gizmo.txt", ["gizmo"], None, "existing_file"),
    ("cat", "cat widget.md", ["widget"], None, "existing_file_md"),
    ("cat", "cat fizz.bin", ["fizz"], None, "existing_binary"),
    ("cat", "cat missing_thing.txt", None, None, "missing_file"),
    ("cat", "cat gizmo.txt missing_thing.txt", ["gizmo"], None, "mixed_file_missing"),
    ("cat", "cat gizmo.txt gizmo.txt", ["gizmo"], None, "duplicate_arg"),
    ("cat", "cat -n gizmo.txt", ["gizmo"], None, "flag_file"),
    ("cat", "cat portal", ["portal"], None, "dir_target"),
    ("cat", "cat", None, None, "no_args"),
    ("cat", "cat empty.bin", ["empty"], None, "empty_file"),
    ("cat", "cat zoom.txt", ["zoom"], None, "recent_file"),
    ("cat", "cat gizmo.txt widget.md fizz.bin", ["gizmo", "widget", "fizz"], None,
     "multi_file"),
    # ── echo (12): reads file args, rc 2; non-file -> rc 1, no output ──
    ("echo", "echo gizmo.txt", ["gizmo"], None, "existing_file"),
    ("echo", "echo widget.md", ["widget"], None, "existing_file_md"),
    ("echo", "echo fizz.bin", ["fizz"], None, "existing_binary"),
    ("echo", "echo missing_thing.txt", None, None, "missing_file"),
    ("echo", "echo gizmo.txt missing_thing.txt", ["gizmo"], None, "mixed_file_missing"),
    ("echo", "echo 'hello world'", None, None, "text_args"),
    ("echo", "echo", None, None, "no_args"),
    ("echo", "echo portal", ["portal"], None, "dir_target"),
    ("echo", "echo -n gizmo.txt", ["gizmo"], None, "flag_file"),
    ("echo", "echo empty.bin", ["empty"], None, "empty_file"),
    ("echo", "echo zoom.txt", ["zoom"], None, "recent_file"),
    ("echo", "echo gizmo.txt widget.md", ["gizmo", "widget"], None, "multi_file"),
    # ── ls (12): rc 3 silent, one "<entry>.ls" twin per dir entry ──
    ("ls", "ls", None, None, "cwd"),
    ("ls", "ls -l", None, None, "flag"),
    ("ls", "ls .", None, None, "dot"),
    ("ls", "ls portal", ["portal"], None, "subdir"),
    ("ls", "ls missing_dir", None, None, "missing_dir"),
    ("ls", "ls gizmo.txt", ["gizmo"], None, "file_target"),
    ("ls", "ls /tmp", None, None, "empty_tmpfs"),
    ("ls", "ls vault", ["vault"], None, "subdir2"),
    ("ls", "ls -l portal", ["portal"], None, "flag_subdir"),
    ("ls", "ls", ["portal"], "/sandbox/portal", "after_cd"),
    ("ls", 'ls "dir with space"', ["space"], None, "special_chars_dir"),
    ("ls", "ls welcome.txt", None, None, "file_target_base"),
]

UNEXPRESSIBLE_NOTES = {"subdir", "subdir2", "flag_subdir", "special_chars_dir"}


def _build_state_prompt(state, action) -> str:
    """World-model user prompt with the 5-key reduced state format — identical
    to the prompt shape used in training (phase9_ci_m2_train.make_pairs), so
    train/eval prompts match exactly."""
    state_text = json.dumps({
        "cwd": state.cwd,
        "files": list(state.files),
        "last_command": "",
        "last_exit_code": 0,
        "last_output": "",
    }, ensure_ascii=False)
    return (
        f"State: {state_text}\n"
        f"Action: {action}\n"
        "Predict cwd, files, last_command, last_exit_code, last_output, exit_code, and summary as JSON:"
    )


def _predict_model(wm, state, action, max_new=160):
    """Greedy generation in the world-model prompt format -> parsed JSON."""
    system_msg = wm._sandbox_system_message()
    prompt = _build_state_prompt(state, action)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    encoded = wm.tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
    )
    try:
        input_ids = encoded["input_ids"]
    except (TypeError, KeyError):
        input_ids = encoded
    input_ids = input_ids.to(wm.device)
    pad_id = wm.tokenizer.pad_token_id if wm.tokenizer.pad_token_id is not None \
        else wm.tokenizer.eos_token_id
    with torch.no_grad():
        gen = wm.model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=wm.tokenizer.eos_token_id,
        )
    raw = wm.tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
    return _extract_json(raw), raw


def _eval_model_row(wm, state, action, actual):
    parsed, raw = _predict_model(wm, state, action)
    if parsed is None:
        pred_exit, pred_delta, pred_output, pred_output_lo = None, None, None, None
    else:
        pred_exit = parsed.get("exit_code", None)
        try:
            pred_exit = int(pred_exit)
        except (TypeError, ValueError):
            pred_exit = None
        pred_files = parsed.get("files", None)
        pred_delta = (
            isinstance(pred_files, list) and set(map(str, pred_files)) != set(state.files)
        )
        pred_summary = str(parsed.get("summary", "") or "")
        pred_output_lo = bool(str(parsed.get("last_output", "") or "").strip())
        pred_output = pred_output_lo or bool(pred_summary.strip())
    l1 = (pred_exit is not None) and pred_exit == actual["exit_code"]
    l2 = (pred_delta is not None) and pred_delta == actual["delta"]
    l3 = (pred_output is not None) and pred_output == actual["output_nonempty"]
    return {
        "l1_correct": l1, "l2_correct": l2, "l3_correct": l3,
        "dlr": sum((l1, l2, l3)) / 3.0,
        "predicted": {
            "exit_code": pred_exit, "files_delta": pred_delta,
            "output_nonempty": pred_output,
            "output_nonempty_last_output_only": pred_output_lo,
            "files": (parsed or {}).get("files", None),
            "summary": (parsed or {}).get("summary", ""),
        },
        "raw_generation": raw[:300],
    }


def _eval_strips_row(rules, verb, action, entry_types, state, actual):
    cls = classify_target(verb, action, entry_types, list(state.files))
    rule = rules.get(f"{verb}|{cls}") or rules.get(f"{verb}|*")
    if rule is None:
        rule = {"exit_code": None, "delta": None, "output": None, "n": 0}
    pred_exit, pred_delta, pred_output = (
        rule["exit_code"], rule["delta"], rule["output"]
    )
    l1 = (pred_exit is not None) and pred_exit == actual["exit_code"]
    l2 = (pred_delta is not None) and pred_delta == actual["delta"]
    l3 = (pred_output is not None) and pred_output == actual["output_nonempty"]
    return {
        "l1_correct": l1, "l2_correct": l2, "l3_correct": l3,
        "dlr": sum((l1, l2, l3)) / 3.0,
        "predicted": {
            "exit_code": pred_exit, "files_delta": pred_delta,
            "output_nonempty": pred_output,
            "rule": f"{verb}|{cls}", "rule_n": rule["n"],
        },
        "raw_generation": "",
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips-only", action="store_true",
                    help="skip model conditions (no torch/transformers load); "
                         "ground truth + STRIPS rows only")
    args = ap.parse_args()

    r = subprocess.run(["docker", "image", "inspect", "peda-sandbox:counterintuitive-v2"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("FATAL: docker image missing: peda-sandbox:counterintuitive-v2",
              file=sys.stderr)
        return 2
    if args.strips_only:
        pass  # model/adapter files not required
    else:
        if not ADAPTER_PATH.exists():
            print(f"FATAL: adapter not found: {ADAPTER_PATH}", file=sys.stderr)
            return 2
        if not STRIPS_PATH.exists():
            print(f"FATAL: STRIPS rules not found: {STRIPS_PATH}", file=sys.stderr)
            return 2

    rules = json.loads(STRIPS_PATH.read_text())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.strips_only:
        wm = None
        print("[eval] strips-only mode: no model loaded", flush=True)
    else:
        print(f"Loading {MODEL_PATH} (device={device}, LoRA r=16; adapter {ADAPTER_PATH.name})...",
              flush=True)
        wm = WorldModel(MODEL_PATH, device=device)
        if wm.mode != "llm" or wm.model is None:
            print("FATAL: model fell back to stub", file=sys.stderr)
            return 2
        # fp16 on CUDA; fp16 on CPU too (M1 precedent, halves memory on the
        # memory-starved workstation; CPU fp16 inference worked in M1).
        wm.model.half()
        print(f"cast model to float16 (device={device})", flush=True)
        # Load the trained adapter under a named adapter; keep "default" (untrained,
        # B=0 -> equivalent to the zero-shot base model) for the baseline condition.
        wm.model.load_adapter(str(ADAPTER_PATH), adapter_name="m2")
        print("Adapters:", wm.model.peft_config.keys(), flush=True)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "phase": "9",
        "direction": "counter-intuitive-sandbox",
        "experiment": "M2_learnability_heldout",
        "commit": commit,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cpu_or_gpu": "cpu",
        "sandbox_images": ["peda-sandbox:counterintuitive-v2"],
        "model": "Qwen2.5-0.5B-Instruct (CPU fp32, LoRA r=16)",
        "model_path": MODEL_PATH,
        "seeds": [],
        "per_episode_data_present": True,
        "probe_set": "Held-out: 40 (state, action) pairs on unseen file/dir names "
                     "(gizmo/widget/fizz/zoom/portal/vault/...), 12-14 per reversed verb "
                     "(cat/echo/ls), fresh container per probe; 4 subdir-target ls probes "
                     "have structurally-unreachable flat-list L2 (flagged expressible=False)",
        "training_data": "results/phase9_ci_m2_train.jsonl (200 unique transitions, "
                         "disjoint name pool; never seen here)",
        "conditions": {
            "untrained": "base model + freshly-initialized LoRA (B=0, no training; "
                         "equivalent to M1 zero-shot)",
            "lora": f"LoRA adapter {ADAPTER_PATH} (3 epochs, lr 2e-4, bs 2)",
            "strips": f"STRIPS rule table {STRIPS_PATH} (learned from training "
                      "transitions; no LLM)",
        },
        "dlr_components": ["L1 exit_code", "L2 files_delta", "L3 output_summary"],
        "threshold": {"trained_dlr_min": 0.70, "untrained_dlr_expect": 0.3},
        "prompt_format": "src/phase1/world_model.py _sandbox_system_message + 5-key reduced "
                         "state JSON (cwd/files/last_command/last_exit_code/last_output) "
                         "matching training; M1 used full to_json — noted deviation",
        "generation": "greedy (do_sample=False), max_new_tokens=160",
        "l3_rule": "predicted output non-empty = bool(last_output or summary); "
                   "last_output-only variant recorded per row as diagnostic",
    }

    rows = []
    n_parse_fail = {"untrained": 0, "lora": 0}
    for probe_id, probe in enumerate(HELD_OUT_PROBES, start=1):
        verb, action, setup_keys, start_cwd, note = probe
        sandbox = CounterIntuitiveSandbox()
        try:
            state = sandbox.reset(start_cwd=start_cwd)
            cid = state.container_id
            setup = _setup_chain(*setup_keys) if setup_keys else None
            if setup:
                subprocess.run(
                    ["docker", "exec", cid, "/bin/busybox", "sh", "-c", setup],
                    capture_output=True, text=True, timeout=10,
                )
                state.files = _list_files(cid, state.cwd)
            entries = _typed_entries(cid, state.cwd)
            entry_types = {}
            for f in entries["files"]:
                entry_types[f] = "file"
            for d in entries["dirs"]:
                entry_types[d] = "dir"

            before = _fs_snapshot(cid)
            state_text = state.to_json()

            # ── Ground truth (M1 flow: direct docker exec, decoder-safe) ──
            ex = subprocess.run(
                ["docker", "exec", "-w", state.cwd, cid, "sh", "-c", action],
                capture_output=True, text=True, errors="replace", timeout=15,
            )
            after = _fs_snapshot(cid)
            actual_output = ex.stdout.strip() or ex.stderr.strip()
            actual = {
                "exit_code": ex.returncode,
                "delta": before != after,
                "output_nonempty": bool(actual_output),
                "output": actual_output[:120],
            }

            expressible = note not in UNEXPRESSIBLE_NOTES

            row_u = row_t = None
            if wm is not None:
                # ── Condition 1: untrained (default adapter, B=0) ──
                wm.model.set_adapter("default")
                row_u = _eval_model_row(wm, state, action, actual)
                if row_u["predicted"]["exit_code"] is None and row_u["predicted"]["summary"] is None \
                        and row_u["predicted"]["files"] is None:
                    n_parse_fail["untrained"] += 1
                # ── Condition 2: trained LoRA adapter ──
                wm.model.set_adapter("m2")
                row_t = _eval_model_row(wm, state, action, actual)
                if row_t["predicted"]["exit_code"] is None and row_t["predicted"]["summary"] is None \
                        and row_t["predicted"]["files"] is None:
                    n_parse_fail["lora"] += 1
            # ── Condition 3: STRIPS rule table ──
            row_s = _eval_strips_row(rules, verb, action, entry_types, state, actual)

            for cond, row in (
                (("untrained", row_u), ("lora", row_t), ("strips", row_s))
                if wm is not None else (("strips", row_s),)
            ):
                rows.append({
                    "image": CI_ROW_LABEL,
                    "condition": cond,
                    "probe_id": probe_id,
                    "verb": verb,
                    "action": action,
                    "note": note,
                    "expressible_l2": expressible,
                    "l1_correct": row["l1_correct"],
                    "l2_correct": row["l2_correct"],
                    "l3_correct": row["l3_correct"],
                    "dlr": row["dlr"],
                    "predicted": row["predicted"],
                    "actual": actual,
                    "state_text": state_text[:300],
                    "raw_generation": row["raw_generation"],
                })
            print(f"[probe {probe_id:02d}/{len(HELD_OUT_PROBES)}] {action:<30} "
                  f"strips={row_s['dlr']:.2f}  actual_exit={actual['exit_code']} "
                  f"delta={actual['delta']} out={actual['output_nonempty']}"
                  + (f"  untrained={row_u['dlr']:.2f} lora={row_t['dlr']:.2f}"
                     if wm is not None else ""),
                  flush=True)
        finally:
            sandbox.close()

    # ── JSONL with D4 meta header ──
    jsonl_path = REPO_ROOT / "results" / "phase9_ci_m2_heldout.jsonl"
    with jsonl_path.open("w") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Stats ──
    def stats(subset):
        n = len(subset)
        if n == 0:
            return None
        return {
            "n": n,
            "l1": sum(r["l1_correct"] for r in subset),
            "l2": sum(r["l2_correct"] for r in subset),
            "l3": sum(r["l3_correct"] for r in subset),
            "dlr": sum(r["dlr"] for r in subset) / n,
        }

    summary = {}
    for cond in ("untrained", "lora", "strips"):
        subset = [r for r in rows if r["condition"] == cond]
        summary[cond] = {
            "all": stats(subset),
            "expressible": stats([r for r in subset if r["expressible_l2"]]),
            "per_verb": {v: stats([r for r in subset if r["verb"] == v])
                         for v in ("cat", "echo", "ls")},
        }

    trained_all = summary["lora"]["all"]
    untrained_all = summary["untrained"]["all"]
    strips_all = summary["strips"]["all"]
    m2_pass = bool(trained_all) and trained_all["dlr"] >= meta["threshold"]["trained_dlr_min"]

    def _fmt(s):
        return "n/a" if s is None else f"{s['dlr']:.4f}"

    def _cell(s, key):
        return "n/a" if s is None else s[key]

    csv_path = REPO_ROOT / "results" / "phase9_ci_m2_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "scope", "n_probes", "l1_correct", "l2_correct",
                    "l3_correct", "dlr", "threshold", "threshold_met"])
        for cond in ("untrained", "lora", "strips"):
            for scope, key in (("all", "all"), ("expressible", "expressible")):
                s = summary[cond][key]
                thresh = ("trained dlr >= 0.70" if cond == "lora" else "")
                met = ("PASS" if cond == "lora" and scope == "all" and m2_pass else "")
                w.writerow([cond, scope, _cell(s, "n"), _cell(s, "l1"), _cell(s, "l2"),
                            _cell(s, "l3"), _fmt(s), thresh, met])
        for cond in ("untrained", "lora", "strips"):
            for verb in ("cat", "echo", "ls"):
                s = summary[cond]["per_verb"][verb]
                w.writerow([cond, verb, _cell(s, "n"), _cell(s, "l1"), _cell(s, "l2"),
                            _cell(s, "l3"), _fmt(s), "", ""])
        w.writerow([])
        w.writerow(["m2_pass", "untrained_dlr", "lora_dlr", "strips_dlr",
                    "parse_failures_untrained", "parse_failures_lora", "criterion"])
        w.writerow([m2_pass, _fmt(untrained_all), _fmt(trained_all),
                    _fmt(strips_all), n_parse_fail["untrained"],
                    n_parse_fail["lora"], "trained held-out DLR >= 0.70 "
                    "(baseline ~0.3 from M1)"])

    # ── stdout report ──
    print("\n" + "=" * 72)
    print("M2 learnability result — LoRA + STRIPS on CI v2 transitions")
    print(f"  commit: {commit}")
    print(f"  probes: {len(HELD_OUT_PROBES)} held-out (unseen names)")
    for cond in ("untrained", "lora", "strips"):
        s = summary[cond]["all"]
        se = summary[cond]["expressible"]
        if s is None:
            print(f"  {cond:<22}: (no rows)")
            continue
        label = {"untrained": "untrained (zero-shot)", "lora": "LoRA trained",
                 "strips": "STRIPS rules"}[cond]
        print(f"  {label:<22}: DLR = {s['dlr']:.3f}  (L1 {s['l1']}/{s['n']}, "
              f"L2 {s['l2']}/{s['n']}, L3 {s['l3']}/{s['n']})  "
              f"expressible-subset DLR = {se['dlr'] if se else float('nan'):.3f}")
        for verb in ("cat", "echo", "ls"):
            v = summary[cond]["per_verb"][verb]
            print(f"      {verb:<5}: DLR = {v['dlr']:.3f}  "
                  f"(L1 {v['l1']}/{v['n']}, L2 {v['l2']}/{v['n']}, L3 {v['l3']}/{v['n']})")
    print(f"  parse failures: untrained={n_parse_fail['untrained']} "
          f"lora={n_parse_fail['lora']}")
    if trained_all is not None:
        print(f"  M2 verdict: {'PASS' if m2_pass else 'FAIL'} "
              f"(criterion: trained held-out DLR >= 0.70; "
              f"trained DLR = {trained_all['dlr']:.3f}, "
              f"untrained baseline = {untrained_all['dlr'] if untrained_all else float('nan'):.3f})")
        if not m2_pass:
            print("  note: DLR < 0.70 -> environment too chaotic for <=200 transitions; "
                  "loosen reversal per FF-CI-4.")
    else:
        print(f"  M2 verdict: PENDING (no LoRA rows yet; STRIPS DLR = "
              f"{strips_all['dlr'] if strips_all else float('nan'):.3f})")
    print(f"  artifacts: {jsonl_path.name}, {csv_path.name}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
