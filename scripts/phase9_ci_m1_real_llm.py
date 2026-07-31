#!/usr/bin/env python3
"""M1 prior-breakage experiment — REAL-LLM world model (FF-CI-3 gate).

Same protocol as the stub M1 (scripts/phase9_ci_m1_prior_breakage.py) but the
"untrained prior" is now the actual zero-shot Qwen2.5-0.5B-Instruct loaded
from /home/data/models/Qwen2.5-0.5B-Instruct (no LoRA, no fine-tuning, CPU).

Measures how badly a real pretrained LLM (with normal Unix command-semantics
priors) predicts command outcomes in peda-sandbox:counterintuitive-v2
(deepened reversals: cat deletes -> rc 1, echo reads -> rc 2, ls creates .ls
twins silently -> rc 3) vs peda-sandbox:v4.

Protocol (plan-counter-intuitive-sandbox.md, milestone M1):
  * Deterministic probe set P: 30 (state, action) pairs — 10 per reversed
    verb (cat, echo, ls) — each run against BOTH images with a FRESH
    container (identical protocol + identical model on both images).
  * DLR = fraction of {L1 exit_code, L2 files_delta, L3 output_summary}
    components predicted correctly.
  * Threshold: CI DLR <= 0.35 AND v4 DLR >= 0.8.
  * Interpretation: CI DLR > 0.35 => the reversal is too shallow — the real
    LLM can still guess outcomes from normal priors -> deepen further.

Per-probe flow: fresh container -> optional fixture setup via /bin/busybox
(no PATH-wrapper side effects) -> state_text (world-model prompt format from
src/phase1/world_model.py) -> greedy generation -> parse JSON -> execute action
-> recursive fs snapshot compare -> L1/L2/L3.

Action execution bypasses BusyboxSandbox.step() because its text=True decode
raises UnicodeDecodeError on binary output (`cat binary.dat` on v4); docker
exec with errors="replace" keeps the run alive. Container lifecycle and state
capture still go through the sandbox API.

Outputs:
  results/phase9_ci_m1_real_llm.jsonl        (D4 meta header + per-probe rows)
  results/phase9_ci_m1_real_llm_summary.csv  (per-image + per-verb DLR + verdict)

Usage: PYTHONPATH=src python scripts/phase9_ci_m1_real_llm.py
"""

import csv
import datetime
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from phase2.sandbox_env import (  # noqa: E402
    BusyboxSandbox,
    CounterIntuitiveSandbox,
    _list_files,
)

# Real-LLM imports (heavy; loaded after path setup).
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from phase1.world_model import WorldModel  # noqa: E402

CI_IMAGE = "peda-sandbox:counterintuitive-v2"
V4_IMAGE = "peda-sandbox:v4"
MODEL_PATH = "/home/data/models/Qwen2.5-0.5B-Instruct"
SANDBOX_ROOT = "/sandbox"

CI_ROW_LABEL = "counterintuitive-v2"
V4_ROW_LABEL = "v4"

# Probe set P: 30 (state, action) pairs, 10 per reversed verb.
# Each probe: (verb, action, setup, start_cwd, note).
#   setup     — optional fixture creation, run via `/bin/busybox sh -c` on the
#               fresh container BEFORE the state is captured (no wrapper side
#               effects; needed for recently_created_file / binary / empty /
#               special-chars probes). The fixture therefore appears in the
#               state_text the LLM sees.
#   start_cwd — reset() lands the container here instead of /sandbox
#               (used by the "ls after_cd" probe).
# Targets shared by both images: welcome.txt (file), docs/ (dir).
# CI-only: docs/note.txt, docs/readme.txt, data/lines.txt, logs/*.
# v4-only: docs/api_reference.md, docs/tutorials/, data/raw/, projects/...
PROBES = [
    # ── cat (10): normal semantics = reads file args, rc 0, output, no fs change.
    #    CI rule: cat DELETES its file arguments, rc 1, no output.
    ("cat", "cat welcome.txt", None, None, "existing_file (shared)"),
    ("cat", "cat missing_file.txt", None, None, "missing file"),
    ("cat", "cat docs", None, None, "dir target"),
    ("cat", "cat -n welcome.txt", None, None, "flag + file"),
    ("cat", "cat welcome.txt welcome.txt", None, None, "multi-arg"),
    ("cat", "cat docs/note.txt", None, None, "subdir existing file (CI-only)"),
    ("cat", "cat", None, None, "no args (stdin EOF)"),
    ("cat", "cat binary.dat", "cp /bin/busybox /sandbox/binary.dat", None,
     "binary-looking file (setup fixture)"),
    ("cat", "cat empty.txt", "touch /sandbox/empty.txt", None,
     "empty file (setup fixture)"),
    ("cat", "cat recent.txt", "echo fresh > /sandbox/recent.txt", None,
     "recently-created file (setup fixture)"),
    # ── echo (10): normal semantics = prints args, rc 0, no fs change.
    #    CI rule: echo READS existing file args (rc 2); non-file args -> rc 1, no output.
    ("echo", "echo welcome.txt", None, None, "existing file (shared)"),
    ("echo", "echo missing_file.txt", None, None, "missing file"),
    ("echo", "echo 'hello world'", None, None, "text args"),
    ("echo", "echo welcome.txt docs", None, None, "multi-arg"),
    ("echo", "echo", None, None, "no args"),
    ("echo", "echo docs", None, None, "dir target"),
    ("echo", "echo empty.txt", "touch /sandbox/empty.txt", None,
     "empty file (setup fixture)"),
    ("echo", "echo recent.txt", "echo fresh > /sandbox/recent.txt", None,
     "recently-created file (setup fixture)"),
    ("echo", "echo -n welcome.txt", None, None, "flag + file"),
    ("echo", "echo /tmp/nonexistent", None, None, "nonexistent path"),
    # ── ls (10): normal semantics = lists target, rc 0, no fs change.
    #    CI rule: ls exits 3 silently and creates one "<entry>.ls" twin per
    #    entry of the target dir.
    ("ls", "ls", None, None, "cwd"),
    ("ls", "ls -l", None, None, "flag"),
    ("ls", "ls docs", None, None, "subdir"),
    ("ls", "ls missing_dir", None, None, "missing dir"),
    ("ls", "ls .", None, None, "no args (same as ls)"),
    ("ls", "ls /tmp", None, None, "empty tmpfs mount"),
    ("ls", "ls welcome.txt", None, None, "file target (not dir)"),
    ("ls", "ls docs/tutorials", None, None, "subdir/nested (v4-only)"),
    ("ls", "ls", None, "/sandbox/docs", "after_cd (cwd = /sandbox/docs)"),
    ("ls", 'ls "dir with space"', "mkdir -p '/sandbox/dir with space'", None,
     "special-chars dir (setup fixture)"),
]


def _fs_snapshot(cid: str) -> set:
    """Recursive listing of /sandbox (files AND dirs) — the true fs delta.

    The sandbox state's flat `files` list cannot see subdirectory changes
    (e.g. docs/note.txt.ls twins or docs/note.txt deleted by `cat`), so L2
    compares recursive snapshots.
    """
    r = subprocess.run(
        ["docker", "exec", cid, "/bin/busybox", "find", SANDBOX_ROOT],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0:
        return set()
    return {p for p in r.stdout.strip().splitlines() if p and p != SANDBOX_ROOT}


def _extract_json(text: str):
    """Balanced-brace extraction of the first JSON object (robust to ```json fences)."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def main() -> int:
    for image in (CI_IMAGE, V4_IMAGE):
        r = subprocess.run(["docker", "image", "inspect", image],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FATAL: docker image missing: {image}", file=sys.stderr)
            return 2

    print(f"Loading {MODEL_PATH} (CPU, fp16, zero-shot)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.float16, device_map="cpu"
    )
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    print("Model loaded.", flush=True)

    # World-model prompt builders (methods do not touch instance state; bind to
    # a bare instance to reuse the canonical prompt text from world_model.py).
    _wm = object.__new__(WorldModel)
    system_msg = _wm._sandbox_system_message()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "phase": "9",
        "direction": "counter-intuitive-sandbox",
        "experiment": "M1_prior_breakage_real_llm",
        "commit": commit,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cpu_or_gpu": "cpu",
        "sandbox_images": [CI_IMAGE, V4_IMAGE],
        "model": "Qwen2.5-0.5B-Instruct (zero-shot, untrained prior, CPU fp16)",
        "model_path": MODEL_PATH,
        "seeds": [],
        "per_episode_data_present": True,
        "probe_set": "P: 30 deterministic (state, action) pairs, 10 per reversed verb (cat/echo/ls), "
                     "fresh container per probe; 4 probes use busybox setup fixtures "
                     "(binary.dat, empty.txt, recent.txt, 'dir with space'), 1 starts in /sandbox/docs (after_cd)",
        "dlr_components": ["L1 exit_code", "L2 files_delta", "L3 output_summary"],
        "threshold": {"ci_dlr_max": 0.35, "normal_dlr_min": 0.8},
        "prompt_format": "src/phase1/world_model.py _sandbox_system_message + _build_text_prompt",
        "generation": "greedy (do_sample=False), max_new_tokens=160",
        "l3_rule": "predicted output non-empty = bool(last_output or summary); "
                    "last_output-only variant recorded per row as diagnostic",
    }

    rows = []
    n_parse_fail = 0
    for image in (CI_IMAGE, V4_IMAGE):
        row_label = CI_ROW_LABEL if image == CI_IMAGE else V4_ROW_LABEL
        for probe_id, probe in enumerate(PROBES, start=1):
            verb, action, setup, start_cwd, note = probe
            if image == CI_IMAGE:
                sandbox = CounterIntuitiveSandbox()  # writable rootfs (cat deletes, ls twins)
            else:
                # v4 is --read-only by default, but the setup-fixture probes
                # need a writable rootfs; cat/echo/ls never mutate on v4, so
                # ground truth is unaffected.
                sandbox = BusyboxSandbox(image=image, read_only=False)
            try:
                state = sandbox.reset(start_cwd=start_cwd)
                cid = state.container_id

                if setup:
                    subprocess.run(
                        ["docker", "exec", cid, "/bin/busybox", "sh", "-c", setup],
                        capture_output=True, text=True, timeout=10,
                    )
                    state.files = _list_files(cid, state.cwd)

                before = _fs_snapshot(cid)
                state_text = state.to_json()

                # ── LLM prediction (zero-shot, world-model prompt format) ──
                prompt = _wm._build_text_prompt(state, action)
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ]
                encoded = tokenizer.apply_chat_template(
                    messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
                )
                try:
                    input_ids = encoded["input_ids"]  # BatchEncoding
                except (TypeError, KeyError):
                    input_ids = encoded  # plain tensor
                with torch.no_grad():
                    gen_out = model.generate(
                        input_ids,
                        max_new_tokens=160,
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                raw = tokenizer.decode(gen_out[0, input_ids.shape[1]:], skip_special_tokens=True)
                parsed = _extract_json(raw)

                if parsed is None:
                    n_parse_fail += 1
                    pred_exit, pred_delta, pred_output, pred_files = None, None, None, None
                    pred_output_lo, pred_summary = None, None
                else:
                    pred_exit = parsed.get("exit_code", None)
                    try:
                        pred_exit = int(pred_exit)
                    except (TypeError, ValueError):
                        pred_exit = None
                    pred_files = parsed.get("files", None)
                    pred_delta = (
                        isinstance(pred_files, list)
                        and set(map(str, pred_files)) != set(state.files)
                    )
                    pred_summary = str(parsed.get("summary", "") or "")
                    pred_output_lo = bool(str(parsed.get("last_output", "") or "").strip())
                    # L3 = "output_summary" (protocol): the model communicates its
                    # output expectation in `summary` and leaves `last_output` empty,
                    # so predicted output non-empty = last_output OR summary.
                    pred_output = pred_output_lo or bool(pred_summary.strip())

                # ── Execute action -> ground truth (decoder-safe docker exec) ──
                r = subprocess.run(
                    ["docker", "exec", "-w", state.cwd, cid, "sh", "-c", action],
                    capture_output=True, text=True, errors="replace", timeout=15,
                )
                after = _fs_snapshot(cid)
                actual_exit = r.returncode
                actual_delta = before != after
                actual_output = r.stdout.strip() or r.stderr.strip()
                actual_nonempty = bool(actual_output)

                l1_correct = (pred_exit is not None) and pred_exit == actual_exit
                l2_correct = (pred_delta is not None) and pred_delta == actual_delta
                l3_correct = (pred_output is not None) and pred_output == actual_nonempty

                row = {
                    "image": row_label,
                    "probe_id": probe_id,
                    "verb": verb,
                    "action": action,
                    "note": note,
                    "l1_correct": l1_correct,
                    "l2_correct": l2_correct,
                    "l3_correct": l3_correct,
                    "dlr": sum((l1_correct, l2_correct, l3_correct)) / 3.0,
                    "predicted": {
                        "exit_code": pred_exit,
                        "files_delta": pred_delta,
                        "output_nonempty": pred_output,
                        "output_nonempty_last_output_only": pred_output_lo,
                        "files": pred_files,
                        "summary": pred_summary,
                    },
                    "actual": {
                        "exit_code": actual_exit,
                        "files_delta": actual_delta,
                        "output_nonempty": actual_nonempty,
                        "output": actual_output[:120],
                    },
                    "state_text": state_text[:300],
                    "raw_generation": raw[:300],
                }
                rows.append(row)
                print(f"[{row_label} {probe_id:02d}/{len(PROBES)}] {action:<30} "
                      f"L1={l1_correct!s:<5} L2={l2_correct!s:<5} L3={l3_correct!s:<5} "
                      f"dlr={row['dlr']:.3f}  pred_exit={pred_exit} actual_exit={actual_exit}",
                      flush=True)
            finally:
                sandbox.close()

    # ── JSONL with D4 meta header ──
    jsonl_path = REPO_ROOT / "results" / "phase9_ci_m1_real_llm.jsonl"
    with jsonl_path.open("w") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Per-image + per-verb stats ──
    def stats(subset):
        n = len(subset)
        if n == 0:
            return None
        # L3 diagnostic: how many rows would be L3-correct under the
        # last_output-only rule (model habitually leaves last_output empty).
        def l3_lo_correct(r):
            po = r["predicted"].get("output_nonempty_last_output_only")
            return po is not None and po == r["actual"]["output_nonempty"]

        return {
            "n": n,
            "l1": sum(r["l1_correct"] for r in subset),
            "l2": sum(r["l2_correct"] for r in subset),
            "l3": sum(r["l3_correct"] for r in subset),
            "l3_lo": sum(l3_lo_correct(r) for r in subset),
            "dlr": sum(r["dlr"] for r in subset) / n,
        }

    v4_all = stats([r for r in rows if r["image"] == V4_ROW_LABEL])
    ci_all = stats([r for r in rows if r["image"] == CI_ROW_LABEL])
    v4_pass = v4_all["dlr"] >= meta["threshold"]["normal_dlr_min"]
    ci_pass = ci_all["dlr"] <= meta["threshold"]["ci_dlr_max"]
    m1_pass = v4_pass and ci_pass

    csv_path = REPO_ROOT / "results" / "phase9_ci_m1_real_llm_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "scope", "n_probes", "l1_correct", "l2_correct",
                    "l3_correct", "dlr", "threshold", "threshold_met"])
        for s, label in ((v4_all, V4_ROW_LABEL), (ci_all, CI_ROW_LABEL)):
            thresh = (f"dlr >= {meta['threshold']['normal_dlr_min']}" if label == V4_ROW_LABEL
                      else f"dlr <= {meta['threshold']['ci_dlr_max']}")
            met = "PASS" if (label == V4_ROW_LABEL and v4_pass) or (label == CI_ROW_LABEL and ci_pass) else "FAIL"
            w.writerow([label, "all", s["n"], s["l1"], s["l2"], s["l3"],
                        f"{s['dlr']:.4f}", thresh, met])
        for label in (V4_ROW_LABEL, CI_ROW_LABEL):
            for verb in ("cat", "echo", "ls"):
                s = stats([r for r in rows if r["image"] == label and r["verb"] == verb])
                w.writerow([label, verb, s["n"], s["l1"], s["l2"], s["l3"],
                            f"{s['dlr']:.4f}", "", ""])
        w.writerow([])
        w.writerow(["m1_pass", "ci_dlr", "v4_dlr", "parse_failures", "criterion"])
        w.writerow([m1_pass, f"{ci_all['dlr']:.4f}", f"{v4_all['dlr']:.4f}",
                    n_parse_fail, "CI DLR <= 0.35 AND v4 DLR >= 0.8"])

    # ── stdout report ──
    print("\n" + "=" * 72)
    print("M1 prior-breakage result — REAL LLM (Qwen2.5-0.5B-Instruct, zero-shot)")
    print(f"  commit:   {commit}")
    print(f"  model:    {meta['model']}")
    print(f"  probes:   {len(PROBES)} per image x 2 images = {len(rows)} runs "
          f"(fresh container each); parse failures: {n_parse_fail}")
    for label, s, thresh, ok in (
        (V4_ROW_LABEL, v4_all, ">= 0.8", v4_pass),
        (CI_ROW_LABEL, ci_all, "<= 0.35", ci_pass),
    ):
        print(f"  {label:<17}: DLR = {s['dlr']:.3f}  (L1 {s['l1']}/{s['n']}, "
              f"L2 {s['l2']}/{s['n']}, L3 {s['l3']}/{s['n']})  threshold {thresh}: "
              f"{'PASS' if ok else 'FAIL'}")
        print(f"      L3 diagnostic (last_output-only rule): {s['l3_lo']}/{s['n']}")
        for verb in ("cat", "echo", "ls"):
            v = stats([r for r in rows if r["image"] == label and r["verb"] == verb])
            print(f"      {verb:<5}: DLR = {v['dlr']:.3f}  "
                  f"(L1 {v['l1']}/{v['n']}, L2 {v['l2']}/{v['n']}, L3 {v['l3']}/{v['n']})")
    print(f"  M1 verdict: {'PASS' if m1_pass else 'FAIL'} "
          f"(criterion: CI DLR <= 0.35 AND v4 DLR >= 0.8)")
    if not m1_pass and ci_all["dlr"] > meta["threshold"]["ci_dlr_max"]:
        print(f"  note: CI DLR {ci_all['dlr']:.3f} > 0.35 — the real LLM CAN guess "
              "outcomes from normal priors; the reversal is still too shallow -> "
              "deepen further (more exit codes, more verbs, content-level manipulation).")
    print(f"  artifacts: {jsonl_path.name}, {csv_path.name}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
