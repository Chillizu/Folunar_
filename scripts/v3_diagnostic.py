#!/usr/bin/env python3
"""Quick diagnostic: test v3 sandbox with the new tasks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase2.sandbox_env import BusyboxSandbox, generate_sandbox_candidates
from phase2.tasks import MICRO_TASKS

sb = BusyboxSandbox()
state = sb.reset(start_cwd="/sandbox")
print(f"CWD: {state.cwd}")
print(f"Files: {state.files}")
print()

candidates = generate_sandbox_candidates(state)
print(f"Candidates ({len(candidates)}):")
for c in candidates[:12]:
    print(f"  {c}")
print()

# Test 1: read_greeting
ns, _, _ = sb.step(state, "cat greeting.txt")
print(f"cat greeting.txt -> exit={ns.last_exit_code}, out={ns.last_output[:60]}")
t = next(t for t in MICRO_TASKS if t["id"] == "read_greeting")
r1 = t["check"](state, "cat greeting.txt", ns)
print(f"  read_greeting check: {r1}")

# Test 2: count_entries
ns2, _, _ = sb.step(state, "wc -l dataset/entries.txt")
print(f"wc -l dataset/entries.txt -> exit={ns2.last_exit_code}, out={ns2.last_output[:60]}")
t2 = next(t for t in MICRO_TASKS if t["id"] == "count_entries")
r2 = t2["check"](state, "wc -l dataset/entries.txt", ns2)
print(f"  count_entries check: {r2}")

# Test 3: find_secret_note via grep
ns3, _, _ = sb.step(state, "grep -r secret .")
print(f"grep -r secret . -> exit={ns3.last_exit_code}, out={ns3.last_output[:60]}")
t3 = next(t for t in MICRO_TASKS if t["id"] == "find_secret_note")
r3 = t3["check"](state, "grep -r secret .", ns3)
print(f"  find_secret_note check (grep): {r3}")

# Test 3b: find_secret_note via cat
ns3b, _, _ = sb.step(state, "cat records/secret_note.txt")
print(f"cat records/secret_note.txt -> exit={ns3b.last_exit_code}, out={ns3b.last_output[:60]}")
r3b = t3["check"](state, "cat records/secret_note.txt", ns3b)
print(f"  find_secret_note check (cat): {r3b}")

# Test 4: read_user_guide
ns4, _, _ = sb.step(state, "cat records/user_guide.md")
print(f"cat records/user_guide.md -> exit={ns4.last_exit_code}, out={ns4.last_output[:60]}")
t4 = next(t for t in MICRO_TASKS if t["id"] == "read_user_guide")
r4 = t4["check"](state, "cat records/user_guide.md", ns4)
print(f"  read_user_guide check: {r4}")

sb.close()
print("\nALL DIAGNOSTICS COMPLETE")
