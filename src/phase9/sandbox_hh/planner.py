"""Phase 9 sandbox-hh: high-layer frontier-goal planner.

The planner owns the only variable of the experiment: which directory the
low layer is steered toward. It maintains a directory graph of the
deterministic sandbox, built incrementally from:

  * the ls listing observed at every visited cwd (`state.files`), and
  * the relative paths printed by `find . -name '*.ext'` probes
    (e.g. `./docs/note.txt` reveals dir `docs` and file `note.txt`
    without a visit).

Frontier goal selection (pre-registered in the report, not tuned):

    unvisited_density(d) = |unvisited candidates at d| / |candidates at d|

where candidates at d are the Phase 8 verb x file matrix for every known
text file (cat / head -n 5 / wc -l) plus one cd per known subdir; an
action is "unvisited" when the explorer's state_action_counts has a zero
count for (state_hash(d), action). Directories with density 0 are not
frontiers and are never selected. Score:

    J(d) = unvisited_density(d) - lam * dist(cwd, d)

dist = BFS cd-step distance in the known graph; a goal must be reachable
(path exists), otherwise it is skipped. Tie-break: smaller dist, then
larger raw unvisited count, then lexicographic path.

Open-loop: selection happens at episode start and again only when the
local frontier at the current directory is exhausted. No mid-plan
re-evaluation.

R1 (FF-SBH-3): the agent also re-enters select right after a cd into a
directory with no readable text files (empty-dir trap, failure analysis
T2/R1); such a textless cwd is never itself selected as goal (it is the
frontier-exhausted equivalent), so the re-selection steers toward a
text-bearing frontier instead of keeping the low layer stuck.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional

from phase8.count_driven_agent import _P8_TEXT_EXTS


def _classify_entry(name: str) -> str:
    """'file' (text extension), 'dir' (no extension), or 'other'."""
    if "." in name:
        return "file" if name.rsplit(".", 1)[-1] in _P8_TEXT_EXTS else "other"
    return "dir"


class DirGraph:
    """Known directory tree of the deterministic sandbox.

    Nodes are absolute directory paths. Parent links are learned from
    cd transitions and from find-path prefixes; children are the
    no-extension entries observed in a directory's ls listing (or
    revealed as path components by find). Content of a directory is
    partial until it is actually visited.
    """

    def __init__(self) -> None:
        self.entries: Dict[str, set] = defaultdict(set)   # dir -> known entries
        self.parent: Dict[str, Optional[str]] = {}        # dir -> parent dir

    # ── knowledge ingestion ────────────────────────────

    def observe_cwd(self, cwd: str, files) -> None:
        """Record the full ls listing of a visited directory."""
        cwd = cwd.rstrip("/") or "/"
        self.parent.setdefault(cwd, None)
        self.entries[cwd] = set(files)
        for e in files:
            if _classify_entry(e) == "dir":
                self.parent.setdefault(f"{cwd}/{e}", cwd)

    def observe_find(self, cwd: str, output: str) -> None:
        """Record directory/file knowledge from a `find` probe output."""
        cwd = cwd.rstrip("/") or "/"
        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("./"):
                continue
            parts = [p for p in line[2:].split("/") if p]
            if not parts:
                continue
            cur = cwd
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    nxt = f"{cur}/{part}"
                    self.parent.setdefault(nxt, cur)
                    self.entries[cur].add(part)
                    cur = nxt
                else:
                    self.entries[cur].add(part)

    def note_parent(self, child: str, parent: str) -> None:
        """Record a cd-transition parent link (only first observation)."""
        child = child.rstrip("/") or "/"
        parent = parent.rstrip("/") or "/"
        self.parent.setdefault(child, parent)

    # ── queries ────────────────────────────────────────

    def known_dirs(self) -> List[str]:
        return list(self.parent.keys())

    def text_files(self, d: str) -> List[str]:
        return [e for e in self.entries[d] if _classify_entry(e) == "file"]

    def subdirs(self, d: str) -> List[str]:
        return [e for e in self.entries[d] if _classify_entry(e) == "dir"]

    def candidates(self, d: str) -> List[str]:
        """Phase 8 verb x file matrix + one cd per known subdir."""
        out: List[str] = []
        for f in self.text_files(d):
            out += [f"cat {f}", f"head -n 5 {f}", f"wc -l {f}"]
        for s in self.subdirs(d):
            out.append(f"cd {s}")
        return out

    def shortest_path(self, cwd: str, goal: str) -> Optional[List[str]]:
        """BFS cd-action sequence from cwd to goal in the known graph.

        Returns None when unreachable. `cd ..` moves to a known parent;
        `cd <name>` moves to a known child.
        """
        cwd = cwd.rstrip("/") or "/"
        goal = goal.rstrip("/") or "/"
        if cwd == goal:
            return []
        prev = {cwd: None}
        act = {cwd: None}
        q = deque([cwd])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            # children
            for s in self.subdirs(cur):
                nxt = f"{cur}/{s}"
                if nxt not in prev:
                    prev[nxt] = cur
                    act[nxt] = f"cd {s}"
                    q.append(nxt)
            # parent
            p = self.parent.get(cur)
            if p and p not in prev:
                prev[p] = cur
                act[p] = "cd .."
                q.append(p)
        if goal not in prev:
            return None
        path: List[str] = []
        node = goal
        while node != cwd:
            path.append(act[node])
            node = prev[node]
        path.reverse()
        return path


class SandboxGoalPlanner:
    """Frontier-goal selector (high layer). See module docstring."""

    def __init__(self, lam: float) -> None:
        self.lam = float(lam)
        self.graph = DirGraph()

    def unvisited_density(self, d: str, explorer) -> tuple:
        """Return (density, unvisited_count, total_count) for dir d."""
        cands = self.graph.candidates(d)
        if not cands:
            return 0.0, 0, 0
        sh = f"{d}|{','.join(sorted(self.graph.entries[d]))}"
        counts = explorer.state_action_counts
        unvisited = sum(1 for c in cands if counts[(sh, c)] == 0)
        return unvisited / len(cands), unvisited, len(cands)

    def select_goal(self, cwd: str, explorer) -> Optional[dict]:
        """Pick the frontier dir maximizing J(d); None when no frontier.

        Returns a log dict with the chosen goal and top contenders.
        """
        scored = []
        for d in self.graph.known_dirs():
            density, unvisited, total = self.unvisited_density(d, explorer)
            if density <= 0:
                continue
            # R1 (failure analysis T2/R1): a cwd with no readable text
            # files is a dead end for local exploration — re-selecting it
            # would keep the low layer stuck there (goal == cwd => no
            # navigation). Treat it as the frontier-exhausted equivalent
            # and exclude it so the forced re-selection picks a
            # text-bearing frontier.
            if d == cwd and not self.graph.text_files(d):
                continue
            path = self.graph.shortest_path(cwd, d)
            if path is None:
                continue  # unreachable in the known graph -> not a goal
            dist = len(path)
            j = density - self.lam * dist
            scored.append({
                "goal": d, "density": round(density, 4), "dist": dist,
                "j": round(j, 4), "unvisited": unvisited, "total": total,
            })
        if not scored:
            return None
        scored.sort(key=lambda r: (-r["j"], r["dist"], -r["unvisited"], r["goal"]))
        best = dict(scored[0])
        best["contenders"] = [
            {k: r[k] for k in ("goal", "density", "dist", "j", "unvisited", "total")}
            for r in scored[:5]
        ]
        return best
