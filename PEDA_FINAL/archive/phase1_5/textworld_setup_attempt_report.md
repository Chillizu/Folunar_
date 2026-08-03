# TextWorld Setup Attempt Report

**Date**: 2026-07-20
**Project**: PEDA Phase 1.5 (Option A)
**Environment**: Isolated uv venv at `.venv_textworld` under `/home/chillizu/Projects/Folunar_/`

---

## 1. What Was Tried

### 1.1 Environment Setup
- Created an isolated Python 3.10 venv using `uv venv --python 3.10 .venv_textworld`
- Python version: 3.10.20 (CPython, downloaded by uv)
- Location: `/home/chillizu/Projects/Folunar_/.venv_textworld/`
- No existing project code or main venv was modified.

### 1.2 Dependencies Installed
- **textworld==1.7.0** (plus 53 dependencies including Jericho 3.3.1, spaCy 3.8.14, numpy 1.26.4)
- **gymnasium==1.3.0** (for compatibility; gym 0.26.2 was initially installed but later replaced)
- numpy was pinned to `<2` (1.26.4) to avoid NumPy 2.0 API breaks.

### 1.3 Import Verification
- `import textworld` succeeds.
- `textworld.__version__` returns `'1.7.0'`.
- All submodules accessible: `textworld.generator`, `textworld.envs`, `textworld.gym`, `textworld.core`, etc.

### 1.4 Smoke Test
A minimal end-to-end test was performed using `textworld.gym.envs.TextworldGymEnv`:

| Step | Action | Result |
|------|--------|--------|
| Create game | `textworld.make(GameOptions())` | `.z8` game file created in `./tw_games/` |
| Create env | `TextworldGymEnv([game_file], EnvInfos(...))` | Environment loaded successfully |
| Reset | `env.reset()` | Returns `(obs_text, info_dict)` with game intro |
| Step 'look' | `env.step('look')` | Returns room description, reward=0 |
| Step 'inventory' | `env.step('inventory')` | Returns "You are carrying nothing." |
| Step 'go east' | `env.step('go east')` | Returns "You can't go that way." |
| Close | `env.close()` | Clean exit |

All steps completed successfully. The environment returns proper text observations, admissible commands, and rewards.

## 2. Result

**TextWorld installation and basic functionality: SUCCESS**

- `pip install textworld` via uv completes without errors.
- Game generation (`textworld.make`) works — produces `.z8` + `.json` files.
- Environment load, reset, and step all function correctly.
- TextWorld 1.7.0 is compatible with Python 3.10.

## 3. Issues Encountered & Resolutions

| Issue | Cause | Resolution |
|-------|-------|------------|
| `TypeError: unhashable type: 'slice'` | Using `TextWorldEnv` directly with `.json` file — the raw `GameState` object doesn't support `[:N]` slicing | Use `TextworldGymEnv` instead, which wraps Jericho and returns plain strings |
| Gym NumPy 2.0 warning | gym 0.26.2 doesn't support NumPy 2.x | Pinned `numpy<2` (installed 1.26.4). Also works fine with `gymnasium` 1.3.0 |
| `gym.error.NameNotFound: Environment tw doesn't exist` | gym 0.26.2 registry format incompatible with `textworld.gym.register_game()` | Replaced `gym` with `gymnasium`; or bypass gym registration entirely by using `TextworldGymEnv` directly |
| `TextWorldEnv.reset()` returns `GameState` not `str` | `.json` file lacks Z-machine runtime for text rendering | Use `.z8` game files + `TextworldGymEnv` which integrates Jericho for proper text output |

## 4. Remaining Caveats

- **Python 3.10 only**: TextWorld 1.7.0 may not be compatible with Python 3.11+. Tested only on 3.10.20.
- **Game generation is random**: Default `GameOptions()` creates a random house/cookhouse/closet layout. For reproducible PEDA experiments, set `options.seeds = <int>`.
- **No GPU needed**: TextWorld is CPU-only; Jericho runs Z-machine games natively.

## 5. Recommendation

**Proceed with TextWorld (Option A).**

TextWorld installs cleanly via `uv` with Python 3.10, and the basic env loop works correctly. Key integration points for PEDA Phase 1.5:

```python
from textworld import GameOptions, EnvInfos
from textworld.gym.envs import TextworldGymEnv

game_file, game = textworld.make(GameOptions())
env = TextworldGymEnv([game_file], EnvInfos(
    description=True, inventory=True, admissible_commands=True
))
obs, info = env.reset()
obs, reward, done, info = env.step('look')
```

No blockers remain for PEDA Phase 1.5 TextWorld integration.
