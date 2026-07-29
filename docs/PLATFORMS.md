# Execution platform separation

Last updated: 2026-07-29

**Running on any particular compute facility is optional.** The core assumes
no platform. Support for a platform lives in a loosely coupled module and is
reached only through that module.

**This separation is held by machinery, not by documentation.**
`tests/test_platform_isolation.py` fails if it is broken. We have seen, with a
concrete example, that a policy written down is not a policy that holds
(Capture repository, `docs/DESIGN.md` §5.26).

---

## 1. Structure

```
platforms/
  base.py            the shared interface. **No platform vocabulary here**
  __init__.py        resolves by name, dynamically. **No table**
  local/backend.py   runs on this machine. The default, and self-contained
  <name>/backend.py  anything else, all optional
```

The core obtains a backend through `platforms.load_backend(name)`.
**The name comes from the caller.** No platform name is written in the code:
the moment one is, the core knows about that platform.

```python
from platforms import load_backend, JobSpec

backend = load_backend(args.platform)      # defaults to "local"
backend.Backend().submit(JobSpec(
    name="ctxpred-step1", command=[...], env_name="py3.10_context_prediction",
    gpus=8, hours=24))
```

---

## 2. The interface

`JobSpec` states **what is needed, in ordinary words**.

| Field | Meaning |
|---|---|
| `name` | job name |
| `command` | the command to run |
| `env_name` | the conda environment to use |
| `gpus` / `hours` | **the amount needed.** Not a resource type or a queue name |
| `workdir` / `env` | working directory and environment variables |

**Translating that into a resource type is each backend's job.** The core
holds no translation table. Sharing one would mean the core knows that
platform's vocabulary.

`Backend` has exactly two methods.

| Method | Contract |
|---|---|
| `is_available()` | **Check, do not assume** — e.g. whether the submit command exists |
| `submit(spec)` | Run or enqueue, and return a `JobResult` |

`JobResult.exit_status` is **`None` when the outcome is not yet known.**
A backend that only enqueues must not return `0`. **0 means "it succeeded",
and an unknown outcome must never be passed off as a success.**

---

## 3. What the machinery holds

| Check | What breaking it would cause |
|---|---|
| Platform-specific vocabulary stays inside `platforms/<name>/` | the core becomes tied to that platform |
| `platforms/local/` always exists | an extra platform becomes mandatory and nothing runs locally |
| No platform-specific vocabulary in the interface | the interface is polluted and the separation is nominal |
| Nothing outside `platforms/` imports a specific platform | importing it *is* knowing about it |
| Resolution is **discovery, not a table** | a newly added platform is not found |
| Both backends implement the same interface | they cannot be swapped |

The last two are checked **by behaviour**: a dummy backend is dropped in to
see whether it is discovered, and `issubclass` is evaluated for real. Deciding
by string matching misfires — it flagged a usage example inside a docstring as
a hard-coded name (it actually did).

---

## 4. Adding a platform

1. Write `Backend(base.Backend)` in `platforms/<name>/backend.py`
2. Have `is_available()` **actually check** whether it can be used
3. Keep the need-to-resource translation **inside that module only**
4. Make `./tests/run-tests.sh` pass

**No registration step.** Put the file there and `available_backends()`
finds it.

---

## 5. Left undecided

- How an asynchronous backend waits for completion after enqueuing
  (polling or notification)
- Who assigns `WORLD_SIZE` across several nodes
- How logs are collected

**These are decided after the two pilot methods are through.** Deciding now
would produce a design that has never met a real job.
