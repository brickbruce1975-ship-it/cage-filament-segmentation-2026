# Agent instructions

<!-- CAGE_LEARNING_SYNC:BEGIN -->
## CAGE learning process and ARC harnesses (synced 2026-11-23)

This repository is part of the CAGE fleet. Any model working here (Claude, ChatGPT/Codex, Gemini, Grok, Cursor,
Copilot, or a local model through Ollama) uses the same learning process:

- **Before starting:** recall what the fleet already learned:
  `python3 -m cage_learning recall "<the work ahead>"`, run from a checkout of
  [cage-agi-orchestrator](https://github.com/brickbruce1975-ship-it/cage-agi-orchestrator/blob/main/cage_learning/README.md).
- **While working:** write down the beliefs you rely on, test each one against what actually happens, audit often,
  and get an outside review when `stuck` says you are stuck (patience, overdue milestones, three failed variants).
- **After:** record the lesson in task terms and in general terms. Lessons export as `cage.task_outcome.v1` records
  ([integration](https://github.com/brickbruce1975-ship-it/cage-agi-ecosystem/blob/main/docs/ARC_LEARNING_INTEGRATION.md)), so what one repo learns can be used by the others.
- **ARC testing ground:** [inducer](https://github.com/brickbruce1975-ship-it/cage-agi-orchestrator/blob/main/competitions/arc_agi2/inducer/README.md) for ARC-AGI-2
  (`--pipeline` follows the leaders' solver and refiner design) and [player](https://github.com/brickbruce1975-ship-it/cage-agi-orchestrator/blob/main/competitions/arc_agi3/player/README.md)
  for ARC-AGI-3.

What the ARC-AGI competitions taught, for any task:

1. **Check before you answer.** Write the check (tests, a replay against the recorded history) and run it. Passing it
   still does not prove the answer: keep the strongest *different* reading as attempt 2.
2. **Start from the strongest existing approach.** Measure it first. If your own variants fail three times, switch
   approaches instead of trying a fourth.
3. **Cover everything first.** Give every item a first attempt before polishing any one of them.
4. **Test limiting beliefs first.** Beliefs like "impossible", "has no effect" or "is a wall" close off options, so test
   them before the others.
5. **Keep the real record.** Keep an authoritative record of what happened outside the working area, and replay it.
   When you are stuck, a read-only reviewer reads that record, not your account of it.
6. **Measure before you credit.** A stronger base model moved scores more than harness tweaks did.

- **Security and sovereignty:** model-written code runs sandboxed (resource limits, no network, answers hidden). No
  secrets go in prompts, logs or commits. Owner data stays private, and local models are supported.
- **Claim boundary:** true_rsi=false, true_agi=false. Benchmark gains and passing tests are not evidence of either.
<!-- CAGE_LEARNING_SYNC:END -->
