---
name: umpire-fortran-interface
description: Shroud and Fortran interface guidance for Umpire. Use when touching `src/umpire/interface/umpire_shroud.yaml`, generated Fortran wrappers, C wrapper files, or Fortran examples and tests.
---

# Umpire Fortran Interface

Treat the Fortran interface as generated code with a single editable source of truth. Use this skill whenever the task involves Shroud configuration, wrapper regeneration, or validating Fortran-facing behavior.

## Workflow

1. Read [references/fortran-interface.md](references/fortran-interface.md) before touching anything under `src/umpire/interface/`.
2. Edit `src/umpire/interface/umpire_shroud.yaml` instead of generated wrapper outputs.
3. Regenerate and test the interface through the build or documented workflow rather than patching generated files manually.

## Operating Rules

- Never hand-edit generated Fortran or wrapper outputs.
- Treat wrapper regeneration and validation as part of the change, not as optional follow-up.
- Update examples or tests when the exposed Fortran surface changes.
