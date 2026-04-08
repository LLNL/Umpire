# Fortran Interface

## Generated Files

Do not edit generated files in `src/umpire/interface/c_fortran/`, including:

- `*.f`
- `wrap*.cpp`
- `wrap*.h`
- `types*.h`
- `genc*.inc`

## Editable Source of Truth

- Make Fortran interface changes in `src/umpire/interface/umpire_shroud.yaml`.
- Regenerate wrappers through the build or the established Shroud workflow after editing the YAML.

## Validation

- Ensure the generated code still compiles.
- Update or run relevant Fortran examples and tests when the wrapped surface changes.
- Check generated outputs only as build artifacts or review evidence, not as the primary edit target.

## Practical Rule

- If the task starts with "change a `.f` file under `c_fortran/`", the task is probably pointed at the wrong file. Start from the YAML instead.
