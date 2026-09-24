# @FAMILY@ reference material

Original paper and Lean sources released by OpenAI.

- Source: https://github.com/openai/NavierStokesAndEuler/tree/@COMMIT@
- Paper: @PAPER_URL@
- `paper.pdf` is the original paper. Source files and the upstream license are preserved.

## Optional verification

On a machine with the version in `lean-toolchain` installed, follow the [upstream prerequisites](https://github.com/openai/NavierStokesAndEuler/blob/@COMMIT@/ComparatorChallenges/README.md) for `landrun`, `lean4export`, and `nanoda_bin`. Then, from this extracted directory:

```sh
lake exe cache get
lake build @FAMILY@ ComparatorChallenges
lake exe comparator ComparatorChallenges/@FAMILY@.json
```

Dependency downloads require network access, and the full build and checks can take substantial time. Verification is optional for using these reference materials. The commands require a configured Lean environment; the Python workspace does not supply one.

This archive scopes the upstream Lake library list to @FAMILY@ while preserving the source contents and original compiler options.
