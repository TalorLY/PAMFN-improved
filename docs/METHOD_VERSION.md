# Which version is the paper method?

| Version | Role | Included here? |
|---|---|---|
| v17 | Unchanged Ordered backbone + E + bounded modality supplement S; historical +1.453 | Frozen bundle; readable Python source; historical comparator |
| v29 fix1 | From-scratch reconstruction of Base/Ordered/E/S using the exact v17 arguments | Yes, unmodified historical runner |
| **v30** | **Paper main candidate: Ordered + E + U (`unbounded_s`), +1.549** | **Yes, primary directory `mainline/v30/`** |
| v31 | Repetition, late-epoch diagnostics, PAMFN timing | Separate evidence release; not a replacement method |
| v32 | Different protocol diagnostic; low Base is not the main 81.277 line | Not part of the paper mainline implementation |
| v33/v34 | Matched feature/backbone comparisons and CoFInAl/E transfer | Separate evidence release |
| v35/v36 | Design controls and mechanism/negative-case evidence | Separate evidence release |
| v37 | FS1000 PAMFN full-method experiment | Separate data adapter and experiment release |
| v38 | FS1000 MCMoE + Ordered/E/U transfer | Separate model adapter and experiment release |
| v39 | Strong-backbone controls and prediction-preserving inference optimization | Separate validation/deployment release, not a new PAMFN main method |

## Frozen identities

- v17 trainer commit: `1437dd4b9ce3fc220bdf530946a83d4bac88c4f7`.
- v17 branch: `codex/july28-qmd-supp-v17`.
- Original v17 bundle SHA256: `4edaede76f50d848b51093996dfafedeaada48327986809ac074440e3c5e98b7`.
- v30 archival snapshot: `e3a266343ed5376525eea9391ebf8d3cfa2e7cce`, branch `codex/archive-v17-evidence-v30`.
- The v30 archival commit is a later saved source snapshot, **not** a retroactively invented historical training commit.

Each distributed frozen file has an exact source hash in `provenance.json`. The new GitHub repository's future commits identify packaging/documentation; they must not be presented as the commits used to train old results.

## Two contributions, three implementation components

The intended structure is (1) ordered quality learning, and (2) backbone-preserving hierarchical fused/modal correction E/U. Three component names do not automatically constitute three novel research contributions.

The inherited E is implemented through `ModalBranch` with the `ordered_modal_fused_gt` recipe; this class name does not mean that the final E uses three separate modality heads. Inspect the recipe, not just the class name. U is `ControlledSupplement('unbounded_s')` in the v30 model, with scale retained.

The paper should retain all results and distinguish:

- the v17 bounded supplement S from the v30 U;
- a matched-baseline gain from a published-paper comparison;
- selected-checkpoint improvement from late-epoch stability (late degradation exists);
- parameter size from observed latency;
- modality-source correction evidence from untested event-level explanations.

No training recipe is changed by this source organization. The final paper still requires author-approved naming and novelty wording; this repository name is descriptive, not a claimed paper title.
