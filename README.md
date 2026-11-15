# hiring-research

Measurement pipeline for the paper "Junior in Title, Senior in Practice": what share of
junior software advertisements asks the candidate for a responsibility ordinarily expected of
more senior staff, in 2019 (Monster USA), 2024 and 2026 (LinkedIn).

## Layout
- `notebooks/` numbered, independently runnable in Colab; each ends with the standard commit cell
  - `00_setup` repository files, the June 2026 subsets and labels put in place, CPU checks
  - `01_subset_2024` the 2024 junior software pool rebuilt from the public LinkedIn postings dataset with the written rule
  - `02_rate` every advertisement rated twice by the language-model rater through the API (cached, tie-broken)
  - `03_labels` blind hand-labelling sheet and the agreement statistics once it is filled in
  - `04_results` template-corrected shares, trend, Table II and Fig. 2
- `src/` shared modules: `ads` (ids, text shown to the rater), `subsets` (selection rules), `rater` (prompt, API calls, cache),
  `validation` (labelling sheet, agreement), `dedup_inference` (template clusters, cluster bootstrap, clustered trend), `results`
- `reports/` committed result tables and JSON; ratings are keyed by advertisement id and title, never by advertisement text
- `figures/` paper figures at 600 dpi
- `tests/` CPU tests (`python -m pytest -q tests`); no network, no paid calls
- `data/` raw downloads, subsets with text, caches and the labelling sheet (gitignored)

## Rater
`src/rater.py` holds the fixed prompt (the six-part rubric of Table I), the model id, and the cache. Every API
call is logged with the model string the API served, token counts and the raw one-word answer. Sampling
parameters cannot be set on the current API models, so every advertisement is rated twice and a third pass
breaks ties; the pass-to-pass agreement is reported.

## Subset rule
`src/subsets.py` is the written selection rule: software-development title, no non-software role sharing the
vocabulary, junior status from a title cue or from the platform's experience-level field, and never a title
that carries an explicit senior cue. The 2024 pool is rebuilt from the raw dataset with this rule; the 2019 and
2026 subsets are the June 2026 files filtered by the same rule.

## License
MIT
