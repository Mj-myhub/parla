# Detection evaluation (real annotated essays)

| config | scorer | precision | recall | F0.5 | tp | fp | fn |
|---|---|---|---|---|---|---|---|
| deterministic | exact | 1.0 | 0.571 | 0.87 | 8 | 0 | 6 |
| deterministic | overlap | 1.0 | 0.571 | 0.87 | 8 | 0 | 6 |
| det+LLM | exact | 0.4 | 0.571 | 0.426 | 8 | 12 | 6 |
| det+LLM | overlap | 0.65 | 0.929 | 0.691 | 13 | 7 | 1 |

_EXACT = strict string match (a floor). OVERLAP = detection by span overlap, wording-agnostic (fair to the LLM). ERRANT is the standard upgrade._
