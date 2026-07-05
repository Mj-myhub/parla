# Detection evaluation (real annotated essays)

| config | scorer | precision | recall | F0.5 | tp | fp | fn |
|---|---|---|---|---|---|---|---|
| deterministic | exact | 0.0 | 0.0 | 0.0 | 0 | 0 | 5 |
| deterministic | overlap | 0.0 | 0.0 | 0.0 | 0 | 0 | 5 |
| det+LLM | exact | 0.0 | 0.0 | 0.0 | 0 | 9 | 5 |
| det+LLM | overlap | 0.444 | 0.8 | 0.488 | 4 | 5 | 1 |

_EXACT = strict string match (a floor). OVERLAP = detection by span overlap, wording-agnostic (fair to the LLM). ERRANT is the standard upgrade._
