# Detection evaluation (real annotated essays)

| config | scorer | precision | recall | F0.5 | tp | fp | fn |
|---|---|---|---|---|---|---|---|
| deterministic | exact | 1.0 | 0.571 | 0.87 | 8 | 0 | 6 |
| deterministic | overlap | 1.0 | 0.571 | 0.87 | 8 | 0 | 6 |
| det+LLM | exact | 0.474 | 0.643 | 0.5 | 9 | 10 | 5 |
| det+LLM | overlap | 0.684 | 0.929 | 0.722 | 13 | 6 | 1 |

_EXACT = strict string match (a floor). OVERLAP = detection by span overlap, wording-agnostic (fair to the LLM). ERRANT is the standard upgrade._
