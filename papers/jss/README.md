# humpday JSS paper

The formal software write-up of `humpday`, in Journal of Statistical
Software format, following the model of the skaters paper.

| File | What it is |
|---|---|
| `humpday-jss.tex` | The paper, in JSS format. |
| `humpday.bib` | Bibliography. |
| `jss.cls` | Vendored minimal JSS class so the paper builds offline. |
| `compile_paper.sh` | One-command build (Tectonic). |

## Build

```bash
./compile_paper.sh     # produces humpday-jss.pdf
```

## SSRN submission checklist (to obtain a DOI)

1. Log in at https://www.ssrn.com and choose Submit a Paper.
2. Upload `humpday-jss.pdf`.
3. Title: *HumpDay: Derivative-Free Optimizers in Pure Python and
   JavaScript, with a Contamination-Resistant Real-World Benchmark*.
4. Abstract: paste the abstract from the PDF (it is defined once in the
   tex as `\humpdayabstract`, so the two cannot drift).
5. Keywords: derivative-free optimization, black-box optimization,
   benchmarking, algorithm selection, benchmark contamination, Python,
   JavaScript.
6. Classification: Computer Science Research Network; also fits
   Econometrics: Mathematical Methods & Programming.
7. On acceptance into the eLibrary SSRN assigns the DOI. Add it to
   `CITATION.cff` in the repository root and to the papers page at
   https://humpday.microprediction.org/papers.html.

The abstract and author metadata match the PDF; nothing else is needed at
submission time.
