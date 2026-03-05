# LaTeX Project

This folder is set up as a LaTeX project with `main.tex` as the entrypoint.

## Build

```bash
make pdf
```

## Live rebuild

```bash
make watch
```

## Clean files

```bash
make clean
make distclean
```

## Notes

- Bibliography placeholder: `bib/refs.bib`
- Figures folder: `figures/`
- Section snippets folder: `sections/`

Your current `main.tex` references image files like `hard_sample_3.png`.
Add those files in this project root (or update paths in `main.tex` to `figures/...`).
