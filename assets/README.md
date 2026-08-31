# Logo source

`logo-source.png` is the full-resolution original: 1312 × 1199 RGBA, 850 KB.
It is the master. Nothing references it at build time — it lives here so the
shipped asset can be regenerated at any size later.

The asset actually used is `docs/src/assets/logo.png`, 400 × 366 and 117 KB.
Documenter picks that path up automatically (no `make.jl` configuration), and
the README floats the same file, so there is one displayed asset and one
master.

Regenerate it with:

```bash
python assets/downscale_logo.py         # 400px, the committed width
python assets/downscale_logo.py 800     # or any other width
```

The script is pure stdlib, so it needs no imaging library. It premultiplies
alpha before averaging — the background is transparent *black*, `(0,0,0,0)`,
so averaging straight RGB would pull every edge pixel toward black and leave a
dark fringe around the artwork.

This directory sits outside `docs/src/` deliberately: everything under
`docs/src/assets/` is copied into the built site, and the 850 KB master has no
reason to be served to readers.
