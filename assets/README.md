# Logo source

`logo-source.svg` is the master: the octopus as an embedded 1536 × 1280 raster,
plus the "DataMimic.jl" wordmark as 12 real vector paths. Nothing references it
at build time — it lives here so the displayed assets can be regenerated.

Three displayed assets are derived from it, all in `docs/src/assets/`:

| File | Size | Used by |
|---|---|---|
| `logo.png` | 400 × 333 | Docs navbar (`Material3(logo=…)`) — the octopus alone |
| `favicon.png` | 256 × 256 | Docs favicon (`Material3(favicon=…)`) — the octopus, squared |
| `logo-wordmark.png` | 400 × 368 | README header — octopus and wordmark |

The docs get the octopus **without** the wordmark on purpose. The navbar
renders the logo 32 px tall and already prints "DataMimic.jl" beside it, so the
wordmark would be both illegible at that size and redundant next to the name.
The README shows it at 320 px, where the wordmark reads clearly and carries the
title.

## Regenerating

Serve the repository over HTTP and open the tool — browsers block both `fetch`
and canvas export on `file://`, so opening it straight from disk will not work:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000/assets/rasterize_logo.html>, pick a width and
variant, and download. It derives the mark by dropping the wordmark group and
cropping to the octopus's own bounds, reading that geometry from the SVG rather
than hard-coding numbers that would rot on re-export.

Any real SVG rasterizer works for the full logo:

```bash
rsvg-convert -w 400 assets/logo-source.svg -o docs/src/assets/logo-wordmark.png
```

## Notes

This directory sits outside `docs/src/` deliberately: everything under
`docs/src/assets/` is copied into the built site, and the megabyte master has
no reason to be served to readers.

The SVG as exported carried its embedded PNG **twice** — once in `<defs>` under
an id nothing referenced, and once in the body — along with an empty Inkscape
flowed-text box. Dropping the unreferenced copy took it from 2247 KB to
1128 KB, and rendered output was verified pixel-identical before and after
(0 differing pixels of 147,200 at 400 px). If the logo is ever re-exported from
a vector editor, check for that duplicate again.
