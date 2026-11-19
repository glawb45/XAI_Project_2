# Poster (24 × 24 in)

`poster.html` is the source; `poster.pdf` is the print-ready export. All numbers, charts and quotes are
read from `../site/data/` (written by `emzoo export-site`), so re-exporting the data and re-printing
keeps the poster in sync with the site.

To rebuild the PDF:

1. From the repo root, start a local server bound to your machine only:
   `python3 -m http.server 8000 --bind 127.0.0.1`
2. Open <http://127.0.0.1:8000/poster/poster.html> in Chrome and wait for the charts to appear.
3. Print → Destination: Save as PDF → Margins: None → More settings: tick "Background graphics".
   The page size (24 × 24 in) comes from the file.

`duke-logo.webp` is the Duke blackletter wordmark (Duke blue, #00539B, transparent background); the
poster's Duke color matches it. The rabbit and its red reflection are an original drawing (a nod to *Us*,
credited as inspiration in the footer), not artwork from the film.
