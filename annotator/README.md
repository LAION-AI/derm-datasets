# Expert annotation app

A small local web app in which a dermatologist assesses synthetic images (verdict, realism, diagnosis recognisable, skin type, site and stage correct, free comment, suggested caption) and compares generators blind, side by side.

- `app.py` — server (Python standard library + SQLite). Assessments are append-only.
- `import_items.py` — import a set of images from a manifest (`synth_id`, `path`, `prompt`, `caption`, metadata).
- `import_compare.py` — import a blind comparison set (up to four arms per row).
- `export.py` — export assessments (`--since <ISO date>`, `--summary`).

Run `python annotator/app.py --help` for the options. The app is meant for a trusted local network; it has no user management.
