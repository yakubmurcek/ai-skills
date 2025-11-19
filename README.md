# AI Skills Analyzer

Evaluate job descriptions for AI-related skills using a reproducible CLI
workflow. The project ships with a small orchestration layer so you and your
advisor can prepare sample data and run the full OpenAI-powered analysis
without guessing which script to execute.

## 1. Environment setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file (or edit `ai_skills/config_settings.py`) with a valid
   `OPENAI_API_KEY`.
3. Place the raw CSV file under `data/inputs/us_relevant.csv` or adjust the
   paths inside `ai_skills/config_settings.py`.

## 2. Command-line usage

The project exposes a single CLI with two subcommands. You can invoke it via
`python -m ai_skills.cli …` or call `python main.py …` if you prefer the
historical entry point.

### Prepare a graded sample

Create a smaller CSV that downstream steps will consume:

```bash
python -m ai_skills.cli prepare-inputs --rows 100
```

Options:

- `--source`: raw CSV to sample (default `data/inputs/us_relevant.csv`)
- `--destination`: output CSV that the pipeline will read
  (default: same folder as `--source` with `_<rows>` appended to the stem)
- `--rows`: number of rows to copy

### Run the OpenAI analysis

```bash
python -m ai_skills.cli analyze
```

By default this shows a progress bar and writes the enriched file specified by
`OUTPUT_CSV` in `ai_skills/config_settings.py`. Add `--no-progress` if you are piping
logs to a file, `--input-csv path/to/file.csv` to analyze a different dataset, and
`--output-csv path/to/result.csv` to pick a custom destination. When you only set
`--input-csv`, the CLI automatically writes to
`data/outputs/<input_stem>_ai<suffix>`.

### Legacy shorthand

`python main.py` now delegates to the same CLI and still runs the `analyze`
command so existing instructions keep working. You can also forward options, e.g.

```bash
python main.py analyze --no-progress
```

Run `python -m ai_skills.cli --help` for the full command reference.
