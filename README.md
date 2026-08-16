# Agentic Job Hunter

## Flow

- If a company has a saved `playwright` recipe in `companies.json`, Playwright is used first.
- If the recipe fails, `browser-use` explores the careers page again.
- The discovered Playwright recipe is saved back into `companies.json`.
- Job results are written to `jobs.csv`.
- Location filtering uses each company's `region` field only.

## Run

```powershell
python find.py
```

## Dependencies

```powershell
pip install browser-use playwright
playwright install chromium
```

The local LLM configuration is in `find.py`.
