# Dataset Recommendation Prototype

This project provides a lightweight recommendation pipeline for dataset detail pages. It covers data extraction, feature generation, model training, and an online API service built with FastAPI.

## Project Structure

```
app/                # FastAPI service
config/             # Configuration helpers
pipeline/           # Data processing and training scripts
models/             # Generated model artifacts
data/               # Raw and processed data outputs (generated)
docs/               # Architecture documentation
scripts/            # Utility scripts
```

## Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set database environment variables** (examples)
   ```bash
   export BUSINESS_DB_HOST=localhost
   export BUSINESS_DB_NAME=dianshu_backend
   export BUSINESS_DB_USER=root
   export BUSINESS_DB_PASSWORD=secret
   export MATOMO_DB_HOST=localhost
   export MATOMO_DB_NAME=matomo
   export MATOMO_DB_USER=root
   export MATOMO_DB_PASSWORD=secret
   ```

3. **Run the data pipeline**
   ```bash
   scripts/run_pipeline.sh --dry-run  # inspect planned extraction
   scripts/run_pipeline.sh            # execute full pipeline
   ```

4. **Start the API service**
   ```bash
   uvicorn app.main:app --reload
   ```

   Available endpoints:
   - `GET /health`
   - `GET /similar/{dataset_id}?limit=10`
   - `GET /recommend/detail/{dataset_id}?user_id=123&limit=10`

## Evaluation

Use `python3 pipeline/evaluate.py` (to be implemented) to compare recommendations with Matomo behaviour logs for click/conversion analysis.

Refer to `docs/ARCHITECTURE.md` for more details on the system design and next steps.
