"""Minimal FastAPI service to browse daily recommendation reports."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

from config.settings import DATA_DIR

REPORT_DIR = Path(os.getenv("DAILY_REPORT_DIR", DATA_DIR / "evaluation" / "daily_reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

app = FastAPI(
    title="Recommendation Report Viewer",
    description="Simple viewer for daily recommendation monitoring reports.",
)


def _list_reports(extension: str) -> List[str]:
    if not REPORT_DIR.exists():
        return []
    return sorted(
        [
            path.name
            for path in REPORT_DIR.iterdir()
            if path.is_file() and path.suffix.lower() == extension.lower()
        ],
        reverse=True,
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    html_reports = _list_reports(".html")
    json_reports = _list_reports(".json")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "report_dir": REPORT_DIR,
            "html_reports": html_reports,
            "json_reports": json_reports,
        },
    )


@app.get("/reports/{file_name}")
async def serve_report(file_name: str) -> FileResponse:
    target = REPORT_DIR / file_name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Report not found")

    media_type = "text/html" if target.suffix.lower() == ".html" else "application/json"
    return FileResponse(target, media_type=media_type, filename=target.name)
