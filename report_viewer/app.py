"""Minimal FastAPI service to browse daily recommendation reports."""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Any

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


def _list_reports(extension: str = ".json") -> List[str]:
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


def _load_report(file_name: str) -> Dict[str, Any]:
    target = REPORT_DIR / file_name
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(file_name)
    with target.open(encoding="utf-8") as stream:
        return json.load(stream)


def _render_report(request: Request, file_name: str, reports: List[str]) -> HTMLResponse:
    try:
        report_data = _load_report(file_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Report not found")
    chart_data_json = json.dumps(report_data.get("history_chart", []))
    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "report_dir": REPORT_DIR,
            "available_reports": reports,
            "selected_report": file_name,
            "report": report_data,
            "chart_data_json": chart_data_json,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def latest(request: Request) -> HTMLResponse:
    reports = _list_reports(".json")
    if not reports:
        return templates.TemplateResponse(
            "report.html",
            {
                "request": request,
                "report_dir": REPORT_DIR,
                "available_reports": [],
                "selected_report": None,
                "report": None,
                "chart_data_json": "[]",
            },
        )
    return _render_report(request, reports[0], reports)


@app.get("/report/{file_name}", response_class=HTMLResponse)
async def report_view(request: Request, file_name: str) -> HTMLResponse:
    reports = _list_reports(".json")
    if file_name not in reports:
        raise HTTPException(status_code=404, detail="Report not found")
    return _render_report(request, file_name, reports)


@app.get("/reports/{file_name}")
async def serve_report(file_name: str) -> FileResponse:
    target = REPORT_DIR / file_name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Report not found")

    media_type = "application/json"
    return FileResponse(target, media_type=media_type, filename=target.name)
