#!/bin/bash
# Quick smoke test for recommendation API

API_PORT="${RECOMMEND_API_HOST_PORT:-8000}"
BASE_URL="http://localhost:${API_PORT}"

set -euo pipefail

function request() {
  local path="$1"
  local url="${BASE_URL}${path}"
  echo "\n→ GET ${url}"
  curl -sS --fail "${url}"
}

request "/health"
request "/similar/123?top_n=5"
request "/recommend?user_id=1&top_n=5"
