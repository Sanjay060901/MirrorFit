#!/usr/bin/env bash
# Serve the web prototype at http://localhost:8000 (root = repo, so all stages reachable)
cd "$(dirname "$0")/.." && python3 -m http.server 8000
