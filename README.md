# MirrorFit

Full-body virtual try-on + brand-aware fit intelligence. Browser-first (MediaPipe
Pose + Three.js, all on-device), FastAPI/AWS backend, future in-store smart-mirror kiosk.

## Quick start
1. Open this folder in VS Code (`code .`) — Claude Code reads CLAUDE.md automatically.
2. `bash scripts/serve.sh` then open http://localhost:8000
3. Follow **docs/OPERATION_PLAN.md** stage by stage. Start: `web/stage0/webcam-canvas.html`
4. Known-good reference: `web/reference/glasses-tryon.html` (study it, don't copy-paste it).

## Phone testing
`getUserMedia` needs HTTPS: `ngrok http 8000` and open the https URL on your phone.
