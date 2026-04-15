"""
XAI Forensic Terminal — Flask Backend
Reads dynamic JSONL files from ../logs/ and serves alerts via REST API.
Run: python app.py  (from inside the dashboard/ directory)
"""

import json
import os
from pathlib import Path
from collections import Counter

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ── Path resolution ──────────────────────────────────────────────────────────
# Point to the DIRECTORY now, rather than a single hardcoded file
LOGS_DIR = Path(__file__).parent.parent / "logs"


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_available_cases() -> list[str]:
    """Scans the logs folder and returns a list of .jsonl files, newest first."""
    if not LOGS_DIR.exists():
        return []
    
    # Find all jsonl files and sort them by modification time (newest first)
    files = list(LOGS_DIR.glob("*.jsonl"))
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return [f.name for f in files]


def read_alerts(filename: str) -> list[dict]:
    """
    Parse a specific JSON-Lines log file.
    Returns a list of alert dicts, newest first.
    """
    alerts: list[dict] = []
    file_path = LOGS_DIR / filename

    # Security check: Ensure the file exists and is actually inside our logs folder
    if not file_path.exists() or not file_path.resolve().is_relative_to(LOGS_DIR.resolve()):
        return alerts

    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    alerts.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"[WARN] Skipping malformed JSONL line in {filename}: {exc}")
    except OSError as exc:
        print(f"[ERROR] Could not open log file {filename}: {exc}")

    # Reverse so index 0 is the most recent alert
    alerts.reverse()
    return alerts


def compute_stats(alerts: list[dict]) -> dict:
    """Derive the three HUD metrics from the full alert list."""
    total = len(alerts)

    highest_conf = 0.0
    if alerts:
        highest_conf = round(
            max(a.get("confidence_score", 0.0) for a in alerts), 2
        )

    # Normalise dest_port to int strings for counting
    port_tokens: list[str] = []
    for a in alerts:
        raw = a.get("dest_port")
        if raw is not None:
            try:
                port_tokens.append(str(int(float(raw))))
            except (ValueError, TypeError):
                pass

    port_counter = Counter(port_tokens)
    most_targeted = port_counter.most_common(1)[0][0] if port_counter else "N/A"

    return {
        "total": total,
        "highest_confidence": highest_conf,
        "most_targeted_port": most_targeted,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page dashboard."""
    return render_template("index.html")

@app.route("/api/cases")
def list_cases():
    """Returns a list of all available log files for the frontend dropdown."""
    return jsonify({"cases": get_available_cases()})


@app.route("/api/alerts")
def get_alerts():
    """
    Returns JSON:
    {
        "current_case": "filename.jsonl",
        "alerts": [ <newest-first, max 300> ],
        "stats":  { "total", "highest_confidence", "most_targeted_port" }
    }
    """
    cases = get_available_cases()
    
    if not cases:
        return jsonify({
            "current_case": "No logs found",
            "alerts": [],
            "stats": {"total": 0, "highest_confidence": 0.0, "most_targeted_port": "N/A"}
        })

    # Frontend can request a specific case via URL parameter (e.g., /api/alerts?case=monday.jsonl)
    requested_case = request.args.get("case")
    
    # If no case is requested, or it's invalid, default to the most recently modified file
    if not requested_case or requested_case not in cases:
        requested_case = cases[0]

    alerts = read_alerts(requested_case)

    return jsonify({
        "current_case": requested_case,
        "alerts": alerts[:300],          # cap payload for browser performance
        "stats": compute_stats(alerts),  # stats use the FULL list
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[*] XAI Forensic Terminal starting...")
    print(f"[*] Scanning for audit logs in: {LOGS_DIR.resolve()}")
    print(f"[*] Found {len(get_available_cases())} log file(s).")
    app.run(debug=True, port=5000, use_reloader=True)