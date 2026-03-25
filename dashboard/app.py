"""
XAI Forensic Terminal — Flask Backend
Reads ../logs/forensics_log.jsonl and serves alerts via REST API.
Run: python app.py  (from inside the dashboard/ directory)
"""

import json
from pathlib import Path
from collections import Counter

from flask import Flask, jsonify, render_template

app = Flask(__name__)

# ── Path resolution ──────────────────────────────────────────────────────────
# This file lives at  <project>/dashboard/app.py
# The log lives at    <project>/logs/forensics_log.jsonl
LOG_PATH = Path(__file__).parent.parent / "logs" / "forensics_log.jsonl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_alerts() -> list[dict]:
    """
    Parse the JSON-Lines log file.
    Returns a list of alert dicts, newest first.
    """
    alerts: list[dict] = []

    if not LOG_PATH.exists():
        return alerts

    try:
        with open(LOG_PATH, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    alerts.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"[WARN] Skipping malformed JSONL line: {exc}")
    except OSError as exc:
        print(f"[ERROR] Could not open log file: {exc}")

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


@app.route("/api/alerts")
def get_alerts():
    """
    Returns JSON:
    {
        "alerts": [ <newest-first, max 300> ],
        "stats":  { "total", "highest_confidence", "most_targeted_port" }
    }
    """
    alerts = read_alerts()

    return jsonify({
        "alerts": alerts[:300],          # cap payload for browser performance
        "stats": compute_stats(alerts),  # stats use the FULL list
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[*] XAI Forensic Terminal starting...")
    print(f"[*] Reading alerts from: {LOG_PATH.resolve()}")
    print(f"[*] Log file exists: {LOG_PATH.exists()}")
    app.run(debug=True, port=5000, use_reloader=True)
