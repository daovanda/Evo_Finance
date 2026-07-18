"""Tiny live dashboard for crypto production predictions.

Run:
    python -m crypto.prod.live.app --model-dir crypto/prod/model/crypto_btc_seed1_12h

The backend can also be run separately:
    python -m crypto.prod.live_backend --model-dir crypto/prod/model/crypto_btc_seed1_12h --loop
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from crypto import config
from crypto.prod.live_backend import DEFAULT_OUTPUT_PATH, run_once


DEFAULT_TRADE_STATE_PATH = Path("crypto/prod/live/trade_state.json")
_RUN_LOCK = threading.Lock()
_RUNNING = False
_LAST_ERROR: str | None = None


def serve(
    host: str,
    port: int,
    prediction_path: str | Path = DEFAULT_OUTPUT_PATH,
    trade_state_path: str | Path = DEFAULT_TRADE_STATE_PATH,
    model_dir: str | Path | None = None,
    data_path: str | Path = config.DATA_PATH,
) -> None:
    prediction_path = Path(prediction_path)
    trade_state_path = Path(trade_state_path)
    model_dir = Path(model_dir) if model_dir else None
    data_path = Path(data_path)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib hook
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(_render_page(prediction_path, trade_state_path, model_dir))
                return
            if parsed.path == "/api/latest":
                self._send_json(_read_prediction(prediction_path, trade_state_path))
                return
            if parsed.path == "/run-once":
                qs = parse_qs(parsed.query)
                override_model_dir = qs.get("model_dir", [None])[0]
                target_model_dir = Path(override_model_dir) if override_model_dir else model_dir
                self._send_json(
                    _start_run_once(
                        model_dir=target_model_dir,
                        data_path=data_path,
                        prediction_path=prediction_path,
                    )
                )
                return
            self.send_error(404, "Not found")

        def log_message(self, fmt: str, *args: object) -> None:
            return

        def _send_json(self, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, int(port)), Handler)
    print(f"Crypto live dashboard: http://{host}:{port}")
    server.serve_forever()


def _start_run_once(
    model_dir: Path | None,
    data_path: Path,
    prediction_path: Path,
) -> dict:
    global _RUNNING, _LAST_ERROR
    if model_dir is None:
        return {"ok": False, "error": "No --model-dir was provided for run-once."}
    with _RUN_LOCK:
        if _RUNNING:
            return {"ok": False, "running": True, "error": "A live update is already running."}
        _RUNNING = True
        _LAST_ERROR = None

    def worker() -> None:
        global _RUNNING, _LAST_ERROR
        try:
            run_once(model_dir=model_dir, data_path=data_path, output_path=prediction_path)
        except Exception as exc:  # pragma: no cover - UI convenience path
            _LAST_ERROR = str(exc)
        finally:
            with _RUN_LOCK:
                _RUNNING = False

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True, "running": True, "message": "Live update started."}


def _read_prediction(path: Path, trade_state_path: Path) -> dict:
    payload = {
        "exists": path.exists(),
        "path": str(path),
        "trade_state_path": str(trade_state_path),
        "running": _RUNNING,
        "last_error": _LAST_ERROR,
    }
    if not path.exists():
        return payload
    try:
        payload["prediction"] = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        payload["error"] = str(exc)
    if trade_state_path.exists():
        try:
            payload["trade_state"] = json.loads(trade_state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload["trade_state_error"] = str(exc)
    return payload


def _render_page(prediction_path: Path, trade_state_path: Path, model_dir: Path | None) -> str:
    model_dir_text = str(model_dir) if model_dir else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crypto Live Prediction</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #161a1d;
      --muted: #667085;
      --line: #d8dee8;
      --blue: #1f77b4;
      --orange: #ff7f0e;
      --green: #0a8f4b;
      --red: #c23b22;
    }}
    body {{
      margin: 0;
      font-family: Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }}
    h1 {{
      font-size: 24px;
      margin: 0;
    }}
    button {{
      border: 1px solid #111827;
      background: #111827;
      color: white;
      border-radius: 6px;
      padding: 9px 14px;
      cursor: pointer;
      font-weight: 600;
    }}
    button:disabled {{
      opacity: 0.55;
      cursor: wait;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }}
    .card, table {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }}
    .card {{
      padding: 14px;
    }}
    .decision {{
      margin-top: 14px;
      padding: 16px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: var(--panel);
    }}
    .decision-title {{
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .error-card {{
      margin-top: 12px;
    }}
    .status-trade {{
      color: var(--green);
    }}
    .status-no-trade {{
      color: var(--red);
    }}
    .label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 4px;
    }}
    .value {{
      font-size: 20px;
      font-weight: 700;
      word-break: break-word;
    }}
    .small {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }}
    .signal {{
      color: var(--green);
    }}
    .nosignal {{
      color: var(--red);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      margin-top: 12px;
    }}
    th, td {{
      padding: 10px 9px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #202833;
      color: white;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Crypto Live Prediction</h1>
      <div class="small">Prediction JSON: {prediction_path}</div>
      <div class="small">Model dir: {model_dir_text or "not set"}</div>
      <div class="small">Trade state: {trade_state_path}</div>
    </div>
    <button id="runBtn" onclick="runOnce()">Run Update</button>
  </header>
  <section id="entries"></section>
  <section class="decision">
    <div class="decision-title">Decision</div>
    <div class="grid" id="decision"></div>
    <div id="errorBox"></div>
  </section>
  <section class="decision">
    <div class="decision-title">Trade Bot</div>
    <div class="grid" id="tradeState"></div>
    <div id="tradeErrorBox"></div>
  </section>
</main>
<script>
const fmt = (v, digits=6) => typeof v === 'number' ? v.toFixed(digits) : (v ?? '');
const pct = (v) => typeof v === 'number' ? (v * 100).toFixed(2) + '%' : '';

async function load() {{
  const res = await fetch('/api/latest?ts=' + Date.now());
  const data = await res.json();
  document.getElementById('runBtn').disabled = !!data.running;
  render(data);
}}

async function runOnce() {{
  document.getElementById('runBtn').disabled = true;
  await fetch('/run-once', {{cache: 'no-store'}});
  setTimeout(load, 1000);
}}

function render(data) {{
  const payload = data.prediction || {{}};
  const entries = payload.entries || [];
  const error = data.last_error || payload.error || data.error || '';
  const finalEnsemble = payload.final_ensemble || null;
  const hasTrade = finalEnsemble
    ? finalEnsemble.ensemble_signal === true
    : entries.some(entry => entry.ensemble_signal === true);
  const status = error ? 'ERROR' : (data.running ? 'running' : (hasTrade ? 'TRADE' : 'NO TRADE'));
  const statusClass = hasTrade && !error ? 'status-trade' : 'status-no-trade';

  const finalHtml = finalEnsemble ? `
    <table>
      <thead>
        <tr><th colspan="5">Final ensemble | members: ${{finalEnsemble.member_count}}
          | signal: <span class="${{finalEnsemble.ensemble_signal ? 'signal' : 'nosignal'}}">${{finalEnsemble.ensemble_signal}}</span>
          | pred mean ${{fmt(finalEnsemble.pred_mean, 6)}}
        </th></tr>
        <tr><th>Member</th><th>Rank</th><th>Label</th><th>Signal</th><th>Pred mean</th></tr>
      </thead>
      <tbody>
        ${{(finalEnsemble.members || []).map(m => `
          <tr>
            <td>${{m.entry_id || ''}}</td>
            <td>${{m.rank ?? ''}}</td>
            <td>${{m.label_mode || ''}} / ${{m.label_direction || 'long'}} @ ${{fmt(m.label_threshold, 4)}}</td>
            <td class="${{m.ensemble_signal ? 'signal' : 'nosignal'}}">${{m.ensemble_signal}}</td>
            <td>${{fmt(m.pred_mean, 6)}}</td>
          </tr>
        `).join('')}}
      </tbody>
    </table>
  ` : '';

  const entryHtml = entries.map(entry => `
    <table>
      <thead>
        <tr><th colspan="5">Rank ${{entry.rank}} | direction: ${{entry.label_direction || 'long'}} | ensemble:
          <span class="${{entry.ensemble_signal ? 'signal' : 'nosignal'}}">${{entry.ensemble_signal}}</span>
          | pred mean ${{fmt(entry.pred_mean, 6)}}
        </th></tr>
        <tr><th>Horizon</th><th>Prediction</th><th>Threshold</th><th>Signal</th><th>Model</th></tr>
      </thead>
      <tbody>
        ${{(entry.predictions || []).map(p => `
          <tr>
            <td>h${{p.horizon}}</td>
            <td>${{fmt(p.pred, 6)}}</td>
            <td>${{fmt(p.threshold, 6)}}</td>
            <td class="${{p.is_signal ? 'signal' : 'nosignal'}}">${{p.is_signal}}</td>
            <td>${{p.model_path}}</td>
          </tr>
        `).join('')}}
      </tbody>
    </table>
  `).join('');
  document.getElementById('entries').innerHTML = finalHtml + entryHtml;

  const decision = [
    ['Signal time', payload.signal_time],
    ['Entry candle', payload.entry_candle_time],
    ['Entry open', fmt(payload.entry_open, 2)],
    ['Final ensemble', finalEnsemble ? finalEnsemble.ensemble_signal : ''],
    ['Status', status, statusClass]
  ];
  document.getElementById('decision').innerHTML = decision.map(([k, v, cls]) => `
    <div class="card"><div class="label">${{k}}</div><div class="value ${{cls || ''}}">${{v ?? ''}}</div></div>
  `).join('');
  document.getElementById('errorBox').innerHTML = `
    <div class="card error-card">
      <div class="label">Error</div>
      <div class="value status-no-trade">${{error || ''}}</div>
    </div>
  `;

  const trade = data.trade_state || {{}};
  const tradeError = trade.error || data.trade_state_error || '';
  const tradeRows = [
    ['Bot status', trade.status],
    ['Position open', trade.position_open],
    ['Blocked', trade.block_new_trades],
    ['Qty', trade.qty],
    ['Avg entry', trade.avg_entry_price],
    ['TP price', trade.take_profit_price]
  ];
  document.getElementById('tradeState').innerHTML = tradeRows.map(([k, v]) => `
    <div class="card"><div class="label">${{k}}</div><div class="value">${{v ?? ''}}</div></div>
  `).join('');
  document.getElementById('tradeErrorBox').innerHTML = `
    <div class="card error-card">
      <div class="label">Trade Error</div>
      <div class="value status-no-trade">${{tradeError || ''}}</div>
    </div>
  `;
}}

load();
setInterval(load, 10000);
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--prediction", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--trade-state", default=str(DEFAULT_TRADE_STATE_PATH))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--data", default=str(config.DATA_PATH))
    args = parser.parse_args()

    serve(
        host=args.host,
        port=args.port,
        prediction_path=args.prediction,
        trade_state_path=args.trade_state,
        model_dir=args.model_dir,
        data_path=args.data,
    )


if __name__ == "__main__":
    main()
