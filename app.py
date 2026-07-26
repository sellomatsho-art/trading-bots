"""Read-only dashboard over the paper-trading ledger.

    python app.py            then open http://localhost:5001

This process does no trading and makes no outbound calls; it renders whatever
`python -m weatherbot scan` has already written to simulation.json.
"""

from __future__ import annotations

from datetime import date

from flask import Flask, jsonify, render_template_string

from weatherbot.calibration import learn_bias
from weatherbot.cities import CITIES
from weatherbot.ledger import Ledger

app = Flask(__name__)

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>weatherbot - paper trading</title>
<style>
  :root { color-scheme: light dark; --bg:#fbfbfa; --fg:#1a1a18; --muted:#6b6b66;
          --line:#e3e3df; --card:#fff; --pos:#0a7d55; --neg:#b4342a; --accent:#3a5a8c; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16161a; --fg:#ececeb; --muted:#9a9a95; --line:#2c2c32;
            --card:#1e1e23; --pos:#42c08b; --neg:#e8695e; --accent:#8fb0e0; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 ui-sans-serif,
         -apple-system, "Segoe UI", Roboto, sans-serif; }
  .wrap { max-width:1080px; margin:0 auto; padding:32px 20px 64px; }
  h1 { font-size:20px; margin:0 0 4px; letter-spacing:-0.01em; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:28px; }
  .badge { display:inline-block; padding:2px 8px; border:1px solid var(--line);
           border-radius:999px; font-size:11px; text-transform:uppercase;
           letter-spacing:.06em; color:var(--muted); }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
           gap:12px; margin-bottom:32px; }
  .tile { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
  .tile .k { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
  .tile .v { font-size:22px; font-variant-numeric:tabular-nums; margin-top:4px; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em;
       color:var(--muted); margin:32px 0 10px; font-weight:600; }
  .scroll { overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:var(--card); }
  table { border-collapse:collapse; width:100%; font-size:13px; min-width:640px; }
  th, td { text-align:left; padding:9px 14px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { font-weight:600; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
  tr:last-child td { border-bottom:none; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; }
  .pos { color:var(--pos); } .neg { color:var(--neg); }
  .q { max-width:380px; overflow:hidden; text-overflow:ellipsis; }
  .empty { padding:20px; color:var(--muted); font-size:13px; }
  footer { margin-top:40px; color:var(--muted); font-size:12px; border-top:1px solid var(--line); padding-top:16px; }
  code { background:var(--bg); padding:1px 5px; border-radius:4px; font-size:12px; }
</style>
</head>
<body><div class="wrap">
  <h1>weatherbot <span class="badge">simulation only</span></h1>
  <div class="sub">Paper positions on Polymarket temperature markets, priced off the 31-member GFS ensemble.
    No wallet is configured and no orders are placed.</div>

  <div class="tiles">
    <div class="tile"><div class="k">Equity</div><div class="v">${{ '%.2f'|format(stats.equity) }}</div></div>
    <div class="tile"><div class="k">Return</div>
      <div class="v {{ 'pos' if stats.return_pct >= 0 else 'neg' }}">{{ '%+.2f'|format(stats.return_pct) }}%</div></div>
    <div class="tile"><div class="k">Realised</div>
      <div class="v {{ 'pos' if stats.realised_pnl >= 0 else 'neg' }}">${{ '%+.2f'|format(stats.realised_pnl) }}</div></div>
    <div class="tile"><div class="k">Open</div><div class="v">{{ stats.open_positions }}</div></div>
    <div class="tile"><div class="k">Settled</div><div class="v">{{ stats.settled_positions }}</div></div>
    <div class="tile"><div class="k">Win rate</div>
      <div class="v">{{ '%.0f%%'|format(stats.win_rate * 100) if stats.win_rate is not none else '--' }}</div></div>
  </div>

  <h2>Open positions</h2>
  <div class="scroll">
  {% if open_positions %}
    <table>
      <tr><th>City</th><th>Date</th><th>Bucket</th><th>Side</th><th class="num">Price</th>
          <th class="num">Model</th><th class="num">Edge</th><th class="num">Stake</th><th>Question</th></tr>
      {% for p in open_positions %}
      <tr><td>{{ p.city_key }}</td><td>{{ p.target_date }}</td><td>{{ p.bucket }}</td>
          <td>{{ p.side }}</td><td class="num">{{ '%.3f'|format(p.price) }}</td>
          <td class="num">{{ '%.3f'|format(p.model_prob) }}</td>
          <td class="num pos">{{ '%+.3f'|format(p.edge) }}</td>
          <td class="num">${{ '%.2f'|format(p.stake) }}</td>
          <td class="q">{{ p.question }}</td></tr>
      {% endfor %}
    </table>
  {% else %}<div class="empty">No open positions. Run <code>python -m weatherbot scan</code>.</div>{% endif %}
  </div>

  <h2>Recently settled</h2>
  <div class="scroll">
  {% if settled %}
    <table>
      <tr><th>City</th><th>Date</th><th>Bucket</th><th>Side</th><th class="num">Actual</th>
          <th class="num">Stake</th><th class="num">PnL</th></tr>
      {% for p in settled %}
      <tr><td>{{ p.city_key }}</td><td>{{ p.target_date }}</td><td>{{ p.bucket }}</td>
          <td>{{ p.side }}</td><td class="num">{{ '%.1f'|format(p.actual_high) }}F</td>
          <td class="num">${{ '%.2f'|format(p.stake) }}</td>
          <td class="num {{ 'pos' if p.pnl >= 0 else 'neg' }}">${{ '%+.2f'|format(p.pnl) }}</td></tr>
      {% endfor %}
    </table>
  {% else %}<div class="empty">Nothing settled yet. Run <code>python -m weatherbot settle</code> after a target date passes.</div>{% endif %}
  </div>

  <h2>Forecast bias by city</h2>
  <div class="scroll">
  {% if biases %}
    <table>
      <tr><th>City</th><th class="num">Samples</th><th class="num">Mean error</th>
          <th class="num">MAE</th><th class="num">Applied</th></tr>
      {% for b in biases %}
      <tr><td>{{ b.city }}</td><td class="num">{{ b.samples }}</td>
          <td class="num {{ 'neg' if b.mean_error > 0 else 'pos' }}">{{ '%+.2f'|format(b.mean_error) }}F</td>
          <td class="num">{{ '%.2f'|format(b.mae) }}F</td>
          <td class="num">{{ '%+.2f'|format(b.applied) }}F</td></tr>
      {% endfor %}
    </table>
  {% else %}<div class="empty">Not enough settled history to measure bias yet.</div>{% endif %}
  </div>

  <footer>Ledger: <code>{{ ledger_path }}</code> &middot; JSON: <code>/api/state</code>
    &middot; {{ city_count }} cities configured &middot; generated {{ today }}</footer>
</div></body></html>
"""


def _bucket_label(low: float, high: float) -> str:
    if low == float("-inf"):
        return f"<={high:.0f}F"
    if high == float("inf"):
        return f">={low:.0f}F"
    if low == high:
        return f"{low:.0f}F"
    return f"{low:.0f}-{high:.0f}F"


def _view(position) -> dict:
    data = position.__dict__.copy()
    data["bucket"] = _bucket_label(position.bucket_low, position.bucket_high)
    return data


def _state() -> dict:
    ledger = Ledger.load()
    settled = sorted(ledger.settled_positions(), key=lambda p: p.settled_at or "", reverse=True)
    biases = [
        {
            "city": b.city_key,
            "samples": b.samples,
            "mean_error": b.mean_error,
            "mae": b.mean_absolute_error,
            "applied": b.applied_bias,
        }
        for b in learn_bias(ledger.forecasts, min_samples=3).values()
    ]
    return {
        "stats": ledger.stats(),
        "open_positions": [_view(p) for p in ledger.open_positions()],
        "settled": [_view(p) for p in settled[:25]],
        "biases": biases,
        "ledger_path": str(ledger.path),
    }


@app.route("/")
def home():
    state = _state()
    return render_template_string(
        PAGE,
        today=date.today().isoformat(),
        city_count=len(CITIES),
        **state,
    )


@app.route("/api/state")
def api_state():
    return jsonify(_state())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
