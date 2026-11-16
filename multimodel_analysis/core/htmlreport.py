"""Builds the single-file HTML report (no external files: plots are embedded as base64 PNG, CSS is inline)."""
import base64
import html
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import pandas as pd

_CSS = """
body{font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#222;background:#fff;
     max-width:980px;margin:32px auto;padding:0 20px;line-height:1.5}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:18px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid #ddd}
.sub{color:#666;font-size:14px;margin-bottom:18px}
.best{background:#eef4f8;border-left:4px solid #1f5f8b;padding:10px 14px;margin:16px 0;font-size:15px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid #e5e5e5;white-space:nowrap}
th{background:#f5f7f9;font-weight:600}
th:first-child,td:first-child{text-align:left}
tr:first-child td{font-weight:600}
.scroll{overflow-x:auto}
img{max-width:100%;height:auto;margin:6px 0}
ul{font-size:13px;color:#444}
.foot{margin-top:40px;color:#888;font-size:12px}
"""


def _png_tag(png: bytes, alt: str) -> str:
    data = base64.b64encode(png).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def _table(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.1f}"
        return html.escape(str(v))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in row) + "</tr>"
                   for row in df.itertuples(index=False, name=None))
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def build_html_report(title: str, subtitle: str, table: pd.DataFrame, best: Optional[str],
                      plots: Sequence[Tuple[str, bytes]], notes: Sequence[Tuple[str, List[str]]],
                      version: str) -> str:
    """Returns a complete HTML document.

    ``plots`` is a list of (heading, png bytes); ``notes`` a list of (heading, bullet lines).
    """
    parts = [f"<h1>{html.escape(title)}</h1>", f'<div class="sub">{html.escape(subtitle)}</div>']
    if best:
        parts.append(f'<div class="best">Top-ranked model: <b>{html.escape(best)}</b></div>')
    parts.append("<h2>Results</h2>")
    parts.append(_table(table))
    for heading, png in plots:
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        parts.append(_png_tag(png, heading))
    for heading, lines in notes:
        if lines:
            parts.append(f"<h2>{html.escape(heading)}</h2><ul>"
                         + "".join(f"<li>{html.escape(line)}</li>" for line in lines) + "</ul>")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts.append(f'<div class="foot">Created by multimodel_analysis {html.escape(version)} on {stamp}</div>')
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
            + "".join(parts) + "</body></html>")
