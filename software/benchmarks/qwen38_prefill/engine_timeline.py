#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Offline engine-occupancy timeline from GVSoC schema-v1 telemetry.

RedMulE GEMM and iDMA backend intervals are exact at the reported model
boundary. Spatz reports cumulative activity periodically; its busy cycles are
distributed proportionally across each sample window only for display. Keep
the observed coverage alongside that estimate instead of treating an unsampled
tail as idle. This page complements the vendored dashboard's FPU/MSHR/NoC
overview without giving those unavailable counters invented values.
"""

import argparse
import html
import json
from pathlib import Path


def load(path):
    with Path(path).open() as stream:
        meta = json.loads(next(stream))
        if meta.get("kind") != "meta" or meta.get("schema_version") != 1:
            raise ValueError("Expected schema-v1 meta as the first record")
        records = [json.loads(line) for line in stream]
    if not records:
        raise ValueError("No activity records")
    return meta, records


def bin_activity(meta, records, width):
    """Return machine occupancy; Spatz has an explicit sample coverage."""
    if width <= 0:
        raise ValueError("Bin width must be positive")
    first = min(row["start"] for row in records)
    last = max(row["end"] for row in records)
    origin = first // width * width
    count = (last - origin + width - 1) // width
    if count > 2000:
        raise ValueError("Use a wider bin to keep the page bounded")
    mesh = meta["mesh"][0] * meta["mesh"][1]
    installed_rm = mesh * meta["tiles_per_group"]
    installed_spatz = installed_rm * meta["cores_per_tile"]
    dma_channels = {row["channel"] for row in records
                    if row["kind"] == "dma" and "channel" in row}
    installed_dma = mesh * (max(dma_channels, default=0) // mesh + 1)
    if dma_channels and meta.get("dma", {}).get("channels"):
        installed_dma = max(installed_dma, len(meta["dma"]["channels"]))
    bins = []
    for index in range(count):
        start = origin + index * width
        bins.append({"start": start, "end": min(last, start + width),
                     "rm_busy": 0.0, "spatz_busy": 0.0,
                     "spatz_covered": 0.0, "dma_active": 0.0})
    for row in records:
        kind = row["kind"]
        label = row.get("label")
        if kind != "dma" and (kind != "stage" or
                              label not in ("redmule array", "spatz vector")):
            continue
        start, end = row["start"], row["end"]
        if end <= start:
            raise ValueError("Nonpositive activity interval")
        for index in range((start - origin) // width,
                           (end - 1 - origin) // width + 1):
            bucket = bins[index]
            overlap = min(end, bucket["end"]) - max(start, bucket["start"])
            if kind == "dma":
                bucket["dma_active"] += overlap
            elif label == "redmule array":
                if row["hsk"] + row["stall"] != end - start:
                    raise ValueError("RedMulE busy interval is incomplete")
                bucket["rm_busy"] += overlap
            else:
                busy = row["hsk"] + row["stall"]
                if busy < 0 or busy > end - start:
                    raise ValueError("Invalid Spatz sampled activity")
                bucket["spatz_busy"] += busy * overlap / (end - start)
                bucket["spatz_covered"] += overlap
    for bucket in bins:
        duration = bucket["end"] - bucket["start"]
        bucket["redmule"] = bucket.pop("rm_busy") / (installed_rm * duration)
        covered = bucket.pop("spatz_covered")
        bucket["spatz_coverage"] = covered / (installed_spatz * duration)
        spatz_busy = bucket.pop("spatz_busy")
        dma_active = bucket.pop("dma_active")
        bucket["spatz"] = spatz_busy / covered if covered else None
        bucket["dma"] = (dma_active / (installed_dma * duration)
                         if dma_channels else None)
        for name in ("redmule", "spatz", "dma", "spatz_coverage"):
            value = bucket[name]
            if value is not None and not -1e-9 <= value <= 1.0 + 1e-9:
                raise ValueError("Invalid machine occupancy in " + name)
    return {"model_cycle_range": [origin, last], "bin_cycles": width,
            "installed": {"redmule": installed_rm, "spatz": installed_spatz,
                          "dma": installed_dma if dma_channels else None},
            "spatz_method": "proportional within each reported sample window",
            "bins": bins}


def chart(name, title, bins, color):
    size = len(bins)
    plot_left, plot_top, plot_width, plot_height = 75, 20, 1100, 170
    lines = [f'<h2>{html.escape(title)}</h2>',
             '<svg viewBox="0 0 1200 225" preserveAspectRatio="none" '
             'role="img" style="width:100%;height:235px">']
    for percent in (0, 25, 50, 75, 100):
        y = plot_top + plot_height * (1 - percent / 100)
        lines.append(f'<path d="M{plot_left} {y}h{plot_width}" '
                     'stroke="#dde5ec" stroke-width="1"/>')
        lines.append(f'<text x="65" y="{y+4}" text-anchor="end" '
                     f'font-size="12">{percent}%</text>')
    bar_width = plot_width / size
    for index, bucket in enumerate(bins):
        value = bucket[name]
        if value is None:
            continue
        x = plot_left + index * bar_width
        height = max(0, min(1, value)) * plot_height
        y = plot_top + plot_height - height
        detail = (f'{bucket["start"]:,}–{bucket["end"]:,} cycles: '
                  f'{value * 100:.2f}%')
        if name == "spatz":
            detail += (f' (sample coverage '
                       f'{bucket["spatz_coverage"] * 100:.1f}%)')
        lines.append(f'<rect x="{x:.3f}" y="{y:.3f}" '
                     f'width="{max(0.8, bar_width):.3f}" '
                     f'height="{height:.3f}" fill="{color}" '
                     f'data-bin="{index}"><title>{html.escape(detail)}</title>'
                     '</rect>')
    for position in (0, .25, .5, .75, 1):
        index = min(size - 1, int((size - 1) * position))
        x = plot_left + plot_width * position
        label = f'{bins[index]["start"] / 1e6:.1f}M'
        lines.append(f'<text x="{x:.1f}" y="215" text-anchor="middle" '
                     f'font-size="12">{label}</text>')
    lines.append('</svg>')
    return "\n".join(lines)


def render(meta, activity, detail_link=None):
    bins = activity["bins"]
    sections = [chart("redmule", "RedMulE busy / installed engines", bins,
                      "#d8763f"),
                chart("spatz", "Spatz busy / sampled unit-cycles", bins,
                      "#3389b5")]
    if activity["installed"]["dma"]:
        sections.append(chart("dma", "iDMA backend active / channels", bins,
                              "#5b9b63"))
    else:
        sections.append('<h2>iDMA</h2><p>No iDMA activity records were '
                        'captured by this model snapshot.</p>')
    detail = (f'<p><a href="{html.escape(detail_link, quote=True)}">'
              'Open the original detailed dashboard</a></p>'
              if detail_link else '')
    timer = meta.get("teranoc", {}).get("guest_timer_window")
    timer_note = (
        f'The guest timer spans model cycles {timer[0]:,}–{timer[1]:,}.'
        if timer else 'This guest did not print an absolute timer start, so '
        'the axis includes model startup.')
    data = json.dumps(activity, separators=(",", ":")).replace("<", "\\u003c")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1">
<title>Engine occupancy timeline</title>
<style>body{{font:16px system-ui,sans-serif;color:#173047;max-width:1260px;
margin:32px auto;padding:0 20px}}h1{{margin-bottom:4px}}h2{{margin-top:28px}}
svg{{background:#f8fafc;border:1px solid #dbe4eb;border-radius:8px}}
rect[data-bin]{{cursor:pointer}}rect[data-bin]:hover{{opacity:.6}}
#detail{{background:#eef6fa;padding:12px;border-radius:8px;min-height:24px}}
.note{{color:#4e6576;line-height:1.45}}</style></head><body>
<h1>Engine occupancy over model time</h1>
<p class="note">{html.escape(meta.get("name", "GVSoC engine activity"))};
{activity["installed"]["redmule"]} RedMulE engines,
{activity["installed"]["spatz"]} Spatz units. Each bar covers
{activity["bin_cycles"]:,} model cycles. RedMulE and iDMA intervals are exact
at their model boundaries. Spatz busy time is distributed proportionally
within each periodic sample; click a bar to see its sample coverage. Missing
NoC, bank and MSHR counters are not inferred. {html.escape(timer_note)}</p>
{detail}<div id="detail">Select a bar for its measured interval.</div>
{''.join(sections)}
<script type="application/json" id="data">{data}</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
document.querySelectorAll('[data-bin]').forEach(el=>el.onclick=()=>{{
  const b=D.bins[Number(el.dataset.bin)];
  const fmt=(v)=>v==null?'not recorded':(100*v).toFixed(2)+'%';
  document.getElementById('detail').textContent=
    `${{b.start.toLocaleString()}}–${{b.end.toLocaleString()}} cycles · `+
    `RedMulE ${{fmt(b.redmule)}} · Spatz ${{fmt(b.spatz)}} `+
    `(sample coverage ${{fmt(b.spatz_coverage)}}) · iDMA ${{fmt(b.dma)}}`;
}});
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--bin-cycles", type=int, default=100000)
    parser.add_argument("--detail-link")
    args = parser.parse_args()
    meta, records = load(args.telemetry)
    activity = bin_activity(meta, records, args.bin_cycles)
    args.out.write_text(render(meta, activity, args.detail_link))
    if args.json_out:
        args.json_out.write_text(json.dumps(activity, indent=2) + "\n")
    print(f"{args.out}: {len(activity['bins'])} source-backed bins")


if __name__ == "__main__":
    main()
