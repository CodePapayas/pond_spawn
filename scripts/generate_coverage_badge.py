"""Generate a simple SVG coverage badge from a coverage.py XML report."""

from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree


def get_badge_color(coverage_percent: int) -> str:
    """Return a badge color based on the rounded coverage percentage."""
    if coverage_percent >= 90:
        return "#4c1"
    if coverage_percent >= 75:
        return "#97CA00"
    if coverage_percent >= 60:
        return "#dfb317"
    if coverage_percent >= 40:
        return "#fe7d37"
    return "#e05d44"


def load_coverage_percent(report_path: Path) -> int:
    """Extract and round line coverage percentage from coverage.xml."""
    root = ElementTree.parse(report_path).getroot()
    line_rate = root.attrib.get("line-rate")
    if line_rate is None:
        raise ValueError("coverage.xml is missing the 'line-rate' attribute")
    return round(float(line_rate) * 100)


def build_svg(coverage_percent: int) -> str:
    """Return a static SVG badge."""
    label = "coverage"
    value = f"{coverage_percent}%"
    label_width = 74
    value_width = max(46, 10 * len(value) + 10)
    total_width = label_width + value_width
    value_x = label_width + (value_width / 2)
    right_rect_x = label_width
    color = get_badge_color(coverage_percent)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20"
role="img" aria-label="{label}: {value}">
<linearGradient id="smooth" x2="0" y2="100%">
<stop offset="0" stop-color="#fff" stop-opacity=".7"/>
<stop offset=".1" stop-color="#aaa" stop-opacity=".1"/>
<stop offset=".9" stop-opacity=".3"/>
<stop offset="1" stop-opacity=".5"/>
</linearGradient>
<clipPath id="round">
<rect width="{total_width}" height="20" rx="3" fill="#fff"/>
</clipPath>
<g clip-path="url(#round)">
<rect width="{label_width}" height="20" fill="#555"/>
<rect x="{right_rect_x}" width="{value_width}" height="20" fill="{color}"/>
<rect width="{total_width}" height="20" fill="url(#smooth)"/>
</g>
<g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
<text x="{label_width / 2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
<text x="{label_width / 2}" y="14">{label}</text>
<text x="{value_x}" y="15" fill="#010101" fill-opacity=".3">{value}</text>
<text x="{value_x}" y="14">{value}</text>
</g>
</svg>
"""


def main() -> int:
    """Read coverage.xml and write coverage.svg."""
    if len(sys.argv) != 3:
        print("Usage: generate_coverage_badge.py <coverage.xml> <coverage.svg>")
        return 1

    report_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    coverage_percent = load_coverage_percent(report_path)
    output_path.write_text(build_svg(coverage_percent), encoding="utf-8")
    print(f"Wrote {output_path} for {coverage_percent}% coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
