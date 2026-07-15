"""Generate the workflow SVG used by the MkDocs site.

This script writes a static SVG so the workflow diagram renders cleanly in
both the web docs and the PDF export.
"""

# --------------------------------------------------
# import necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


# --------------------------------------------------
# constants
# --------------------------------------------------
OUTPUT_PATH = Path("docs/assets/workflow.svg")

WIDTH = 1220
HEIGHT = 420

BOX_WIDTH = 146
BOX_HEIGHT = 70
BOX_RX = 16

LEFT_MARGIN = 40
TOP_ROW_Y = 96
MID_ROW_Y = 186
BOTTOM_ROW_Y = 276

BACKGROUND = "#f8fafc"
TITLE_COLOR = "#233142"
TEXT_COLOR = "#334155"
MUTED_COLOR = "#64748b"
STROKE_COLOR = "#b7c6d6"
ARROW_COLOR = "#5d738a"

TOP_FILL = "#dcecff"
BOTTOM_FILL = "#f3f6fa"
HIGHLIGHT_FILL = "#e5f5ea"
CARD_STROKE = "#d6e0ea"
SEED_PILL_FILL = "#edf7f0"
SEED_PILL_STROKE = "#b7d8c0"
SEED_PILL_TEXT = "#2f5f45"


# --------------------------------------------------
# helpers
# --------------------------------------------------
def box_svg(
    x: int,
    y: int,
    label: str | list[str],
    fill: str,
    *,
    subtitle: str | list[str] | None = None,
    dashed: bool = False,
) -> str:
    """Build one rounded SVG box with centered text.

    Args:
        x: Left position in pixels.
        y: Top position in pixels.
        label: Main label shown inside the box.
        fill: Fill color for the rectangle.
        subtitle: Optional second line.

    Returns:
        SVG fragment as a string.
    """

    # build
    if isinstance(label, str):
        label_lines = [label]
    else:
        label_lines = label

    svg_lines: list[str] = []

    stroke_dash = ' stroke-dasharray="7 5"' if dashed else ""
    svg_lines.append(
        f'<rect x="{x}" y="{y}" width="{BOX_WIDTH}" height="{BOX_HEIGHT}" '
        f'rx="{BOX_RX}" fill="{fill}" stroke="{STROKE_COLOR}" stroke-width="1.4"{stroke_dash} />'
    )

    if subtitle is None:
        if len(label_lines) == 1:
            label_text = escape(label_lines[0])
            svg_lines.append(
                f'<text x="{x + BOX_WIDTH / 2:.1f}" y="{y + 34}" text-anchor="middle" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="19" '
                f'font-weight="600" fill="{TITLE_COLOR}">{label_text}</text>'
            )
        else:
            first_line = escape(label_lines[0])
            second_line = escape(label_lines[1])
            svg_lines.append(
                f'<text x="{x + BOX_WIDTH / 2:.1f}" y="{y + 27}" text-anchor="middle" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="17" '
                f'font-weight="600" fill="{TITLE_COLOR}">{first_line}</text>'
            )
            svg_lines.append(
                f'<text x="{x + BOX_WIDTH / 2:.1f}" y="{y + 46}" text-anchor="middle" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="17" '
                f'font-weight="600" fill="{TITLE_COLOR}">{second_line}</text>'
            )
        return "\n".join(svg_lines)

    if isinstance(subtitle, str):
        subtitle_lines = [subtitle]
    else:
        subtitle_lines = subtitle

    if len(label_lines) == 1:
        label_text = escape(label_lines[0])
        svg_lines.append(
            f'<text x="{x + BOX_WIDTH / 2:.1f}" y="{y + 27}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="18" '
            f'font-weight="600" fill="{TITLE_COLOR}">{label_text}</text>'
        )
    else:
        first_line = escape(label_lines[0])
        second_line = escape(label_lines[1])
        svg_lines.append(
            f'<text x="{x + BOX_WIDTH / 2:.1f}" y="{y + 23}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="16" '
            f'font-weight="600" fill="{TITLE_COLOR}">{first_line}</text>'
        )
        svg_lines.append(
            f'<text x="{x + BOX_WIDTH / 2:.1f}" y="{y + 40}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="16" '
            f'font-weight="600" fill="{TITLE_COLOR}">{second_line}</text>'
        )

    if subtitle_lines is not None:
        subtitle_start_y = y + 56
        subtitle_line_height = 12
        for subtitle_index, subtitle_line in enumerate(subtitle_lines):
            subtitle_text = escape(subtitle_line)
            subtitle_y = subtitle_start_y + subtitle_index * subtitle_line_height
            svg_lines.append(
                f'<text x="{x + BOX_WIDTH / 2:.1f}" y="{subtitle_y}" text-anchor="middle" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="11.5" '
                f'fill="{MUTED_COLOR}">{subtitle_text}</text>'
            )

    return "\n".join(svg_lines)


def arrow_svg(x1: int, y1: int, x2: int, y2: int, *, dashed: bool = False) -> str:
    """Build a straight SVG arrow.

    Args:
        x1: Start x position.
        y1: Start y position.
        x2: End x position.
        y2: End y position.

    Returns:
        SVG fragment as a string.
    """

    # build
    stroke_dash = ' stroke-dasharray="7 5"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{ARROW_COLOR}" stroke-width="2.4"{stroke_dash} marker-end="url(#arrowhead)" />'
    )


def label_svg(x: int, y: int, text: str | list[str]) -> str:
    """Build muted annotation text above a branch arrow."""

    # build line list
    if isinstance(text, str):
        text_lines = [text]
    else:
        text_lines = text

    svg_lines: list[str] = []
    line_height = 16

    for line_index, line_text in enumerate(text_lines):
        body = escape(line_text)
        line_y = y + line_index * line_height
        svg_lines.append(
            f'<text x="{x}" y="{line_y}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="14" '
            f'fill="{MUTED_COLOR}">{body}</text>'
        )

    return "\n".join(svg_lines)


def pill_svg(x: int, y: int, width: int, text: str) -> str:
    """Build a rounded callout pill."""

    # build
    body = escape(text)
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{width}" height="26" rx="13" '
            f'fill="{SEED_PILL_FILL}" stroke="{SEED_PILL_STROKE}" stroke-width="1.1" />',
            f'<text x="{x + width / 2:.1f}" y="{y + 17}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="12.5" '
            f'font-weight="600" fill="{SEED_PILL_TEXT}">{body}</text>',
        ]
    )


def panel_svg(x: int, y: int, width: int, height: int, title: str | list[str]) -> str:
    """Build a soft panel used to group related boxes."""

    # build
    if isinstance(title, str):
        title_lines = [title]
    else:
        title_lines = title

    svg_lines = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="20" '
        f'fill="#ffffff" stroke="{CARD_STROKE}" stroke-width="1.1" />'
    ]

    if len(title_lines) == 1:
        title_text = escape(title_lines[0])
        svg_lines.append(
            f'<text x="{x + width / 2:.1f}" y="{y + 24}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="14" '
            f'font-weight="600" fill="{MUTED_COLOR}">{title_text}</text>'
        )
    else:
        first_line = escape(title_lines[0])
        second_line = escape(title_lines[1])
        svg_lines.append(
            f'<text x="{x + width / 2:.1f}" y="{y + 22}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="13.5" '
            f'font-weight="600" fill="{MUTED_COLOR}">{first_line}</text>'
        )
        svg_lines.append(
            f'<text x="{x + width / 2:.1f}" y="{y + 38}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="13.5" '
            f'font-weight="600" fill="{MUTED_COLOR}">{second_line}</text>'
        )

    return "\n".join(svg_lines)


# --------------------------------------------------
# main function
# --------------------------------------------------
def main() -> None:
    """Write the workflow SVG to docs/assets."""

    # build step positions
    init_x = LEFT_MARGIN
    lastrac_x = 220
    parsing_x = 470
    spectra_x = 470
    tracking_x = 690
    visualize_x = 988
    processing_x = 988
    clean_x = 988

    # build svg fragments
    fragments: list[str] = []

    # background
    fragments.append(
        f'<rect x="1" y="1" width="{WIDTH - 2}" height="{HEIGHT - 2}" rx="24" '
        f'fill="{BACKGROUND}" stroke="{CARD_STROKE}" stroke-width="1.2" />'
    )

    # build phase panels first so workflow boxes render above them
    fragments.append(
        panel_svg(
            22,
            58,
            384,
            300,
            ["Initialization and", "Meanflow Preparation"],
        )
    )
    fragments.append(panel_svg(442, 58, 470, 300, "Setup Runs"))
    fragments.append(
        panel_svg(
            940,
            58,
            262,
            300,
            "Postprocessing and Cleanup",
        )
    )

    # build boxes
    fragments.append(box_svg(init_x, MID_ROW_Y, "initialization", TOP_FILL, subtitle="create lst.cfg"))
    fragments.append(
        box_svg(
            lastrac_x,
            MID_ROW_Y,
            ["meanflow", "preparation"],
            TOP_FILL,
            subtitle="write meanflow.bin",
        )
    )
    fragments.append(
        box_svg(
            parsing_x,
            TOP_ROW_Y,
            ["setup", "parsing"],
            TOP_FILL,
            subtitle="broad instability sweep",
        )
    )
    fragments.append(
        box_svg(
            spectra_x,
            BOTTOM_ROW_Y,
            ["setup", "spectra"],
            BOTTOM_FILL,
            subtitle="fixed-station branch",
            dashed=True,
        )
    )
    fragments.append(
        box_svg(
            tracking_x,
            TOP_ROW_Y,
            ["setup", "tracking"],
            HIGHLIGHT_FILL,
            subtitle="requires parsing",
        )
    )
    fragments.append(
        box_svg(
            visualize_x,
            TOP_ROW_Y,
            "visualization",
            TOP_FILL,
            subtitle=["from parsing", "or tracking"],
        )
    )
    fragments.append(
        box_svg(
            processing_x,
            MID_ROW_Y,
            "processing",
            TOP_FILL,
            subtitle=["from tracking", "or spectra"],
        )
    )
    fragments.append(
        box_svg(
            clean_x,
            BOTTOM_ROW_Y,
            "cleaning",
            TOP_FILL,
            subtitle=["remove generated", "artifacts"],
        )
    )

    # build labels
    seed_pill_width = 248
    seed_pill_x = int(tracking_x + BOX_WIDTH / 2 - seed_pill_width / 2)
    fragments.append(
        pill_svg(
            seed_pill_x,
            TOP_ROW_Y + BOX_HEIGHT + 14,
            seed_pill_width,
            "writes seed_alpha.dat when enabled",
        )
    )
    # build primary arrows
    init_mid_y = MID_ROW_Y + BOX_HEIGHT / 2
    parsing_mid_y = TOP_ROW_Y + BOX_HEIGHT / 2
    spectra_mid_y = BOTTOM_ROW_Y + BOX_HEIGHT / 2
    fragments.append(arrow_svg(init_x + BOX_WIDTH, int(init_mid_y), lastrac_x, int(init_mid_y)))

    # build lastrac branches
    lastrac_right = lastrac_x + BOX_WIDTH
    lastrac_mid = MID_ROW_Y + BOX_HEIGHT / 2
    fragments.append(
        f'<path d="M {lastrac_right:.1f} {lastrac_mid:.1f} '
        f'L {parsing_x - 24:.1f} {lastrac_mid:.1f} '
        f'L {parsing_x - 24:.1f} {parsing_mid_y:.1f} '
        f'L {parsing_x:.1f} {parsing_mid_y:.1f}" '
        f'fill="none" stroke="{ARROW_COLOR}" stroke-width="2.4" marker-end="url(#arrowhead)" />'
    )
    fragments.append(
        f'<path d="M {lastrac_right:.1f} {lastrac_mid:.1f} '
        f'L {spectra_x - 24:.1f} {lastrac_mid:.1f} '
        f'L {spectra_x - 24:.1f} {spectra_mid_y:.1f} '
        f'L {spectra_x:.1f} {spectra_mid_y:.1f}" '
        f'fill="none" stroke="{ARROW_COLOR}" stroke-width="2.4" stroke-dasharray="7 5" marker-end="url(#arrowhead)" />'
    )

    # build parsing to tracking
    fragments.append(arrow_svg(parsing_x + BOX_WIDTH, int(parsing_mid_y), tracking_x, int(parsing_mid_y)))

    # write
    fragments_block = "\n  ".join(fragments)
    svg_text = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
    <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="6.2" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="{ARROW_COLOR}" />
        </marker>
    </defs>
    {fragments_block}
</svg>
'''

    # write
    OUTPUT_PATH.write_text(svg_text, encoding="utf-8")


if __name__ == "__main__":
    main()