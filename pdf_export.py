"""
pdf_export.py

Converts the AI-generated protocol draft (markdown-ish text with
## headings, **bold**, and PMID links) into a clean, professional PDF
using reportlab. This is the final step of the pipeline:

    ... -> OpenAI GPT -> Final Answer -> Export as PDF

The draft text follows a predictable structure (see prompts.py's
PROTOCOL_TEMPLATE), so this converter handles:
    - "## Heading" lines      -> section headings
    - "**bold text**"         -> bold inline text
    - "[label](url)" links    -> clickable links (e.g. PMID citations)
    - blank lines             -> paragraph breaks
    - "- item" / "* item"     -> bullet list items
"""

import re
import io
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, HRFlowable
)


def _inline_markdown_to_reportlab(text: str) -> str:
    """
    Convert a line of markdown-ish text into reportlab's mini-HTML markup,
    which Paragraph objects understand (<b>, <link>, etc.).
    """
    # Escape raw ampersands first so we don't break our own tags later
    text = text.replace("&", "&amp;")

    # Bold: **text** -> <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    # Markdown links: [label](url) -> clickable blue link
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^\)]+)\)",
        r'<link href="\2" color="#1a73e8">\1</link>',
        text,
    )
    return text


def generate_protocol_pdf(topic: str, draft_markdown: str) -> bytes:
    """
    Build a PDF from the AI-generated draft and return it as raw bytes,
    ready to hand to Streamlit's download button (no temp file needed).
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ProtocolTitle", parent=styles["Title"], fontSize=18, spaceAfter=4
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=12
    )
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e"),
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=8
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=9, textColor=colors.red,
        spaceBefore=14, borderPadding=6,
    )

    story = []

    # --- Header ---
    story.append(Paragraph("Clinical Protocol Draft", title_style))
    story.append(Paragraph(f"Topic: {topic}", meta_style))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')} — "
            "AI-generated draft. Requires clinician review before use.",
            meta_style,
        )
    )
    story.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 10))

    # --- Body: parse the markdown-ish draft line by line ---
    bullet_buffer = []

    def _flush_bullets():
        if bullet_buffer:
            items = [
                ListItem(Paragraph(_inline_markdown_to_reportlab(b), body_style))
                for b in bullet_buffer
            ]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=18))
            bullet_buffer.clear()

    for raw_line in draft_markdown.splitlines():
        line = raw_line.strip()

        if not line:
            _flush_bullets()
            continue

        if line.startswith("## "):
            _flush_bullets()
            story.append(Paragraph(_inline_markdown_to_reportlab(line[3:]), heading_style))
        elif line.startswith("# "):
            _flush_bullets()
            story.append(Paragraph(_inline_markdown_to_reportlab(line[2:]), heading_style))
        elif line.startswith(("- ", "* ")):
            bullet_buffer.append(line[2:])
        else:
            _flush_bullets()
            story.append(Paragraph(_inline_markdown_to_reportlab(line), body_style))

    _flush_bullets()

    # --- Footer disclaimer (always included, regardless of what GPT wrote) ---
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc")))
    story.append(
        Paragraph(
            "⚠ This is an AI-generated draft grounded in retrieved literature. "
            "It is not validated for clinical use. A qualified clinician must "
            "review and approve this protocol before any real-world application.",
            disclaimer_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
