"""
PDF Report Generator for Retinal AI
Generates downloadable clinical reports for ophthalmologists
"""

import os
import base64
from datetime import datetime
from pathlib import Path
import requests


def generate_pdf_report(data: dict) -> str:
    """
    Generate a professional PDF report using reportlab.
    Returns path to the generated PDF.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    uid = data.get("uid", "unknown")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = f"outputs/reports/{uid}_report.pdf"

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#1a3a6b"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#4a5568"),
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1a3a6b"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=4,
    )
    warning_style = ParagraphStyle(
        "Warning",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#c05621"),
        alignment=TA_CENTER,
        spaceAfter=4,
    )

    # Header
    story.append(Paragraph("🔬 Retinal AI Analysis Report", title_style))
    story.append(Paragraph("Clinical Decision Support System for Ophthalmology", subtitle_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a3a6b")))
    story.append(Spacer(1, 0.3 * cm))

    # Meta
    meta_data = [
        ["Report ID:", uid],
        ["Analysis Date:", timestamp],
        ["Inference Time:", f"{data.get('inference_time', '-')} seconds"],
    ]
    meta_table = Table(meta_data, colWidths=[4 * cm, 10 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#4a5568")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5 * cm))

    # Images side-by-side
    image_url = data.get("image_url", "")
    heatmap_url = data.get("heatmap_url", "")

    img_paths = []
    for url in [image_url, heatmap_url]:
        if url.startswith("/outputs/"):
            local = url.lstrip("/")
            if os.path.exists(local):
                img_paths.append(local)

    if img_paths:
        story.append(Paragraph("Fundus Image & Grad-CAM Visualization", section_style))
        img_row = []
        for path in img_paths[:2]:
            try:
                img = RLImage(path, width=8 * cm, height=6 * cm)
                img_row.append(img)
            except Exception:
                img_row.append(Paragraph("(Image unavailable)", body_style))
        while len(img_row) < 2:
            img_row.append(Paragraph("", body_style))
        img_table = Table([img_row], colWidths=[8.5 * cm, 8.5 * cm])
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(img_table)

        caption_table = Table(
            [["Original Fundus Image", "Grad-CAM Heatmap"]],
            colWidths=[8.5 * cm, 8.5 * cm]
        )
        caption_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#718096")),
        ]))
        story.append(caption_table)
        story.append(Spacer(1, 0.5 * cm))

    # Results section
    story.append(Paragraph("AI Analysis Results", section_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 0.3 * cm))

    disease_configs = [
        ("Diabetic Retinopathy", "dr", "#e53e3e", "#fc8181"),
        ("Glaucoma", "glaucoma", "#d69e2e", "#f6ad55"),
        ("Age-Related Macular Degeneration", "amd", "#805ad5", "#b794f4"),
    ]

    for name, key, color_present, color_light in disease_configs:
        result = data.get(key, {})
        present = result.get("present", False)
        severity = result.get("severity", "Unknown")
        confidence = result.get("confidence", 0)
        all_scores = result.get("all_scores", {})

        status_color = colors.HexColor(color_present) if present else colors.HexColor("#38a169")
        status_text = "DETECTED" if present else "NOT DETECTED"

        header_data = [[name, f"{status_text}  |  {severity}  |  Confidence: {confidence:.1%}"]]
        header_table = Table(header_data, colWidths=[6 * cm, 11 * cm])
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7fafc")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (1, 0), (1, 0), status_color),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(header_table)

        # Score breakdown
        if all_scores:
            score_rows = []
            for label, score in all_scores.items():
                bar_width = int(score * 100)
                score_rows.append([
                    label,
                    f"{score:.1%}",
                ])
            score_table = Table(score_rows, colWidths=[10 * cm, 7 * cm])
            score_table.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TEXTCOLOR", (1, 0), (-1, -1), colors.HexColor("#4a5568")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(score_table)

        story.append(Spacer(1, 0.4 * cm))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "⚠️  IMPORTANT DISCLAIMER",
        ParagraphStyle("Warn", parent=styles["Heading3"], textColor=colors.HexColor("#c05621"), fontSize=11)
    ))
    story.append(Paragraph(
        "This AI tool is a clinical decision support system and NOT a replacement for professional diagnosis. "
        "All findings should be reviewed and confirmed by a qualified ophthalmologist before clinical decisions are made. "
        "This report is generated by an automated AI system and has not been reviewed by a medical professional.",
        warning_style,
    ))

    doc.build(story)
    return output_path
