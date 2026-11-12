"""Professional Clinical Histopathology PDF Report Generator.

Produces publication-grade A4 diagnostic PDF reports combining specimen metadata,
three-panel Grad-CAM visual evidence, probability bar charts, AJCC staging badges,
and clinical management recommendations using Matplotlib's vector PDF backend.
"""

import io
import textwrap
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

__REPORT_VERSION__ = 3


def _build_clinical_report_figure(
    diagnosis: Dict[str, Any],
    orig_img: Image.Image,
    heat_img: Image.Image,
    foci_img: Image.Image,
    report_id: Optional[str] = None,
    analyzed_at: Optional[str] = None,
    alpha: float = 0.45,
    salient_threshold: float = 0.60,
):
    """Builds the Matplotlib A4 figure for the clinical diagnostic report."""
    staging = diagnosis["staging"]
    stage_color = staging.get("color", "#2563eb")
    pred_class = str(diagnosis["predicted_class"]).upper()
    conf = float(diagnosis["confidence_percent"])
    pri_prob = float(diagnosis["primary_probability"])
    meta_prob = float(diagnosis["metastatic_probability"])
    specimen_id = str(diagnosis.get("specimen_id", "Unknown Specimen"))
    tta_label = "4-View Test-Time Augmentation (TTA)" if diagnosis.get("tta_enhanced", True) else "Single-View Evaluation"

    if not report_id:
        report_id = f"MEL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not analyzed_at:
        analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # A4 portrait dimensions in inches
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="#ffffff")

    # Background canvas axes (normalized 0..1 coordinates)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ------------------------------------------------------------------
    # 1. HEADER BANNER
    # ------------------------------------------------------------------
    ax.add_patch(Rectangle((0, 0.885), 1, 0.115, facecolor="#0f172a", edgecolor="none"))
    ax.add_patch(Rectangle((0, 0.879), 1, 0.006, facecolor="#2563eb", edgecolor="none"))

    ax.text(
        0.04, 0.955,
        "CLINICAL HISTOPATHOLOGY DIAGNOSTIC REPORT",
        color="#ffffff", fontsize=12.5, fontweight="bold", va="center", family="sans-serif"
    )
    ax.text(
        0.04, 0.922,
        "Cutaneous Melanoma Subtyping (Primary vs. Metastatic)  •  Explainable AI",
        color="#93c5fd", fontsize=8.2, va="center", family="sans-serif"
    )
    ax.text(
        0.04, 0.898,
        "CONFIDENTIAL MEDICAL DECISION SUPPORT RECORD",
        color="#64748b", fontsize=7.2, fontweight="bold", va="center", family="sans-serif"
    )

    # Report ID Pill on top right (sized and positioned to prevent any title overlap)
    ax.add_patch(FancyBboxPatch(
        (0.715, 0.926), 0.245, 0.040,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor="#1e293b", edgecolor="#3b82f6", linewidth=1.2
    ))
    ax.text(
        0.8375, 0.946,
        f"ID: {report_id}",
        color="#e2e8f0", fontsize=7.8, fontweight="bold", ha="center", va="center", family="monospace"
    )

    # ------------------------------------------------------------------
    # 2. SPECIMEN & ASSESSMENT METADATA CARD
    # ------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch(
        (0.05, 0.775), 0.90, 0.085,
        boxstyle="round,pad=0.004,rounding_size=0.01",
        facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.0
    ))

    # Truncate very long filenames gracefully
    disp_specimen = specimen_id if len(specimen_id) <= 36 else specimen_id[:33] + "..."

    # Left column metadata
    ax.text(0.07, 0.840, "SPECIMEN IDENTIFIER", color="#64748b", fontsize=7.5, fontweight="bold")
    ax.text(0.07, 0.820, disp_specimen, color="#0f172a", fontsize=9.5, fontweight="bold")

    ax.text(0.07, 0.796, "EXAMINATION TIMESTAMP", color="#64748b", fontsize=7.5, fontweight="bold")
    ax.text(0.07, 0.781, f"{analyzed_at}  |  α={alpha:.2f}, Foci >{int(salient_threshold*100)}%",
            color="#334155", fontsize=8.0)

    # Vertical divider
    ax.plot([0.495, 0.495], [0.782, 0.852], color="#cbd5e1", linewidth=0.9)

    # Right column metadata
    ax.text(0.515, 0.840, "PREDICTED SUBTYPE", color="#64748b", fontsize=7.5, fontweight="bold")
    ax.text(0.515, 0.820, f"{pred_class} MELANOMA", color=stage_color, fontsize=10.5, fontweight="bold")

    ax.text(0.515, 0.796, "DIAGNOSTIC CONFIDENCE & STAGE", color="#64748b", fontsize=7.5, fontweight="bold")
    conf_mode = "4-View TTA" if diagnosis.get("tta_enhanced", True) else "Single View"
    ax.text(0.515, 0.781, f"{conf:.2f}% ({conf_mode})  |  {staging['stage']} — {staging['risk']}",
            color="#0f172a", fontsize=7.9, fontweight="bold")

    # ------------------------------------------------------------------
    # 3. THREE-PANEL HISTOPATHOLOGY & EXPLAINABILITY VISUALS
    # ------------------------------------------------------------------
    ax.text(0.05, 0.742, "I. HISTOPATHOLOGICAL ROI & EXPLAINABILITY MAPS",
            color="#0f172a", fontsize=10.5, fontweight="bold")
    ax.plot([0.05, 0.95], [0.734, 0.734], color="#2563eb", linewidth=1.2)

    panel_specs = [
        ("1. Standardized ROI Patch", orig_img, "Input Tissue Morphology (224×224)", 0.055),
        ("2. Grad-CAM Heatmap", heat_img, f"Class-Discriminative Regions (α={alpha:.2f})", 0.365),
        ("3. Salient Nuclear Foci", foci_img, f"High-Attention Nuclei (>{int(salient_threshold*100)}%)", 0.675),
    ]

    for title, pil_im, caption, x_pos in panel_specs:
        # Card frame
        ax.add_patch(FancyBboxPatch(
            (x_pos, 0.488), 0.27, 0.232,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=0.9
        ))
        ax.text(x_pos + 0.135, 0.703, title, color="#1e293b", fontsize=8.5,
                fontweight="bold", ha="center", va="center")

        # Image inset axes
        img_ax = fig.add_axes([x_pos + 0.012, 0.515, 0.246, 0.175])
        img_ax.imshow(pil_im)
        img_ax.set_xticks([])
        img_ax.set_yticks([])
        for spine in img_ax.spines.values():
            spine.set_edgecolor("#e2e8f0")
            spine.set_linewidth(0.8)

        ax.text(x_pos + 0.135, 0.499, caption, color="#64748b", fontsize=7.5,
                ha="center", va="center")

    # ------------------------------------------------------------------
    # 4. QUANTITATIVE PROBABILITIES & STAGING STRATIFICATION
    # ------------------------------------------------------------------
    ax.text(0.05, 0.454, "II. QUANTITATIVE SUBTYPE PROBABILITIES & RISK STRATIFICATION",
            color="#0f172a", fontsize=10.5, fontweight="bold")
    ax.plot([0.05, 0.95], [0.446, 0.446], color="#2563eb", linewidth=1.2)

    # Left card: Probability bar chart
    ax.add_patch(FancyBboxPatch(
        (0.05, 0.265), 0.44, 0.165,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.9
    ))
    ax.text(0.07, 0.412, "Class Probability Distribution", color="#1e293b", fontsize=9, fontweight="bold")

    bar_ax = fig.add_axes([0.155, 0.285, 0.305, 0.11])
    bar_ax.set_facecolor("#f8fafc")
    labels = ["Metastatic", "Primary"]
    values = [meta_prob, pri_prob]
    colors = ["#dc2626", "#2563eb"]
    bars = bar_ax.barh(labels, values, color=colors, height=0.52, edgecolor="none")
    bar_ax.set_xlim(0, 115)
    bar_ax.set_xlabel("Probability (%)", fontsize=7.5, color="#475569")
    bar_ax.tick_params(axis="both", labelsize=8, colors="#1e293b")
    for spine in ["top", "right"]:
        bar_ax.spines[spine].set_visible(False)
    bar_ax.spines["left"].set_color("#94a3b8")
    bar_ax.spines["bottom"].set_color("#94a3b8")
    for bar, val in zip(bars, values):
        bar_ax.text(val + 2.0, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8, fontweight="bold", color="#0f172a")

    # Right card: AJCC Staging Box
    ax.add_patch(FancyBboxPatch(
        (0.515, 0.265), 0.435, 0.165,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor="#ffffff", edgecolor=stage_color, linewidth=1.4
    ))
    # Colored top badge strip
    ax.add_patch(FancyBboxPatch(
        (0.528, 0.382), 0.409, 0.036,
        boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor=stage_color, edgecolor="none"
    ))
    ax.text(0.7325, 0.400, f"{staging['stage'].upper()}  —  {staging['risk'].upper()}",
            color="#ffffff", fontsize=9.5, fontweight="bold", ha="center", va="center")

    ax.text(0.535, 0.354, "HISTOLOGIC TIER", color="#64748b", fontsize=7.5, fontweight="bold")
    tier_wrapped = "\n".join(textwrap.wrap(staging["tier"], width=36))
    ax.text(0.535, 0.336, tier_wrapped, color="#0f172a", fontsize=9.5, fontweight="bold", va="top")

    ax.text(0.535, 0.292, "EVALUATION PROTOCOL", color="#64748b", fontsize=7.5, fontweight="bold")
    ax.text(0.535, 0.276, tta_label, color="#334155", fontsize=8.5)

    # ------------------------------------------------------------------
    # 5. RECOMMENDED CLINICAL MANAGEMENT CALLOUT
    # ------------------------------------------------------------------
    ax.text(0.05, 0.232, "III. RECOMMENDED ONCOLOGICAL MANAGEMENT",
            color="#0f172a", fontsize=10.5, fontweight="bold")
    ax.plot([0.05, 0.95], [0.224, 0.224], color="#2563eb", linewidth=1.2)

    ax.add_patch(FancyBboxPatch(
        (0.05, 0.125), 0.90, 0.085,
        boxstyle="round,pad=0.003,rounding_size=0.008",
        facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.9
    ))
    ax.add_patch(Rectangle((0.05, 0.125), 0.010, 0.085, facecolor=stage_color, edgecolor="none"))

    action_wrapped = "\n".join(textwrap.wrap(staging["action"], width=82))
    ax.text(0.075, 0.194, f"Clinical Action Directive ({staging['risk']}):",
            color=stage_color, fontsize=9, fontweight="bold", va="top")
    ax.text(0.075, 0.172, action_wrapped,
            color="#1e293b", fontsize=9.5, va="top", linespacing=1.4)

    # ------------------------------------------------------------------
    # 6. SIGN-OFF & CLINICAL DISCLAIMER FOOTER
    # ------------------------------------------------------------------
    ax.plot([0.05, 0.95], [0.098, 0.098], color="#cbd5e1", linewidth=0.8)

    ax.text(0.05, 0.070, "Attending Pathologist Signature: _______________________________",
            color="#334155", fontsize=8.5)
    ax.text(0.65, 0.070, "Review Date: ___________________",
            color="#334155", fontsize=8.5)

    disclaimer = (
        "CLINICAL DISCLAIMER: This report is generated by an AI-assisted histopathological decision support system "
        "designed to augment dermatopathologist review. Quantitative staging probabilities and Grad-CAM heatmaps must "
        "be interpreted alongside clinical history, immunohistochemical staining, and board-certified pathologist confirmation."
    )
    disclaimer_wrapped = "\n".join(textwrap.wrap(disclaimer, width=112))
    ax.text(0.05, 0.044, disclaimer_wrapped, color="#64748b", fontsize=7.0, va="top", linespacing=1.35)

    return fig


def generate_clinical_report_with_preview(
    diagnosis: Dict[str, Any],
    orig_img: Image.Image,
    heat_img: Image.Image,
    foci_img: Image.Image,
    report_id: Optional[str] = None,
    analyzed_at: Optional[str] = None,
    alpha: float = 0.45,
    salient_threshold: float = 0.60,
) -> Tuple[bytes, bytes]:
    """Generates both the A4 PDF report bytes and a PNG preview image."""
    fig = _build_clinical_report_figure(
        diagnosis=diagnosis,
        orig_img=orig_img,
        heat_img=heat_img,
        foci_img=foci_img,
        report_id=report_id,
        analyzed_at=analyzed_at,
        alpha=alpha,
        salient_threshold=salient_threshold,
    )
    pdf_buf = io.BytesIO()
    png_buf = io.BytesIO()
    try:
        fig.savefig(pdf_buf, format="pdf", dpi=300)
        fig.savefig(png_buf, format="png", dpi=150)
    finally:
        plt.close(fig)
    pdf_buf.seek(0)
    png_buf.seek(0)
    return pdf_buf.getvalue(), png_buf.getvalue()


def generate_clinical_pdf_report(
    diagnosis: Dict[str, Any],
    orig_img: Image.Image,
    heat_img: Image.Image,
    foci_img: Image.Image,
    report_id: Optional[str] = None,
    analyzed_at: Optional[str] = None,
    alpha: float = 0.45,
    salient_threshold: float = 0.60,
) -> bytes:
    """Renders a structured A4 clinical diagnostic report and returns PDF bytes."""
    pdf_bytes, _ = generate_clinical_report_with_preview(
        diagnosis=diagnosis,
        orig_img=orig_img,
        heat_img=heat_img,
        foci_img=foci_img,
        report_id=report_id,
        analyzed_at=analyzed_at,
        alpha=alpha,
        salient_threshold=salient_threshold,
    )
    return pdf_bytes

