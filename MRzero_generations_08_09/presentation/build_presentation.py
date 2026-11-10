"""Revise the user's selected 15-slide deck from reproducible numeric artifacts.

Run with the system Python plus python-pptx/Pillow/lxml installed, or use
PYTHONPATH=/tmp/presentation_deps python presentation/build_presentation.py.
LaTeX, pdftocairo and pdftoppm generate vector equations and compatibility PNGs.
The input presentation is never modified. This script does not run simulations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys

try:
    import pptx
except ImportError:
    sys.path.insert(0, "/tmp/presentation_deps")
from PIL import Image
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.oxml import parse_xml
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SOURCE = PACKAGE / "2026_11_11_ACDC_MRzero_Generations08_09_WeeklyResearchUpdate.pptx"
OUTPUT = PACKAGE / "2026_11_11_ACDC_MRzero_WeeklyUpdate_REVISED.pptx"
EQUATIONS = HERE / "equations"
FIGURES = HERE / "figures"
COLOR = {"ink": "20262C", "blue": "234C78", "teal": "007F83", "muted": "5A626A", "line": "99A9B8", "red": "963C35"}
FONT = "Calibri"

LATEX = {
    "tn_k": r"t_n=t_{\mathrm{ADC},n}-t_{\mathrm{RF},n},\qquad \mathbf{k}_n=(k_x(t_n),k_y(t_n))",
    "J": r"J_{j,n}=\int_{t_{\mathrm{RF}}}^{t_{\mathrm{ADC},n}}I_j^{\mathrm{actual}}(\tau)\,d\tau",
    "q": r"q_{a,v,n}=\mathbf{k}_n\!\cdot\!\mathbf{e}_{a,v}+g_{B0,a,v}\,t_n+\sum_{j=1}^{32}g_{\mathrm{ACDC},a,v,j}\,J_{j,n}",
    "F": r"F_{v,n}=\prod_{a=1}^{3}\operatorname{sinc}(q_{a,v,n})",
    "center": r"\widetilde m_{v,n}^{\mathrm{fixed}}",
    "s_sum": r"s_n(I)=\sum_v m^{\mathrm{point}}_{v,n}(I)\,F_{v,n}(I)",
    "rho": r"\widehat\rho(\mathbf r_m)=\frac{1}{N_s}\sum_n s_n\exp(+i2\pi\,\mathbf k_n\!\cdot\!\mathbf r_m)",
    "point": r"m^{\mathrm{point}}_{v,n}(I)=\widetilde m^{\mathrm{fixed}}_{v,n}\exp\!\left[-i2\pi\sum_{j=1}^{32}b_{v,j}J_{j,n}\right]",
    "PD": r"\mathrm{PD}^{\mathrm{eff}}_v=A_v(t_{\mathrm{ref}})\,\exp(R_{2,v}\,t_{\mathrm{ref}})",
    "unresolved": r"\widetilde m^{\mathrm{fixed}}_{v,n}=m^{\mathrm{MRzero}}_{v,n}\,W^{\mathrm{unresolved}}_{v,n}",
    "phi": r"\phi_v(\mathbf u,t_n)=\phi_{\mathrm{center},v}(t_n)+2\pi\sum_{a=1}^{3}q_{a,v,n}\,u_a",
    "u": r"u_a\in[-\tfrac12,\tfrac12],\qquad a\in\{1,2,3\}",
    "integral": r"F_{v,n}=\int_{[-1/2,1/2]^3}\!\exp\!\left[-i2\pi\sum_a q_{a,v,n}u_a\right]d^3u=\prod_{a=1}^{3}\operatorname{sinc}(q_{a,v,n})",
    "sinc": r"\operatorname{sinc}(q)=\frac{\sin(\pi q)}{\pi q},\qquad \operatorname{sinc}(0)=1",
    "gain": r"g^\star=\underset{g>0}{\operatorname{argmin}}\ \sum_{m\in M}\big(g|S_m|-E_m\big)^2",
    "nrmse": r"\mathrm{NRMSE}_{\mathrm{OFF}}=100\,\frac{\|g^\star|S|-E\|_{2,M}}{\|E\|_{2,M}}\ \%",
    "svd": r"\widehat\rho_{\mathrm{V2}}=A^+s\quad(\lambda=0)",
    "epsilon": r"\epsilon_s=\frac{\|s_{09}-s_{08}\|_2}{\|s_{08}\|_2}",
    "B0effective": r"B^{\mathrm{eff}}_{0,v,n}=B_{0,v}+\frac{1}{t_n}\sum_{j=1}^{32}b_{v,j}\,J_{j,n}",
    "sample_center": r"s_n\approx S(t_{\mathrm{ADC},n})",
    "sample_window": r"\overline s_n=\frac{1}{\Delta t}\int_{t_n-\Delta t/2}^{t_n+\Delta t/2}S(t)\,dt",
    "rss": r"I_{\mathrm{RSS}}(\mathbf r)=\sqrt{\sum_c\left|\mathcal R\{s_{c,\cdot}\}(\mathbf r)\right|^2}",
    "receiver_forward": r"s_{c,n}=\sum_v C_c(\mathbf r_v)\,m^{\mathrm{point}}_{v,n}\,F_{v,n}",
    "coordinates": r"\mathbf r_{\mathrm{coil}}=Q\mathbf r_{\mathrm{data}},\qquad\mathbf r_{\mathrm{Fourier}}=PQ\mathbf r_{\mathrm{data}}",
    "sign": r"\mathrm{forward}:\ e^{-i2\pi\mathbf k\cdot\mathbf r}\qquad\mathrm{adjoint}:\ e^{+i2\pi\mathbf k\cdot\mathbf r}",
    "loss": r"J_{\mathrm{image}}(I)=\frac{\sum_{m\in M}(g|\mathrm{EPI}_{\mathrm{sim},m}(I)|-\mathrm{GRE}_m)^2}{\sum_{m\in M}\mathrm{GRE}_m^2}",
    "fixed_assumptions": r"\alpha=90^{\circ},\qquad T_1=\infty,\qquad B_1^+=1,\qquad D=0",
}
# Side limits keep long linked expressions legible at presentation size.
LATEX = {name: formula.replace(r"\sum_", r"\sum\nolimits_").replace(
    r"\prod_", r"\prod\nolimits_") for name, formula in LATEX.items()}


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def scientific_text(value):
    """Editable presentation notation, without software-style e exponents."""
    mantissa, exponent = f"{value:.2e}".split("e")
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return mantissa + " × 10" + str(int(exponent)).translate(superscripts)


def generate_equations():
    """Generate genuine pdflatex vector formulas, with literal sources retained."""
    EQUATIONS.mkdir(exist_ok=True)
    manifest = {}
    for name, latex in LATEX.items():
        tex = (r"\documentclass[12pt]{article}" "\n"
               r"\usepackage{amsmath,amssymb}" "\n"
               r"\pagestyle{empty}\newsavebox{\equationbox}" "\n"
               r"\begin{document}" "\n"
               r"\sbox{\equationbox}{$\displaystyle " + latex + r"$}" "\n"
               r"\pdfpagewidth=\dimexpr\wd\equationbox+6pt\relax" "\n"
               r"\pdfpageheight=\dimexpr\ht\equationbox+\dp\equationbox+6pt\relax" "\n"
               r"\hoffset=-1in\voffset=-1in" "\n"
               r"\shipout\vbox{\offinterlineskip\kern3pt\hbox{\kern3pt\usebox{\equationbox}\kern3pt}\kern3pt}" "\n"
               r"\end{document}" "\n")
        path = EQUATIONS / f"{name}.tex"
        changed = not path.exists() or path.read_text() != tex
        path.write_text(tex)
        pdf, svg, png = [path.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
        if changed or not all(p.exists() for p in (pdf, svg, png)):
            result = subprocess.run(["pdflatex", "-interaction=batchmode", "-halt-on-error", "-output-directory", str(EQUATIONS), str(path)], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"LaTeX failed for {name}: {(EQUATIONS / (name + '.log')).read_text()[-2000:]}")
            subprocess.run(["pdftocairo", "-svg", str(pdf), str(svg)], check=True)
            subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "600", str(pdf), str(png.with_suffix(''))], check=True)
        # PDFs are tightly cropped; their SVG width/height preserve point size.
        vector = etree.fromstring(svg.read_bytes())
        width = float(re.sub(r"[^0-9.]", "", vector.get("width")))
        height = float(re.sub(r"[^0-9.]", "", vector.get("height")))
        manifest[name] = dict(latex=latex, tex=str(path.relative_to(HERE)),
            svg=str(svg.relative_to(HERE)), pdf=str(pdf.relative_to(HERE)),
            png_preview=str(png.relative_to(HERE)), width_pt=width, height_pt=height,
            sha256=file_hash(svg), engine="pdflatex; embedded SVG vector with 600-dpi compatibility PNG")
    (HERE / "equations_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class Deck:
    def __init__(self, slide_data, draft=False):
        self.data = slide_data
        self.draft = draft
        self.prs = Presentation(str(SOURCE))
        if len(self.prs.slides) != 15:
            raise ValueError("The manually selected source deck must contain exactly 15 slides.")
        self.equations = generate_equations()
        self.used = {index: {"equations": [], "figures": []} for index in range(1, 16)}
        self.change_log = []
        self._svg_count = 0

    def fact(self, dotted, formatter=None, default="pending"):
        value = self.data
        try:
            for key in dotted.split("."):
                value = value[key]
        except (KeyError, TypeError):
            if not self.draft:
                raise KeyError(f"Missing generated presentation fact: {dotted}")
            return default
        if formatter:
            return formatter(value)
        return str(value)

    def add_text(self, slide, text, x, y, w, h, *, size=16, bold=False,
                 color="ink", align=PP_ALIGN.LEFT, margin=0.015):
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = shape.text_frame
        tf.clear()
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(margin)
        tf.margin_top = tf.margin_bottom = Inches(margin)
        for index, line in enumerate(str(text).split("\n")):
            p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
            p.text = line
            p.font.name = FONT
            p.font.size = Pt(size)
            p.font.bold = bold
            p.font.color.rgb = RGBColor.from_string(COLOR.get(color, color))
            p.alignment = align
            p.space_after = Pt(5)
        return shape

    def line(self, slide, x1, y1, x2, y2, *, color="line", width=1, arrow=False):
        shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        shape.line.color.rgb = RGBColor.from_string(COLOR.get(color, color))
        shape.line.width = Pt(width)
        if arrow:
            tail = OxmlElement("a:tailEnd")
            tail.set("type", "triangle")
            shape.line._get_or_add_ln().append(tail)
        return shape

    def box(self, slide, x, y, w, h, title, body="", *, title_size=17, body_size=14, color="blue"):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.background()
        shape.line.color.rgb = RGBColor.from_string(COLOR[color])
        shape.line.width = Pt(.8)
        self.add_text(slide, title, x+.10, y+.10, w-.20, .46, size=title_size, bold=True, color=color,
                      align=PP_ALIGN.CENTER)
        if body:
            self.add_text(slide, body, x+.12, y+.65, w-.24, h-.70, size=body_size, align=PP_ALIGN.CENTER)
        return shape

    def equation(self, slide, name, x, y, w, h, *, size=22, center=False):
        record = self.equations[name]
        natural_w = record["width_pt"] / 72 * size / 12
        natural_h = record["height_pt"] / 72 * size / 12
        scale = min(1., w / natural_w, h / natural_h)
        width, height = natural_w*scale, natural_h*scale
        if center:
            x += (w-width)/2
        y += (h-height)/2
        picture = slide.shapes.add_picture(str(HERE / record["png_preview"]), Inches(x), Inches(y),
                                           width=Inches(width), height=Inches(height))
        # PowerPoint uses the SVG. Older renderers use a crisp, typeset preview.
        self._svg_count += 1
        part = Part(PackURI(f"/ppt/media/acdc_equation_{self._svg_count:03d}.svg"),
                    "image/svg+xml", slide.part.package, (HERE / record["svg"]).read_bytes())
        rid = slide.part.relate_to(part, RT.IMAGE)
        blip = picture._pic.blipFill.blip
        ext_lst = OxmlElement("a:extLst")
        ext = OxmlElement("a:ext")
        ext.set("uri", "{96DAC541-7B7A-43D3-8B79-37D633B846F1}")
        svg_blip = etree.Element("{http://schemas.microsoft.com/office/drawing/2016/SVG/main}svgBlip",
            nsmap={"asvg": "http://schemas.microsoft.com/office/drawing/2016/SVG/main"})
        svg_blip.set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", rid)
        ext.append(svg_blip); ext_lst.append(ext); blip.append(ext_lst)
        picture._pic.nvPicPr.cNvPr.set("descr", "Typeset vector LaTeX: " + record["latex"])
        index = list(self.prs.slides).index(slide)+1
        self.used[index]["equations"].append(name)
        return picture

    def figure(self, slide, alias, x, y, w, h):
        entry = self.data.get("figures", {}).get(alias)
        path = None
        if entry:
            path = Path(entry["path"] if isinstance(entry, dict) else entry)
            if not path.is_absolute():
                path = HERE / path
        if path is None or not path.is_file():
            if not self.draft:
                raise FileNotFoundError(f"Generated figure alias {alias!r} is missing: {path}")
            self.box(slide, x, y, w, h, "Awaiting generated figure", alias, title_size=16)
            return
        with Image.open(path) as img:
            ratio = img.width/img.height
        height = min(h, w/ratio); width = height*ratio
        pic = slide.shapes.add_picture(str(path), Inches(x+(w-width)/2), Inches(y+(h-height)/2),
                                      width=Inches(width), height=Inches(height))
        pic._pic.nvPicPr.cNvPr.set("descr", f"Data figure {alias}; source {path}; see figures_manifest.json")
        self.used[list(self.prs.slides).index(slide)+1]["figures"].append(dict(alias=alias, path=str(path.resolve()), sha256=file_hash(path)))

    def slide(self, number, title, category, footer):
        slide = self.prs.slides[number-1]
        # Retain original selected slide, theme, layout, title/footer and divider
        # objects. Revise its content area; never create a replacement blank deck.
        for shape in list(slide.shapes):
            if shape.shape_id not in (2, 3, 4, 5, 6, 7):
                element = shape._element
                element.getparent().remove(element)
        for shape in slide.shapes:
            if shape.shape_id == 2:
                shape.text = title
                shape.left, shape.top, shape.width, shape.height = map(Inches, (.58,.20,12.12,.95))
                p = shape.text_frame.paragraphs[0]
                p.font.name = FONT; p.font.size = Pt(25); p.font.bold = True
                p.font.color.rgb = RGBColor.from_string(COLOR["ink"])
                shape.text_frame.margin_left = shape.text_frame.margin_right = 0
                shape.text_frame.margin_top = shape.text_frame.margin_bottom = 0
            elif shape.shape_id in (5, 6, 7):
                shape.text = category if shape.shape_id == 5 else footer if shape.shape_id == 6 else str(number)
                p = shape.text_frame.paragraphs[0]
                p.font.name = FONT; p.font.size = Pt(8 if shape.shape_id == 6 else 9)
                p.font.bold = shape.shape_id == 5
                p.font.color.rgb = RGBColor.from_string(COLOR["muted"])
        return slide

    def par(self, slide, problem, approach, result, y=1.20, size=13):
        # A consistent scientific reading order, without decorative cards.
        for x, label, text in ((.62,"Problem",problem),(4.80,"Approach",approach),(8.96,"Result",result)):
            self.add_text(slide, label, x,y,3.70,.25,size=size,bold=True,color="blue")
            self.add_text(slide, text,x,y+.29,3.70,.72,size=size)

    def cube(self, slide, x, y, w=2.2, h=1.7, *, color="blue", axes=False):
        d=.36
        vertices=[(x,y+d),(x+w-d,y+d),(x+w-d,y+h),(x,y+h),
                  (x+d,y),(x+w,y),(x+w,y+h-d),(x+d,y+h-d)]
        for a,b in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)):
            self.line(slide,*vertices[a],*vertices[b],color=color,width=1.2)
        if axes:
            cx,cy=x+w/2,y+h/2
            for ex,ey,label in ((cx+.8,cy,"u₁"),(cx,cy-.7,"u₂"),(cx-.5,cy+.5,"u₃")):
                self.line(slide,cx,cy,ex,ey,color="teal",width=1.2,arrow=True)
                self.add_text(slide,label,ex-.08,ey-.20,.4,.3,size=15,color="teal")

    def notes(self, number, change, narrative, minutes, *, extra=""):
        self.change_log.append(dict(slide=number, change=change, minutes=minutes))
        common = self.data.get("provenance", {})
        used = self.used[number]
        equations = "\n\n".join(name + ":\n" + self.equations[name]["latex"] for name in used["equations"])
        text = (f"Slide {number} of 15; {'main update' if number <= 11 else 'backup, optional'}; planned speaking time {minutes} min.\n"
            f"Change from user's manually selected source slide {number}: {change}\n\n"
            f"Problem → Approach → Result\n{narrative}\n\n{extra}\n\n"
            f"Presentation source: {SOURCE}\nSource SHA256: {file_hash(SOURCE)}\n"
            f"Reproducible notebook: {HERE / '00_generate_weekly_figures.ipynb'}\n"
            f"Figure manifest: {HERE / 'figures_manifest.json'}\n"
            f"Generated facts: {HERE / 'presentation_slide_data.json'}\n"
            f"Provenance / exact data and implementation paths:\n{json.dumps(common,indent=2)}\n\n"
            f"Data figures on this slide:\n{json.dumps(used['figures'],indent=2)}\n\n"
            f"Literal LaTeX equation sources (editable-source .tex files in presentation/equations):\n{equations}\n"
            "Equation rendering: genuine pdflatex vector SVG, with a 600-dpi compatibility preview. Native editable Office-math conversion was tested but rendered incorrectly in the audit tool, so the permitted vector fallback is used. Text, diagram labels, scientific shapes and arrows remain native editable PowerPoint objects.")
        notes = self.prs.slides[number-1].notes_slide
        if notes.notes_text_frame is None:
            placeholder = parse_xml(b'''<p:sp xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:nvSpPr><p:cNvPr id="100" name="Scientific speaker notes"/><p:cNvSpPr/><p:nvPr><p:ph type="body" idx="13"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody></p:sp>''')
            notes._element.spTree.append(placeholder)
        notes.notes_text_frame.text = text

    def build(self):
        self.slide01(); self.slide02(); self.slide03(); self.slide04(); self.slide05()
        self.slide06(); self.slide07(); self.slide08(); self.slide09(); self.slide10()
        self.slide11(); self.slide12(); self.slide13(); self.slide14(); self.slide15()
        self.prs.save(str(OUTPUT))
        (HERE / "slide_changes.json").write_text(json.dumps(self.change_log,indent=2))
        (HERE / "presentation_build_manifest.json").write_text(json.dumps(dict(
            source=str(SOURCE),source_sha256=file_hash(SOURCE),output=str(OUTPUT),
            output_sha256=file_hash(OUTPUT),slide_count=15,main_slides=11,backup_slides=4,
            timestamp_UTC=datetime.now(timezone.utc).isoformat(),draft=self.draft,
            build_source_sha256={str((HERE/name).resolve()):file_hash(HERE/name) for name in
                ('build_presentation.py','export_slide_data.py','render_presentation.py')},
            slide_data_sha256=file_hash(HERE/'presentation_slide_data.json') if (HERE/'presentation_slide_data.json').exists() else None,
            equations_manifest=str(HERE/'equations_manifest.json'),slides=self.used),indent=2))
        print(OUTPUT)

    def slide01(self):
        s=self.slide(1,"Can the EPI twin predict measured OFF well enough to guide ACDC currents?","INTRODUCTION","30-minute research update · 11 Nov 2026 · Generations 08 / 08 V2 / 09")
        self.add_text(s,"Research question",.74,1.48,11.8,.4,size=17,bold=True,color="blue")
        self.add_text(s,"Validate experimental OFF-model fidelity before interpreting simulated current optimization.",.74,2.03,11.8,1.02,size=26,bold=True)
        for i,(title,body) in enumerate((("Measured data","GRE / EPI OFF / MEGRE / shim maps"),("Physics","MRzero + one finite 3D voxel"),("Prediction","Complex ADC samples → EPI"),("Optimization","32 ACDC current waveforms"))):
            x=.68+3.10*i
            self.box(s,x,3.66,2.66,1.28,title,body,title_size=17,body_size=14)
            if i<3:self.line(s,x+2.70,4.30,x+3.00,4.30,arrow=True,color="blue")
        self.par(s,"Structured simulated intensity does not identify its physical cause.","Separate reconstruction, implementation and scanner comparisons.","Report rerun fidelity, parity and compute cost independently.",y=5.52,size=14)
        self.notes(1,"Preserved the central question and four-stage pipeline; made OFF fidelity versus optimization explicit.","Problem: a useful optimizer needs a defensible forward model. Approach: isolate numerical reconstruction and implementation questions from scanner fidelity. Result: today's evidence is scoped to those separate tests; it is not a claim of successful physical ACDC optimization.",1.5)

    def slide02(self):
        s=self.slide(2,"Legacy intensity structure motivates controlled tests, not a causal claim","INTRODUCTION","Archived 06 V3 arrays · PI hypotheses from 10 Nov 2026 · not current 08/09 validation")
        self.figure(s,"legacy",.55,1.32,12.20,3.70)
        self.add_text(s,"PI observation: less apparent speckle in the legacy zero-B0 diagnostic. Pattern metrics are confounded by object structure and residual coherence.",.68,5.00,12.0,.61,size=14)
        self.add_text(s,"Test these hypotheses separately",.68,5.70,3.04,.55,size=15,bold=True,color="blue")
        for i,label in enumerate(("Encoded k-grid / conventions","Finite ADC dwell","Receive sensitivity / RSS","B0 / voxel model")):
            x=3.80+2.21*i
            self.box(s,x,5.72,2.05,.80,label,title_size=13)
        self.notes(2,"Replaced legacy deck screenshots with regenerated archived-array panels; separated PI observation from mechanisms; added an editable diagnostic decision row.","Problem: structured intensity/apparent speckle is visible in the legacy 06 V3 simulation. Approach: regenerate the actual archived arrays and compare transparent fixed-mask metrics, then isolate k-grid, ADC-aperture, receive and B0 hypotheses. Result: visual appearance alone does not establish stochastic noise or a B0 cause. CV is confounded by the structured object and fitted unresolved coherence; it must not be read as a pure speckle measure.",2.5,extra=json.dumps(self.data.get('legacy',{}),indent=2))

    def slide03(self):
        s=self.slide(3,"OFF-model fidelity is a measured-scan test, separate from optimization","INTRODUCTION","Measured unshimmed EPI versus simulated I=0 · fixed physical grid, mask and gain")
        self.par(s,"An optimized simulation can inherit an incorrect baseline.","Measured RSS OFF E, simulated OFF S, one fixed mask M and gain.",self.fact('off.nrmse_percent',lambda v:f'{v:.2f}% OFF NRMSE')+"; separate from the GRE optimization objective.",size=14)
        self.figure(s,"off_comparison",.60,2.28,12.09,3.50)
        self.equation(s,"gain",.78,5.93,5.57,.72,size=22)
        self.equation(s,"nrmse",6.76,5.93,5.77,.72,size=22)
        self.add_text(s,"Model validation: measured OFF. Current optimization: simulated EPI versus measured GRE. Experimental ACDC ON/OFF: a separate comparison.",.83,6.73,11.64,.34,size=12.5)
        self.notes(3,"Added the measured OFF versus current simulated OFF comparison and genuine gain/NRMSE equations; removed the invented figure slot.","Problem: implementation parity and lower optimization loss do not validate a measured image. Approach: compare magnitude images on the same physical grid and fixed mask using one stated positive least-squares gain. Result: use the newly generated NRMSE printed on the slide and the exact mask/gain below. This metric is separate from optimized simulated EPI versus measured GRE and from experimentally measured ACDC ON/OFF. A fresh offline replay of archived currents is included below only as supplementary evidence; it is not a new optimization or measured ACDC-ON validation, and it is not restored as a removed presentation topic.",2.5,extra=json.dumps(dict(off=self.data.get('off',{}),supplementary_archived_command_replay=self.data.get('optimization_replay',{})),indent=2))

    def slide04(self):
        s=self.slide(4,"Four measured inputs constrain geometry and contrast; receive fields remain unmodeled","METHODS","Current EPI code audit · receiver streams and ACDC shim channels are different systems")
        inputs=(("GRE reference","Measured image target; effective amplitude/contrast"),("EPI, ACDC OFF","Independent baseline validation; Pulseq ADC coordinates"),("32-echo MEGRE","B0, residual decay and unresolved coherence"),("3D ACDC maps","32 shim fields [Hz/A]; local slopes and corner checks"))
        for i,(title,body) in enumerate(inputs):
            x=.68+3.12*i
            self.box(s,x,1.58,2.73,1.75,title,body,title_size=17,body_size=14)
            self.line(s,x+1.36,3.36,x+1.36,3.75,arrow=True,color="blue")
        self.box(s,.83,3.84,7.45,1.23,"One physical frame",self.fact('geometry.summary'),title_size=18,body_size=15)
        self.equation(s,"coordinates",.91,5.27,7.25,.65,size=21)
        self.cube(s,9.35,3.85,w=2.20,h=1.40)
        self.add_text(s,"Full slice thickness\n"+self.fact('geometry.slice_thickness_mm',lambda v:f'{v:g} mm'),9.00,5.47,3.00,.69,size=15,align=PP_ALIGN.CENTER)
        self.add_text(s,"Receive channels c",.86,6.21,3.12,.3,size=15,bold=True,color="blue")
        self.add_text(s,self.fact('receiver.slide_summary'),3.60,6.16,8.93,.62,size=14)
        self.notes(4,"Preserved four measured-input roles; improved editable geometry/arrows; distinguished 32 ACDC shim coils from nonconsecutive MRI receiver IDs.","Problem: RSS contrast and coordinate errors can be confused with a physics error. Approach: audit the current Twix readers and EPI forward model. Result: measured data use individual receive streams and RSS/complex echo processing, but current EPI forward simulation uses one uniform receiver and an effective RSS-derived amplitude. The spiral receiver feature is separate and does not establish receiver-sensitive EPI simulation. B1+ excitation scaling and B1− receive sensitivity are distinct.",2.5,extra=json.dumps(self.data.get('receiver',{}),indent=2))

    def slide05(self):
        s=self.slide(5,"Every voxel contributes to every ADC sample before image reconstruction","METHODS","Generation 08 · one finite full-thickness voxel per retained native pixel · j: shim; v: voxel; n: ADC; a: voxel axis")
        # Preserve the source slide's voxel-to-sample scientific flow, making
        # sampling, volume integration, summation and reconstruction explicit.
        stages=((.58,1.78,"Native image grid"),(2.62,2.14,"MRzero center"),(5.02,2.22,"Finite 3D voxel"),(7.51,2.72,"One ADC sample n"),(10.48,2.29,"All ADC samples"))
        for x,w,title in stages:self.box(s,x,1.45,w,1.68,title,title_size=15)
        for x1,x2 in ((2.39,2.56),(4.79,4.96),(7.27,7.45),(10.25,10.42)):
            self.line(s,x1,2.28,x2,2.28,arrow=True,color="blue",width=1.2)
        for row in range(5):
            for col in range(5):
                shape=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(.94+.20*col),Inches(2.10+.13*row),Inches(.20),Inches(.13))
                shape.fill.background();shape.line.color.rgb=RGBColor.from_string(COLOR['line']);shape.line.width=Pt(.4)
        self.add_text(s,self.fact('geometry.matrix_label'),.73,2.80,1.50,.25,size=12,align=PP_ALIGN.CENTER)
        self.equation(s,"center",2.94,2.08,1.48,.53,size=25,center=True)
        self.add_text(s,"one complex value\nper voxel and ADC",2.75,2.65,1.88,.44,size=12,align=PP_ALIGN.CENTER)
        self.equation(s,"F",5.13,2.10,2.00,.60,size=20,center=True)
        self.equation(s,"s_sum",7.65,2.10,2.44,.61,size=20,center=True)
        self.add_text(s,self.fact('geometry.sample_label')+"\n(kx, ky, time)",10.62,2.17,2.00,.69,size=15,align=PP_ALIGN.CENTER)
        self.add_text(s,"For each retained voxel v at each ADC n: q is a local slope in cycles across that voxel.",.72,3.37,11.95,.36,size=16,bold=True,color="blue")
        self.equation(s,"q",.68,3.83,11.96,.76,size=22,center=True)
        self.equation(s,"J",.72,4.57,5.55,.82,size=21)
        self.equation(s,"tn_k",6.47,4.62,6.08,.64,size=20)
        self.line(s,12.87,3.18,12.87,5.88,color="blue")
        self.line(s,12.87,5.88,12.64,5.88,arrow=True,color="blue")
        self.add_text(s,"Reconstruct all samples",.74,5.56,3.60,.36,size=17,bold=True,color="blue")
        self.equation(s,"rho",.72,6.00,10.20,.66,size=21)
        self.add_text(s,"EPI image",11.16,5.78,1.41,.43,size=18,bold=True,color="blue",align=PP_ALIGN.CENTER)
        self.add_text(s,"Residual fitted coherence is already included in m-point; no second attenuation factor. Encoding uses kz=0; z denotes the slice normal.",.76,6.77,11.86,.34,size=11.5)
        self.notes(5,"Rebuilt the requested editable voxel→one ADC→all samples→image diagram, with genuine linked LaTeX and explicit indices/time integration.","Problem: an image pixel is not an independent MRI signal sample. Approach: one center history per retained full-thickness voxel contributes to every ADC. At fixed n, calculate the hardware-response integrated currents J, local q along each full voxel edge, three-axis coherence F, then sum all voxel contributions to one complex s_n. Repeat for all nominal Pulseq k/time coordinates and reconstruct. Result: changing n changes encoding, elapsed time, current history and voxel-specific coherence. These coordinates are calculated from the supplied sequence at actual ADC centers, not independently measured gradient trajectories. Fitted unresolved coherence belongs once in the fixed/point amplitude. Current EPI receive sensitivities are not separately simulated. The direct sum shown is the adjoint convention; the production 08 V2 inverse is distinguished on slide 8.",4,extra="q uses k extended with kz=0 in the local slice frame. e is a full edge vector [m]; b is center shim response [Hz/A]; g_B0 is Hz across one full voxel axis; g_ACDC is Hz/A across that axis; J is A·s. RF-relative integrals retain the hardware response and zero-current windows. See acdc_gpu.py response_matrices and voxel_3d.py _voxel_integrated_currents.")

    def slide06(self):
        s=self.slide(6,"Generation 08 caches fixed center histories, then recomputes current-dependent terms","METHODS","experiment_mrzero.py · experiment_echo.py · voxel_3d.py · effective amplitude is not quantitative PD")
        self.par(s,"Repeated MRzero setup would dominate current optimization.","Cache native center histories; retain fixed unresolved coherence.","Recompute center phase, three sinc factors and exact derivatives.",size=14)
        for i,(title,body) in enumerate((("Measured .seq","RF-relative ADC time; gradient moments; receiver phase"),("MRzero PDG","Ideal 90° history; voxel centers; fixed B0 and residual T2"),("Native history cache","Fixed signal = native center × unresolved coherence, applied once"))):
            x=.75+i*4.22
            self.box(s,x,2.56,3.64,1.43,title,body,title_size=17,body_size=14)
            if i<2:self.line(s,x+3.69,3.24,x+4.12,3.24,arrow=True,color="blue")
        self.equation(s,"PD",.86,4.22,5.70,.67,size=22)
        self.equation(s,"unresolved",6.78,4.22,5.45,.67,size=21)
        self.equation(s,"point",.85,5.15,11.49,.86,size=25,center=True)
        self.equation(s,"fixed_assumptions",.83,6.23,5.62,.49,size=19)
        self.add_text(s,"Effective echo-anchored amplitude retains measured contrast/RSS weighting; no quantitative PD.",6.82,6.22,5.63,.70,size=14)
        self.notes(6,"Corrected cache scope and receiver claim; made the once-only unresolved coherence and effective-PD convention explicit with vector LaTeX.","Problem: an effective RSS-derived object is not independently measured proton density. Approach: undo residual exponential decay to RF time, let MRzero evolve a fixed ideal excitation history, and include fitted unresolved coherence once. Result: generation 08 caches native center histories once per ADC block before iterations; the cache build makes multiple compute_graph/execute_graph calls and is not one graph invocation. In the reported CPU run, _base_cache is None and _native_cpu_blocks contains unmodified native point histories; voxel_3d._make_base applies the fixed unresolved coherence each time those blocks are read. If the full device cache fits, that full cache contains the coherence factor already. The displayed fixed-signal product is a mathematical definition, not a claim that every cache implementation literally stores the product. Current-dependent center phase and 3D voxel factors remain outside the fixed cache. This factorization assumes stationary spins after one ideal excitation. Transmit B1+=1 does not imply measured or modeled receive C_c(r).",2.5)

    def slide07(self):
        s=self.slide(7,"A local linear phase across one 3D voxel integrates to three sinc factors","METHODS","voxel_3d.py · full voxel edge vectors · normalized internal coordinates")
        self.add_text(s,"One voxel; three internal axes",.78,1.48,3.60,.39,size=17,bold=True,color="blue")
        self.cube(s,1.14,2.17,w=2.57,h=2.18,axes=True)
        self.equation(s,"u",.73,4.67,3.61,.66,size=21,center=True)
        self.add_text(s,"q varies with voxel v and sample n.\nIt is a slope across the voxel, not the phase itself.",.85,5.58,3.49,.87,size=16,bold=True,color="blue")
        self.equation(s,"phi",4.69,1.55,7.75,.82,size=22)
        self.equation(s,"integral",4.67,2.67,7.86,1.10,size=24)
        self.equation(s,"sinc",5.12,4.08,7.00,.70,size=21)
        self.add_text(s,"Encoding: k [cycles/m] · full edge e [m] → cycles\nB0 slope: gB0 [Hz across axis] × t [s] → cycles\nShim slope: gACDC [Hz/A across axis] × J [A·s] → cycles",4.83,5.12,7.65,1.16,size=15)
        self.add_text(s,"Through-slice B0 slope: assumed zero. Through-slice ACDC slope: calibrated 3D maps.\n06 V3: 8 z samples + 2 sinc factors → 08: one full-thickness box + 3 sinc factors.",.83,6.58,11.71,.52,size=12.5)
        self.notes(7,"Replaced pseudo-equations with the normalized 3D integral, explicit local q and genuine sinc definition; added an editable wireframe voxel and units.","Problem: treating q as a global phase obscures what causes intravoxel cancellation. Approach: use normalized internal coordinates u_a in [-1/2,1/2]. A locally linear field produces a voxel-specific phase slope at a given ADC; integrating exp(-i phase) over the rectangular volume yields the product of three normalized sinc factors. Result: calibrated ACDC gradients can change all three slopes, while the unmeasured normal B0 slope remains zero by assumption. Unresolved echo dispersion is not assigned a z gradient. At zero ACDC current, kz=0 and the assumed normal B0 slope is zero, so q_z=0 and the third sinc equals one. Therefore 06/08 OFF predictions are expected to agree for identical object/support/relaxation and other conventions; replacing eight z samples by one full-thickness box alone does not improve OFF fidelity.",3)

    def slide08(self):
        s=self.slide(8,self.fact('reconstruction.title'),"METHODS / RESULTS","Same complex 08 k-space · same grid/sign/origin/normalization/mask/gain · Pulseq-calculated ADC coordinates")
        self.add_text(s,"Required hypothesis test",.74,1.35,3.85,.31,size=15,bold=True,color="blue")
        self.add_text(s,"08 automatic Pulseq-coordinate operator  ↔  forced direct adjoint IDFT",.74,1.77,11.77,.48,size=18,bold=True)
        self.figure(s,"reconstruction",.60,2.27,12.08,3.03)
        self.equation(s,"rho",.73,5.32,7.92,.68,size=19)
        self.equation(s,"svd",9.00,5.34,3.49,.64,size=20)
        self.add_text(s,"Complex image error: "+self.fact('reconstruction.complex_relative_error',scientific_text)+
            "   |   magnitude NRMSE: "+self.fact('reconstruction.magnitude_nrmse_percent',lambda v:scientific_text(v)+'%')+
            "   |   Cartesian sanity: "+self.fact('reconstruction.synthetic_relative_error',scientific_text),.79,6.12,11.78,.44,size=13,bold=True,color="blue")
        self.add_text(s,"08 V2 uses an SVD inverse; direct adjoint is the control. The simulated intensity pattern is unchanged.\nThis rejects supplied-k operator numerics, not unmeasured scanner trajectory errors; DFT is not intrinsically superior.",.79,6.59,11.74,.55,size=12.5)
        self.notes(8,"Inserted matched real-coordinate reconstruction figures and Cartesian sanity test; distinguished current 08 V2 SVD inverse from the requested direct-adjoint control.","Problem: an encoded-grid/reconstruction-grid mismatch was proposed as a cause of apparent speckle. Approach: hold the exact complex generation-08 k-space fixed; compare the automatic actual-coordinate reconstruction with a forced direct adjoint at the same coordinates, then evaluate the current production 08 V2 matrix pseudoinverse and a deliberately forced nominal-grid IFFT separately. Result: use rerun metrics and title; on exactly Cartesian data, an ordinary IFFT and direct IDFT must agree. This experiment does not justify a blanket DFT accuracy claim.",3,extra=json.dumps(self.data.get('reconstruction',{}),indent=2))

    def slide09(self):
        s=self.slide(9,"Generation 09 changes where the same voxel physics is evaluated","METHODS","native_b0_model.py · locally modified native_pdg.execute_graph · stationary single-RF, D=0 assumptions")
        self.add_text(s,"08 — cached external correction",.93,1.41,5.45,.42,size=20,bold=True,color="blue")
        self.add_text(s,"09 — local MRzero executor",7.10,1.41,5.35,.42,size=20,bold=True,color="blue")
        left=("Standard PDG once → fixed center cache","Center phase from actual-current integrals","Analytic 3D sinc × cached center histories","Exact analytic current derivatives")
        right=("Recompute PDG at every evaluation","b0_at_adc_Hz: accumulated total field","local_k_at_adc_mInv: per-voxel box kernel","PyTorch autograd through native emission")
        for col,items in ((.89,left),(7.03,right)):
            for i,text in enumerate(items):
                self.box(s,col,2.12+i*.83,5.28,.64,text,title_size=14)
                if i<3:self.line(s,col+2.64,2.78+i*.83,col+2.64,2.91+i*.83,arrow=True,color="blue")
        self.equation(s,"B0effective",1.10,5.75,10.93,.76,size=25,center=True)
        self.add_text(s,"Same currents, center B0, local gradients, residual coherence and reconstruction. Agreement establishes implementation equivalence under these assumptions—not scanner accuracy.",.88,6.59,11.57,.55,size=14,bold=True)
        self.notes(9,"Retained the two implementation paths; corrected the local-executor API names and scope; typeset effective B0 and removed any implication of a new physical model.","Problem: external correction and a modified native executor must not be conflated with different physics. Approach: give 09 the exact hardware-response accumulated center field and per-voxel local trajectories for the native MRzero box kernel; rerun compute_graph/execute_graph at every evaluation and differentiate emission with autograd. Result: under one ideal excitation, no motion and D=0, event-averaged B0 reproduces the same phase and sinc physics as 08. Current EPI receive sensitivities remain uniform in both paths.",2.5)

    def slide10(self):
        s=self.slide(10,self.fact('off.title'),"RESULTS","Current input/echo-fit settings · zero currents · before optional EPI nuisance calibration")
        self.figure(s,"off_main",.56,1.35,12.19,3.95)
        self.add_text(s,self.fact('off.nrmse_percent',lambda v:f'{v:.2f}% OFF NRMSE'),.83,5.48,4.10,.62,size=26,bold=True,color="teal")
        self.add_text(s,"Magnitude gain g = "+self.fact('off.gain',lambda v:f'{v:.6g}')+"\n"+self.fact('off.mask_summary'),5.00,5.50,7.43,.78,size=15)
        self.add_text(s,self.fact('off.context_summary'),.85,6.47,11.58,.53,size=13)
        self.notes(10,"Replaced every missing figure slot and stale 45.98% number with fresh matched GRE, measured OFF, simulated OFF and residual results; recorded gain/mask/stage/timing.","Problem: the old slide quoted an archived first-run number without rendering the comparison. Approach: rerun the current selected acquisition and 32-echo preparation settings, forward simulate generation 08 at zero current, and compare on the stated fixed full-FOV mask after one reporting gain. Result: the title and NRMSE are generated from this run. Quantitative OFF-model disagreement remains separate from implementation parity and from any simulated current-optimization objective. No measured ACDC success is inferred.",3,extra=json.dumps(self.data.get('off',{}),indent=2))

    def slide11(self):
        s=self.slide(11,self.fact('equivalence.title'),"RESULTS","Matched 08/08 V2/09 input, device and precision · separate setup, reconstruction and loss+gradient timings")
        self.equation(s,"epsilon",.81,1.36,3.29,.65,size=23)
        self.add_text(s,"= "+self.fact('equivalence.max_signal_relative_error',scientific_text)+"  (maximum tested)",4.15,1.50,6.50,.43,size=24,bold=True,color="teal")
        self.add_text(s,self.fact('equivalence.case_summary'),.84,1.97,11.61,.30,size=12.5)
        self.figure(s,"equivalence",.58,2.30,5.95,3.55)
        self.figure(s,"runtime",6.77,2.30,5.95,3.55)
        self.add_text(s,self.fact('runtime.context_summary'),.84,5.98,11.61,.66,size=11.5)
        self.add_text(s,"Implementation parity is not scanner validation. Setup/cache, per-evaluation cost and file I/O are distinct.",.84,6.76,11.59,.31,size=13.5,bold=True,color="blue")
        self.notes(11,"Replaced archived parity with rerun per-ADC/image residuals and added fair reproducible speed benchmarks with setup separated from repeated work.","Problem: numerical agreement and runtime claims need matched data and explicit timing scope. Approach: compare full complex ADC arrays and matched images for zero and deterministic non-optimized currents; benchmark the same prepared object, device, dtype and reconstruction grid. Time setup/cache separately, then repeated forward+reconstruction, reconstruction alone, and loss+gradient with documented warmups/repetitions. Result: report generated medians and IQR; no timing is inferred from file timestamps or confused with preprocessing. CPU timings do not imply a GPU speed ranking.",3,extra=json.dumps(dict(equivalence=self.data.get('equivalence',{}),runtime=self.data.get('runtime',{})),indent=2))

    def slide12(self):
        s=self.slide(12,"Backup A — Finite ADC dwell is a separate diagnostic from center sampling","BACKUP","Production evaluates ADC centers; a rectangular-window quadrature test does not silently change production physics")
        self.equation(s,"sample_center",.85,1.27,4.97,.54,size=23)
        self.equation(s,"sample_window",6.99,1.27,5.43,.54,size=22)
        self.figure(s,"sampling_controls",.61,1.88,12.08,4.70)
        self.add_text(s,self.fact('uniform.summary'),.79,6.59,5.78,.43,size=11.8)
        self.add_text(s,self.fact('adc_window.summary'),6.95,6.59,5.53,.43,size=11.8)
        self.notes(12,"Kept the ADC-dwell topic; typeset center/window formulas and inserted raw uniform-phantom and aperture controls with their limitations.","Problem: sampling at a center time neglects finite ADC aperture. Approach: compare the production midpoint signal with explicitly documented rectangular-window quadrature at zero ACDC current. The test varies exact nominal Pulseq k(t), B0 phase, decay/coherence and the voxel box factors at sub-times. Dynamic-current evolution during the aperture was not tested. Do not modify the optimizer or claim the scanner has this exact rectangular receive aperture. Also control amplitude, support, exponential decay and unresolved coherence in a separate uniform-phantom B0 comparison: switching measured B0 changes center phase AND local slopes, so its variation is not a pure intravoxel-gradient effect. Result: use generated differences/convergence and stated dwell. Uniform-phantom variation includes support and voxel-window effects and is not automatically random noise; its mask differs from the legacy structured-object diagnostic and its CV values must not be compared across those masks.",0,extra=json.dumps(dict(adc_window=self.data.get('adc_window',{}),uniform_phantom=self.data.get('uniform',{})),indent=2))

    def slide13(self):
        s=self.slide(13,"Backup B — Reconstruction controls retain Pulseq coordinates and conventions","BACKUP","Pulseq-calculated k [cycles/m], RF-relative ADC time [s], common image grid [m] and receiver/navigation ordering")
        self.figure(s,"coordinates",.65,1.40,12.01,3.59)
        checks=(("Coordinates","No silent rounding to a nominal k-grid."),("Sample order","Reverse ADC times with data on alternating lines; then sort rows."),("Origin / frame","Use the same output grid, origin and frame matrix O."),("Normalization","Fixed inverse scale, mask, gain and navigator handling."))
        for i,(title,body) in enumerate(checks):
            x=.81+3.13*i
            self.add_text(s,title,x,5.18,2.82,.30,size=14,bold=True,color="blue")
            self.add_text(s,body,x,5.57,2.82,.64,size=12.5)
        self.equation(s,"sign",.88,6.24,11.54,.66,size=24,center=True)
        self.notes(13,"Retained coordinate/convention checks; added fresh k-space/timestamp diagnostics and real Fourier-sign math.","Problem: reversing samples without their timestamps or silently rounding k can manufacture a reconstruction difference. Approach: use the same actual k arrays, ADC elapsed times, sample order, output origin/grid and passive frame, then use a clearly labeled forced nominal-grid diagnostic only as a control. Result: the required matched automatic/direct experiment changes the operator implementation, not the input signal or physical coordinate convention.",0,extra=json.dumps(self.data.get('reconstruction',{}),indent=2))

    def slide14(self):
        s=self.slide(14,"Backup C — Measured receiver streams are not modeled EPI receive fields","BACKUP","Current EPI implementation audit · transmit B1+ and receive sensitivity Cc are distinct")
        rows=(("RF excitation","Ideal nonselective 90° history",self.fact('assumptions.rf')),
              ("Relaxation","Echo-relative effective amplitude / residual R2",self.fact('assumptions.extrapolation')),
              ("Transmit / receive","Uniform B1+; EPI forward uses one uniform receiver",self.fact('receiver.slide_summary')),
              ("Intravoxel fields","Linear spatial phase in a rectangular full-thickness voxel","Measured in-plane B0; normal B0 slope assumed zero"),
              ("ADC / spin physics","Center-time samples; stationary spins; no diffusion","Dwell test is diagnostic; no refocusing or GRE steady state"))
        for i,(label,model,qualification) in enumerate(rows):
            y=1.47+i*.90
            self.line(s,.82,y+.76,12.48,y+.76,color="line",width=.65)
            self.add_text(s,label,.84,y,2.10,.55,size=16,bold=True,color="blue")
            self.add_text(s,model,3.03,y,4.23,.65,size=14)
            self.add_text(s,qualification,7.58,y,4.83,.67,size=14)
        self.add_text(s,"Planned receive-sensitive EPI forward model (not implemented here)",.86,6.08,11.54,.31,size=14,bold=True,color="blue")
        self.equation(s,"receiver_forward",.88,6.48,5.27,.47,size=18)
        self.equation(s,"rss",6.67,6.45,5.63,.51,size=18)
        self.notes(14,"Updated stale receiver assumptions after auditing current EPI and separate spiral code; retained assumptions topic and showed future receiver/RSS equations explicitly as planned.","Problem: reading 32 complex receiver streams is not the same as simulating 32 receive-sensitive EPI signals. Approach: identify actual physical IDs and inspect the EPI forward summation. Result: current EPI has a uniform receiver and RSS-derived effective amplitude. A future C_c model must normalize sensitivities without double-counting that amplitude, reconstruct each receiver consistently and combine RSS. No before/after receiver improvement was measured, and the spiral-only feature is not claimed for EPI.",0,extra=json.dumps(dict(receiver=self.data.get('receiver',{}),assumptions=self.data.get('assumptions',{})),indent=2))

    def slide15(self):
        s=self.slide(15,"Backup D — Fresh figures are reproducible from exact inputs and working-tree hashes","BACKUP","Current manually selected 15-slide source preserved · source deck and original physics notebooks remain unchanged")
        self.add_text(s,"Current revision",.89,1.44,2.63,.34,size=17,bold=True,color="blue")
        self.add_text(s,self.fact('provenance.git_sha'),3.74,1.44,8.62,.50,size=17)
        self.add_text(s,"Code identity",.89,2.24,2.63,.34,size=17,bold=True,color="blue")
        self.add_text(s,"Generation files include untracked/modified sources; SHA256 hashes identify the actual code, beyond Git HEAD.",3.74,2.24,8.62,.77,size=17)
        self.add_text(s,"Reproduce",.89,3.35,2.63,.34,size=17,bold=True,color="blue")
        self.add_text(s,"Run presentation/00_generate_weekly_figures.ipynb top to bottom.\nRUN_FAST uses validated caches; RUN_EXPENSIVE reruns simulation and timing.",3.74,3.35,8.62,.89,size=16)
        self.add_text(s,"Artifacts",.89,4.55,2.63,.34,size=17,bold=True,color="blue")
        self.add_text(s,"figures_manifest.json · results_metrics.csv · data/*.npz\nequations/*.tex + SVG/PDF · slide_changes.json · rendered PDF/PNGs",3.74,4.55,8.62,.86,size=16)
        self.add_text(s,"Limits",.89,5.78,2.63,.34,size=17,bold=True,color="blue")
        self.add_text(s,self.fact('provenance.limitations_summary'),3.74,5.78,8.62,.88,size=16)
        self.notes(15,"Replaced archived-README-only provenance with actual rerun inputs, Git/working-tree hashes, portable figure notebook, artifact paths and limitations.","Problem: an archived README or Git SHA alone cannot identify today's untracked scientific implementation. Approach: save exact executed preparation cells, source and input hashes, metadata, masks, gains, arrays, timing scope and literal equation sources. Result: every data figure is regenerated from project arrays/functions rather than a slide screenshot; provenance links it to the selected acquisition and current working tree. The original 15-slide selection and order are retained.",0,extra=json.dumps(self.data,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=None)
    parser.add_argument('--draft',action='store_true',help='Allow visibly labeled pending numeric/figure fields during layout review only.')
    parser.add_argument('--equations-only',action='store_true')
    args=parser.parse_args()
    if args.equations_only:
        generate_equations();return
    if args.data is None and not args.draft:
        from export_slide_data import export_slide_data
        data=export_slide_data()
    else:
        data=read_json(args.data or HERE/'presentation_slide_data.json',{})
    Deck(data,draft=args.draft).build()


if __name__=='__main__':
    main()
