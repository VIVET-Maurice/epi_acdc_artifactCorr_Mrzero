"""Render the revised deck and record objective structural checks.

Run after build_presentation.py. A virtual display may need permission to open
its local socket in restricted environments. Human visual review of all PNGs
remains necessary; structural checks do not establish scientific correctness.
"""
from __future__ import annotations
import json
from hashlib import sha256
import os
from pathlib import Path
import re
import subprocess
import sys

try:
    import pptx
except ImportError:
    sys.path.insert(0,"/tmp/presentation_deps")
from PIL import Image,ImageDraw
from pptx import Presentation

HERE=Path(__file__).resolve().parent
DECK=HERE.parent/'2026_11_11_ACDC_MRzero_WeeklyUpdate_REVISED.pptx'
OUT=HERE/'rendered'


def render():
    OUT.mkdir(exist_ok=True)
    env=os.environ.copy()
    for name,suffix in (('XDG_RUNTIME_DIR','runtime'),('XDG_CONFIG_HOME','config'),('XDG_CACHE_HOME','cache')):
        path=Path('/tmp')/f'acdc-presentation-{suffix}'
        path.mkdir(exist_ok=True)
        if name=='XDG_RUNTIME_DIR':path.chmod(0o700)
        env[name]=str(path)
    pdf=OUT/(DECK.stem+'.pdf')
    # Never mistake an earlier draft conversion for the current deck render.
    if pdf.exists():pdf.unlink()
    subprocess.run(['xvfb-run','-a','libreoffice','-env:UserInstallation=file:///tmp/acdc-final-render-profile',
        '--headless','--convert-to','pdf','--outdir',str(OUT),str(DECK)],env=env,check=True)
    if not pdf.exists():raise RuntimeError('LibreOffice did not create the requested PDF.')
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
    if pages!=15:raise AssertionError(f'Expected 15 rendered pages, found {pages}.')
    subprocess.run(['pdftoppm','-png','-r','180',str(pdf),str(OUT/'slide')],check=True)
    prs=Presentation(str(DECK));issues=[];slides=[]
    for number,slide in enumerate(prs.slides,1):
        text='\n'.join(shape.text for shape in slide.shapes if shape.has_text_frame)
        if re.search(r'FIGURE TO ADD|Awaiting generated figure|\bpending\b',text,re.I):
            issues.append(dict(slide=number,issue='unfinished placeholder text'))
        for shape in slide.shapes:
            if (shape.left < -2000 or shape.top < -2000 or
                shape.left+shape.width > prs.slide_width+2000 or shape.top+shape.height > prs.slide_height+2000):
                issues.append(dict(slide=number,shape=shape.name,issue='shape extends beyond slide bounds'))
        notes=slide.notes_slide.notes_text_frame
        if notes is None or 'Literal LaTeX' not in notes.text:
            issues.append(dict(slide=number,issue='missing scientific source/equation notes'))
        slides.append(dict(slide=number,shape_count=len(slide.shapes),native_text_boxes=sum(x.has_text_frame for x in slide.shapes),
                           preview=str(OUT/f'slide-{number:02d}.png')))
    files=sorted(OUT.glob('slide-*.png'))
    sheet=Image.new('RGB',(1600,1600),'#dddddd')
    for i,path in enumerate(files):
        image=Image.open(path).convert('RGB');image.thumbnail((520,293))
        x=(i%3)*533;y=(i//3)*320
        sheet.paste(image,(x,y));ImageDraw.Draw(sheet).text((x+6,y+297),str(i+1),fill='black')
    sheet.save(OUT/'contact_sheet.png')
    report=dict(deck=str(DECK),deck_sha256=sha256(DECK.read_bytes()).hexdigest(),pdf=str(pdf),
                deck_mtime=DECK.stat().st_mtime,pdf_mtime=pdf.stat().st_mtime,
                slide_count=len(prs.slides),rendered_pages=pages,
                native_text_and_shapes=True,slides=slides,structural_issues=issues,
                human_visual_audit='Review all 15 PNGs after final build; see visual_audit.md.')
    (OUT/'render_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(pdf=str(pdf),rendered_slides=pages,structural_issues=issues),indent=2))


if __name__=='__main__':render()
