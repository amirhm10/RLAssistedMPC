"""Package every verified figure and derived table; never copy native models."""
from pathlib import Path
import csv
import hashlib
import html
import json
import textwrap
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps, ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        while b:=f.read(4*1024**2):
            h.update(b)
    return h.hexdigest()


titles={
 "01_handoff_rewards":"Handoff reward and direct threshold comparisons",
 "02_learning_polymer":"Polymer: all families, all 300 episodes",
 "02_learning_column":"Column: all families, all 300 episodes",
 "03_paired_handoff":"Paired handoff tracking errors and input effort",
 "04_final_evaluation":"Final deterministic episode and trade-offs",
 "05_physical_polymer":"Polymer final outputs and inputs",
 "05_physical_column":"Column final outputs and inputs",
 "06_e2_comparators":"E2 comparators and supplementary threshold recovery",
 "07_gate_decision_regret":"E1 first-action regret at release",
 "08_gate_acceptance_errors":"E1 unfavorable acceptances and magnitude sensitivity",
 "09_prediction_accuracy":"E3 held-out prediction ratios",
 "10_capture_coverage":"Complete polymer and partial column E1 coverage",
 "11_reward_decomposition":"Why scalar reward and output RMSE can rank differently",
 "12_recovery_reward":"Recovery to matched-MPC reward",
 "13_recovery_both_output_rmse":"Recovery to both matched-MPC output RMSEs",
 "14_channel_effect_sizes":"Channel consequences and gating benefit",
 "15_steady_state_effects":"Pre-decision steady state and first-input effects",
 "16_unfavorable_choices_context":"Context and magnitude of unfavorable decisions",
 "17_reward_equivalence_and_supervisor":"Reward-equivalent settings and supervisor retention",
}
figures=sorted((HERE/"figures").glob("*.png"))
assert len(figures)==19
registry=[]
for p in figures:
    with Image.open(p) as image:
        image.verify()
    ET.parse(p.with_suffix(".svg"))
    registry.append(dict(title=titles[p.stem],png=p.relative_to(HERE).as_posix(),
        svg=p.with_suffix(".svg").relative_to(HERE).as_posix(),
        png_sha256=sha(p),svg_sha256=sha(p.with_suffix(".svg"))))
(HERE/"data/figure_registry.json").write_text(json.dumps(registry,indent=2)+"\n",encoding="utf-8")
cards="\n".join(f'<figure><h2>{i+1}. {html.escape(r["title"])}</h2><a href="{r["svg"]}"><img src="{r["png"]}" loading="lazy" alt="{html.escape(r["title"])}"></a><figcaption><a href="{r["svg"]}">Vector SVG</a> | <a href="{r["png"]}">PNG</a></figcaption></figure>' for i,r in enumerate(registry))
gallery=f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Practical handoff and recovery</title>
<style>body{{font:16px system-ui;margin:32px auto;max-width:1500px;padding:0 18px;background:#f3f5f7;color:#14262f}}figure{{background:white;padding:20px;margin:24px 0;border-radius:8px}}img{{width:100%;height:auto}}h2{{font-size:20px}}a{{color:#126c91}}</style>
<h1>Practical handoff and recovery assessment</h1><p>100 completed seeded training runs; 3,733 verified matched decisions; 19 figures.
Column E1 is partial at the frozen 22:36 UTC snapshot. Original data preserved; manuscript and responses unchanged.</p>
<p><a href="ASSESSMENT.md">Interpretation and definitions</a> | <a href="PRACTICAL_RESULTS.pdf">Complete PDF booklet</a> | <a href="data/recovery_times.csv">Recovery data</a> | <a href="data/steady_state_decisions.csv">Steady-state data</a></p>{cards}</html>"""
(HERE/"index.html").write_text(gallery,encoding="utf-8")
pdf=HERE/"PRACTICAL_RESULTS.pdf"
W,H=landscape(A4)
doc=canvas.Canvas(str(pdf),pagesize=(W,H))
doc.setTitle("Practical handoff, recovery and steady-state assessment")
doc.setCreator("Saved-result analysis")
doc.setFillColorRGB(.07,.20,.27)
doc.setFont("Helvetica-Bold",23)
doc.drawString(40,H-60,"Practical handoff and recovery")
doc.setFont("Helvetica",13)
doc.drawString(40,H-85,"Saved simulation evidence | 24 October 2026")
lines=[
 "100 completed seeded runs: two plants, five families, SG/plain, five seeds.",
 "3,733 paired E1 decisions: 3,200 polymer + 533/640 column (frozen 22:36 UTC).",
 "19 figures; complete data, SHA-256 provenance and independent verification.",
 "",
 "Combined SG reduces handoff disruption compared with ungated RL in all five",
 "paired seeds in both plants. Benefits are strongest for residual and combined.",
 "",
 "Median first recovery of BOTH output RMSEs to matched MPC (three episodes):",
 "  Polymer: SG 5 episodes; ungated RL 19.",
 "  Column: SG 24; ungated RL 78 (one ungated seed not recovered by episode 200).",
 "",
 "Different horizon/weight settings are often nearly equivalent near steady state.",
 "Supervisor retention in those cases varies by plant and channel.",
 "",
 "Counterexamples are retained: threshold can win individual recovery metrics;",
 "late polymer performance and E1 critic decisions are not uniformly better.",
 "",
 "No new simulation or manuscript/reviewer-response edit. Raw data and recovery",
 "checkpoints remain in their verified original locations and lossless archives.",
 "Full definitions, per-seed results and interpretation: ASSESSMENT.md and data/."
]
y=H-125
doc.setFont("Helvetica",12)
for line in lines:
    doc.drawString(42,y,line)
    y-=20
doc.setFont("Helvetica",9)
doc.drawRightString(W-24,15,"1 / "+str(len(figures)+1))
doc.showPage()
for i,(path,record) in enumerate(zip(figures,registry),start=2):
    doc.setPageSize((W,H))
    doc.setFillColorRGB(.07,.20,.27);doc.setFont("Helvetica-Bold",11)
    doc.drawString(20,H-20,record["title"])
    with Image.open(path) as im:
        width,height=im.size
    scale=min((W-32)/width,(H-55)/height)
    pw,ph=width*scale,height*scale
    doc.drawImage(ImageReader(str(path)),(W-pw)/2,24+(H-55-ph)/2,width=pw,height=ph,preserveAspectRatio=True)
    doc.setFont("Helvetica",8)
    doc.drawString(20,10,path.name)
    doc.drawRightString(W-20,10,f"{i} / {len(figures)+1}")
    doc.showPage()
doc.save()
tmp=HERE/"tmp";tmp.mkdir(exist_ok=True)
sheet=Image.new("RGB",(4*430,5*310),"#eeeeee")
draw=ImageDraw.Draw(sheet)
for i,p in enumerate(figures):
    with Image.open(p) as im:
        im=im.convert("RGB")
        im.thumbnail((420,280))
        x=(i%4)*430+(430-im.width)//2;y=(i//4)*310+25
        sheet.paste(im,(x,y))
        draw.text(((i%4)*430+8,(i//4)*310+5),p.name,fill="black")
sheet.save(tmp/"contact_sheet.jpg",quality=90)
archive=HERE/"RESULTS_PACKAGE.zip"
files=[p for p in HERE.rglob("*") if p.is_file() and "tmp" not in p.relative_to(HERE).parts
       and "__pycache__" not in p.parts and p.name not in ("RESULTS_PACKAGE.zip","RESULTS_PACKAGE.sha256","package_manifest.json")
       and not p.name.endswith(".log")]
with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in sorted(files):
        z.write(path,path.relative_to(HERE).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
(HERE/"RESULTS_PACKAGE.sha256").write_text(sha(archive)+"  RESULTS_PACKAGE.zip\n",encoding="ascii")
manifest=dict(figures=len(figures),pdf_pages=len(figures)+1,pdf_sha256=sha(pdf),
    zip_sha256=sha(archive),zip_crc_passed=True,zip_bytes=archive.stat().st_size,
    source_files=[dict(path=p.relative_to(HERE).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)],
    original_scientific_results_removed=0,manuscript_or_reviewer_response_edits=0)
(HERE/"package_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:manifest[k] for k in ("figures","pdf_pages","zip_bytes","zip_crc_passed")}))
