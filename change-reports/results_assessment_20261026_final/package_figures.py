"""Package every verified figure and derived table; never copy native models."""
from pathlib import Path
import csv
import hashlib
import html
import json
import re
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
 "10_capture_coverage":"Complete polymer and column E1 coverage",
 "11_reward_decomposition":"Why scalar reward and output RMSE can rank differently",
 "12_recovery_reward":"Recovery to matched-MPC reward",
 "13_recovery_both_output_rmse":"Recovery to both matched-MPC output RMSEs",
 "14_channel_effect_sizes":"Channel consequences and gating benefit",
 "15_steady_state_effects":"Pre-decision steady state and first-input effects",
 "16_unfavorable_choices_context":"Context and magnitude of unfavorable decisions",
 "17_reward_equivalence_and_supervisor":"Reward-equivalent settings and supervisor retention",
 "18_threshold_handoff":"Direct threshold comparison: handoff disruption",
 "19_threshold_recovery":"Direct threshold comparison: recovery speed",
 "20_final_combined_decisions":"Completed combined decisions: favorable and adverse outcomes",
 "21_final_combined_training":"Accepted final combined main-training trajectory",
 "22_combined_steady_state":"Steady-state ambiguity in the final combined context",
}
figures=sorted((HERE/"figures").glob("*.png"))
assert len(figures)==24
registry=[]
for p in figures:
    with Image.open(p) as image:
        image.verify()
    ET.parse(p.with_suffix(".svg"))
    number=int(p.stem.split('_')[0])
    generator=('final_comparison.py' if number>=18 else 'steady_state_analysis.py' if number>=15
               else 'recovery_analysis.py' if number>=12 else 'reward_check.py' if number==11 else 'plot_results.py')
    registry.append(dict(title=titles[p.stem],png=p.relative_to(HERE).as_posix(),
        svg=p.with_suffix(".svg").relative_to(HERE).as_posix(),
        png_sha256=sha(p),svg_sha256=sha(p.with_suffix(".svg")),
        figure_generator=generator,
        analysis_scripts=[q.name for q in HERE.glob("*.py")],
        source_provenance=["data/repeated_provenance.json","data/e2_provenance.json","data/gate_provenance.json","data/e3_verification.json"],
        metric_definitions="ASSESSMENT.md and data/recovery_definitions.json",
        plant="column" if p.stem.startswith(("20_","21_","22_")) else "plants distinguished in panels",
        mode="critic/threshold/nominal" if p.stem.startswith(("18_","19_")) else "as labeled; replicated and conditional studies are separate",
        units="Physical outputs and identification-range normalization as labeled; reward comparable within plant only",
        claim_supported=titles[p.stem],limitations="Read figure subtitle and ASSESSMENT.md; single-seed threshold/E1 diagnostics do not establish universal superiority or formal safety"))
(HERE/"data/figure_registry.json").write_text(json.dumps(registry,indent=2)+"\n",encoding="utf-8")
cards="\n".join(f'<figure><h2>{i+1}. {html.escape(r["title"])}</h2><a href="{r["svg"]}"><img src="{r["png"]}" loading="lazy" alt="{html.escape(r["title"])}"></a><figcaption><a href="{r["svg"]}">Vector SVG</a> | <a href="{r["png"]}">PNG</a></figcaption></figure>' for i,r in enumerate(registry))
gallery=f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Practical handoff and recovery</title>
<style>body{{font:16px system-ui;margin:32px auto;max-width:1500px;padding:0 18px;background:#f3f5f7;color:#14262f}}figure{{background:white;padding:20px;margin:24px 0;border-radius:8px}}img{{width:100%;height:auto}}h2{{font-size:20px}}a{{color:#126c91}}</style>
<h1>Practical handoff and recovery assessment</h1><p>100 completed seeded training runs; 3,840 verified matched decisions; 24 figures.
Column E1 is complete, including combined320/320. The critic decision audit remains mixed; all counterexamples and original versions are retained. Manuscript and responses are unchanged.</p>
<p><a href="ASSESSMENT.md">Interpretation and definitions</a> | <a href="REVIEWER_READINESS.md">24-comment readiness check</a> | <a href="PRACTICAL_RESULTS.pdf">Complete PDF booklet</a> | <a href="data/recovery_times.csv">Recovery data</a> | <a href="data/steady_state_decisions.csv">Steady-state data</a></p>{cards}</html>"""
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
doc.drawString(40,H-85,"Saved simulation evidence | 26 October 2026")
lines=['100 completed replicated runs: two plants, five families, SG/plain, five seeds.', '3,840 paired E1 decisions: 3,200 polymer + all640 column, combined320/320.', '24 figures; complete raw-data checks, provenance and an unchanged prior assessment.', '', 'Critic versus simple threshold, handoff episodes14-20 (single-seed E2):', '  Polymer: output RMSEs37.0% /46.3% lower; input activity76.6% lower.', '  Column: output RMSEs6.3% /41.1% lower; input activity61.9% lower.', '', 'Reward recovery after release14, critic versus threshold: polymer1 vs25; column5 vs37.', 'Both-output RMSE recovery: polymer1 vs25; column37 vs22 (threshold wins).', 'Recovery requires three consecutive qualifying episodes relative to matched MPC.', '', 'Full combined E1:100/200 accepted policy actions have lower paired200-step return.', 'This supports practical handoff mitigation, not reliable critic ordering or formal safety.', 'Steady-state equivalence is partly supported; adverse decisions remain.', '', 'Reviewer preparation can begin. Unseen references, E4/E5 and analytical/literature', 'dependencies remain distinct. No comments answered or manuscript changes.', 'Read ASSESSMENT.md and REVIEWER_READINESS.md for definitions and limitations.']

lines=[re.sub(r"(?<=[a-z])(?=\d)"," ",line).replace("E1:100","E1: 100").replace("/46.3%","/ 46.3%").replace("/41.1%","/ 41.1%") for line in lines]

y=H-119
doc.setFont("Helvetica",11)
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
sheet=Image.new("RGB",(4*430,((len(figures)+3)//4)*310),"#eeeeee")
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
