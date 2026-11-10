from pathlib import Path
import json,hashlib,zipfile
R=Path(__file__).resolve().parents[1]
paths=[]
for root in ['src','artifacts/v2','data/processed','data/youtube','data_topic3','.streamlit','output/latex']:
 for p in (R/root).rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts or p.name=='.DS_Store':continue
  if p.suffix in ['.zip','.gz','.pyc','.log','.cjs'] or p.name.startswith('probe'):continue
  if p.name in ['deployment.json','web_model.json','browser_validation.json','inference_validation.json','prepare_website.py','make_report.py'] or p.name.startswith('website-'):continue
  paths.append(p)
for name in ['app.py','README.md','PROJECT_REQUIREMENTS.md','requirements.txt','SESSION_WORKLOG.md','output/pdf/Group6_Final_Report.pdf','output/session_transcript.md','output/交付说明.txt','references/Group6_Plan.pdf','references/ieee/IEEEtran.cls','references/ieee/SOURCE.txt']:
 paths.append(R/name)
paths=sorted(set(paths));checks={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};ck=R/'output/SHA256SUMS.json';ck.write_text(json.dumps(checks,indent=2,ensure_ascii=False));paths.append(ck)
archive=R/'output/Group6_Final_Project.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in paths:z.write(p,Path('Group6_Final_Project')/p.relative_to(R))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert all('/archive/' not in p and '/artifacts/v1/' not in p for p in z.namelist())
 print('Final ZIP:',len(z.namelist()),'files;',round(archive.stat().st_size/1024/1024,2),'MiB')
