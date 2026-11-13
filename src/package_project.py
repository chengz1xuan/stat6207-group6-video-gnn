"""Package executable source, frozen experiments and report; exclude local credentials/caches."""
from pathlib import Path
import json,hashlib,zipfile
R=Path(__file__).resolve().parents[1]


def collect_paths(root=R):
    roots=['src','Scripts','artifacts/v2','data/processed','data/youtube','data_topic3','.streamlit','output/latex']
    config=root/'artifacts/final_v4.json'
    if config.exists():
        final=json.loads(config.read_text())
        roots += [final['prepared_directory'],final['paired_directory']]
    paths=set()
    if config.exists() and final.get('showcase_path'):
        brief=(root/final['showcase_path']).resolve()
        if root.resolve() not in brief.parents or not brief.is_file():
            raise ValueError('The report-bound author brief is missing or outside the project.')
        paths.add(brief)
    for directory in roots:
        base=(root/directory).resolve()
        if root.resolve() not in base.parents:
            raise ValueError('Package input must remain within the project.')
        for p in base.rglob('*'):
            relative=p.relative_to(root)
            if not p.is_file() or any(s in relative.parts for s in ['__pycache__','tmp','.git','.local-tools','.project-deps']):continue
            if p.name=='.DS_Store' or p.suffix in ['.zip','.gz','.pyc','.log','.cjs','.aux','.out']:continue
            if p.name.startswith('probe'):continue
            if p.name in ['deployment.json','web_model.json','browser_validation.json','inference_validation.json','prepare_website.py','make_report.py'] or p.name.startswith('website-'):continue
            paths.add(p)
    required=['app.py','AUTHORS.json','README.md','PROJECT_REQUIREMENTS.md','requirements.txt','SESSION_WORKLOG.md','output/pdf/Group6_Final_Report.pdf','references/Group6_Plan.pdf','references/ieee/IEEEtran.cls','references/ieee/SOURCE.txt']
    for name in required:
        p=root/name
        if not p.is_file():raise FileNotFoundError(p)
        paths.add(p)
    for name in ['artifacts/final_v4.json','output/session_transcript.md','output/交付说明.txt','INTEGRATION_GUIDE.md','INTEGRATION_TEST_REPORT.md']:
        p=root/name
        if p.is_file():paths.add(p)
    assert root/'Scripts/text_policy.py' in paths
    return sorted(paths)


def package(root=R):
    paths=collect_paths(root)
    checks={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    ck=root/'output/SHA256SUMS.json';ck.write_text(json.dumps(checks,indent=2,ensure_ascii=False));paths.append(ck)
    archive=root/'output/Group6_Final_Project.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,Path('Group6_Final_Project')/p.relative_to(root))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert 'Group6_Final_Project/Scripts/text_policy.py' in z.namelist()
        assert all('/.local-tools/' not in n and '/.git/' not in n for n in z.namelist())
        print('Final ZIP:',len(z.namelist()),'files;',round(archive.stat().st_size/1024/1024,2),'MiB')
    return archive


if __name__=='__main__':package()
