from pathlib import Path
import requests,json,re
r=requests.get('https://www.youtube.com/watch?v=98Q60sqW-sQ',timeout=25)
print('YouTube:',r.status_code,'bytes',len(r.content),'player JSON','ytInitialPlayerResponse' in r.text)
Path('data/youtube/probe.html').write_text(r.text)
r=requests.get('https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest',timeout=25);j=r.json();print('Tectonic assets:',[(a['name'],a['browser_download_url']) for a in j.get('assets',[]) if 'apple' in a['name']])
