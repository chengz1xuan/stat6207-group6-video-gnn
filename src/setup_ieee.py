from pathlib import Path
import requests,tarfile,zipfile,io
out=Path('.project-deps/tex');out.mkdir(parents=True,exist_ok=True)
url='https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-aarch64-apple-darwin.tar.gz'
r=requests.get(url,timeout=60);r.raise_for_status()
with tarfile.open(fileobj=io.BytesIO(r.content),mode='r:gz') as t:
 m=next(m for m in t.getmembers() if Path(m.name).name=='tectonic');data=t.extractfile(m).read();p=out/'tectonic';p.write_bytes(data);p.chmod(0o755)
u='https://mirrors.ctan.org/macros/latex/contrib/IEEEtran.zip';r=requests.get(u,timeout=60);r.raise_for_status();z=zipfile.ZipFile(io.BytesIO(r.content));dest=Path('references/ieee');dest.mkdir(exist_ok=True)
for name in ['IEEEtran.cls','bare_conf.tex','IEEEtran_HOWTO.pdf']:
 member=next(n for n in z.namelist() if n.endswith('/'+name));(dest/name).write_bytes(z.read(member))
(dest/'SOURCE.txt').write_text('IEEEtran distribution: '+u+'\nSelected conference mode with a4paper.\nTectonic: '+url+'\n')
print('Downloaded IEEEtran class and Tectonic to project-local directories.')
