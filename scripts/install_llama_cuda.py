"""Install the official CUDA 12.4 reference binaries locally; preserve system PATH."""
import concurrent.futures,hashlib,json,urllib.request,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
tag='b11669'
req=urllib.request.Request(f'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}',headers={'User-Agent':'AdaForge-local-validation'})
with urllib.request.urlopen(req) as r:release=json.load(r)
names=[f'llama-{tag}-bin-win-cuda-12.4-x64.zip','cudart-llama-bin-win-cuda-12.4-x64.zip']
assets=[next(a for a in release['assets'] if a['name']==name) for name in names]
folder=root/'tools'/f'llama-{tag}-cuda12.4';folder.mkdir(parents=True,exist_ok=True)
binary=folder/'bin';binary.mkdir(exist_ok=True)
def download(asset):
    path=folder/asset['name'];print('Download: '+asset['name'],flush=True)
    if not path.exists() or path.stat().st_size!=asset['size']:
        with urllib.request.urlopen(asset['browser_download_url'],timeout=60) as r,path.open('wb') as f:
            while chunk:=r.read(4*1024*1024):f.write(chunk)
    digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    if asset.get('digest'):assert asset['digest']=='sha256:'+digest
    assert path.stat().st_size==asset['size']
    print('Verified: '+asset['name'],flush=True)
    return path,{'name':asset['name'],'url':asset['browser_download_url'],'sha256':digest}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:archives=list(pool.map(download,assets))
for archive,_ in archives:
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            assert (binary/info.filename).resolve().is_relative_to(binary.resolve())
        z.extractall(binary)
(folder/'provenance.json').write_text(json.dumps({'tag':tag,'assets':[a for _,a in archives]},indent=2),encoding='utf-8')
print('Installed under '+str(binary),flush=True)
