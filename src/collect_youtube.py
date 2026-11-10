"""One frozen public-metadata pass over supplied URLs; no credentials or login bypass."""
from pathlib import Path
import sys,concurrent.futures,json,re,hashlib,datetime,csv,time
import pandas as pd,requests
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/youtube';OUT.mkdir(exist_ok=True,parents=True)
def extract(text,name):
 m=re.search(r'(?:var )?'+re.escape(name)+r'\s*=\s*',text)
 if not m:return {}
 try:return json.JSONDecoder().raw_decode(text[m.end():])[0]
 except ValueError:return {}
def walk(x):
 if isinstance(x,dict):
  yield x
  for v in x.values():yield from walk(v)
 elif isinstance(x,list):
  for v in x:yield from walk(v)
def first(x,key):return next((z[key] for z in walk(x) if key in z),{})
def text(x):
 if isinstance(x,str):return x
 return x.get('simpleText') or x.get('content') or ''.join(v.get('text','') for v in x.get('runs',[]))
def exact(x):
 s=str(x).replace(',','').replace(' ','');return int(s) if re.fullmatch(r'\d+',s) else None
def count_from_text(s):
 m=re.search(r'(?<![\w.])([\d,]+)(?![\d.,kKmMbB])',s);return exact(m.group(1)) if m and not re.search(r'\d\s*[kKmMbB]|thousand|million|billion',s) else None
def collect(row):
 target=OUT/(row['youtube_id']+'.json')
 if target.exists():return json.loads(target.read_text())
 z={'video_id':row['video_id'],'youtube_id':row['youtube_id'],'source_url':row['source_url'],'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'views':None,'likes':None,'comments':None,'channel_id':None,'subscriber_count_display':None,'method':'YouTube public watch HTML embedded metadata; no authentication','status':'unavailable'}
 try:
  r=requests.get(row['source_url'],params={'hl':'en'},timeout=25);z['http_status']=r.status_code;z['response_sha256']=hashlib.sha256(r.content).hexdigest();j=extract(r.text,'ytInitialData');player=extract(r.text,'ytInitialPlayerResponse');primary=first(j,'videoPrimaryInfoRenderer');vc=first(primary,'videoViewCountRenderer');z['views']=exact(vc.get('originalViewCount')) or (exact(player.get('videoDetails',{}).get('viewCount')));z['platform_title']=text(primary.get('title',{}));owner=first(j,'videoOwnerRenderer');z['channel_id']=owner.get('navigationEndpoint',{}).get('browseEndpoint',{}).get('browseId');z['channel_name']=text(owner.get('title',{}));z['subscriber_count_display']=text(owner.get('subscriberCountText',{})) or None
  like=first(primary,'likeButtonViewModel');access=[a.get('accessibilityText','') for a in walk(like) if a.get('accessibilityText')];raw=next((a for a in access if re.search(r'like this video along with',a,re.I)),'');mm=re.search(r'along with ([\d,]+) other',raw);z['likes']=exact(mm.group(1)) if mm else None;z['likes_display']=raw or None
  comments=first(j,'commentsHeaderRenderer');z['comments']=count_from_text(text(comments.get('countText',{})))
  pub=re.search(r'<meta\s+itemprop="datePublished"\s+content="([^"]+)"',r.text);z['platform_publish_date']=pub.group(1) if pub else player.get('microformat',{}).get('playerMicroformatRenderer',{}).get('publishDate')
  z['playability_status']=player.get('playabilityStatus',{}).get('status');z['playability_reason']=player.get('playabilityStatus',{}).get('reason');z['status']='ok' if z['views'] is not None else 'no_exact_view_count';z['evidence']={'views':{'originalViewCount':vc.get('originalViewCount'),'display':text(vc.get('viewCount',{}))},'likes_accessibility':raw,'comments_count_text':text(comments.get('countText',{})),'channel':{'id':z['channel_id'],'name':z['channel_name'],'subscribers_display':z['subscriber_count_display']},'paths':{'views':'ytInitialData.videoPrimaryInfoRenderer.viewCount.videoViewCountRenderer.originalViewCount','likes':'videoPrimaryInfoRenderer.likeButtonViewModel.accessibilityText','comments':'commentsHeaderRenderer.countText','channel':'videoOwnerRenderer'}}
 except requests.RequestException as e:z['status']='request_failed';z['error']=type(e).__name__+': '+str(e)[:180]
 target.write_text(json.dumps(z,ensure_ascii=False,indent=2));return z
if __name__=='__main__':
 df=pd.read_csv(ROOT/'data/processed/videos.csv').fillna('');rows=df.to_dict('records');allrows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for i,z in enumerate(pool.map(collect,rows)):
   allrows.append(z)
   if (i+1)%20==0:print(f'{i+1}/{len(rows)} processed; exact views {sum(v["views"] is not None for v in allrows)}',flush=True)
 pd.DataFrame([{k:v for k,v in z.items() if k!='evidence'} for z in allrows]).to_csv(OUT/'snapshot.csv',index=False)
 summary={'attempted':len(allrows),'views':sum(z['views'] is not None for z in allrows),'likes':sum(z['likes'] is not None for z in allrows),'comments':sum(z['comments'] is not None for z in allrows),'channel_ids':sum(z['channel_id'] is not None for z in allrows),'started_utc':min(z['retrieved_at_utc'] for z in allrows),'ended_utc':max(z['retrieved_at_utc'] for z in allrows),'policy':'Exact public counts only. Missing/hidden counts remain null; approximate display subscriber counts are not exact labels. No login or access-control bypass. Archived values remain separately preserved.'};(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
