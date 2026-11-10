"""Export only actual visible user/assistant messages from this session's local log.
Never exports system/developer instructions, reasoning, tool payloads or credentials.
"""
import os,json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
thread=os.environ.get('CODEX_THREAD_ID','01a1c017-9000-7ac0-a279-50e1e59cfcc3')
paths=list((Path.home()/'.codex/sessions').glob(f'**/*{thread}.jsonl'))
if len(paths)!=1:raise RuntimeError('Cannot identify exactly one current session log.')
entries=[]
for line in paths[0].open():
 x=json.loads(line);p=x.get('payload',{})
 if x.get('type')=='event_msg' and p.get('type') in ['user_message','agent_message']:
  entries.append((x['timestamp'],'User' if p['type']=='user_message' else 'Assistant',p['message']))
if not entries:raise RuntimeError('No visible session messages found.')
out=['# STAT6207 Group 6 - Actual session transcript','',
'This is an export of the actual visible user and assistant text messages in this project session. Messages are reproduced as logged, including intermediate statements later corrected. It excludes internal reasoning, system/developer instructions, tool payloads, credentials, and UI approval interactions. It is not a fabricated conversation or the work-log summary.',
'',f'Export cutoff: {entries[-1][0]}. Time labels below use Asia/Hong_Kong. The final handoff after this export is not included.','']
for timestamp,role,message in entries:
 t=datetime.fromisoformat(timestamp.replace('Z','+00:00')).astimezone(ZoneInfo('Asia/Hong_Kong'))
 out.extend([f'## {role} - {t:%Y-%m-%d %H:%M:%S %Z}','',message.rstrip(),''])
Path('output/session_transcript.md').write_text('\n'.join(out),encoding='utf8');print(f'Exported {len(entries)} actual visible messages to output/session_transcript.md')
