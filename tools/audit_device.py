"""Read-only sampling audit plus explicit paper dry-run (no raster sent to UART)."""
import argparse,json,pathlib,time,serial
root=pathlib.Path(__file__).resolve().parents[1]
(root/'local_reports').mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--port',required=True);p.add_argument('--seconds',type=float,default=120);a=p.parse_args()
s=serial.Serial();s.port=a.port;s.baudrate=115200;s.timeout=.05;s.dtr=False;s.rts=False;s.open()
start=time.monotonic();next_status=0;dry=False;stream=False;last_seq=None;gaps=0;total=0;statuses=[];inference=[];printer=[];bad=[];sent_time=False;pending_line=""
with (root/'local_reports/device_stress.jsonl').open('w') as log:
 while time.monotonic()-start<a.seconds:
  elapsed=time.monotonic()-start
  if elapsed>=next_status:
   s.write(b'status\n');next_status=elapsed+5
  if elapsed>=5 and not sent_time:
   s.write(f'time {int(time.time()*1000)}\n'.encode());sent_time=True
  if elapsed>=38 and not dry:
   s.write(b'dryrun 60\n');dry=True;print('Started 60-second simulated printer load (no paper output)',flush=True)
  if elapsed>=40 and not stream:
   s.write(b'stream on\n');stream=True
  chunk=s.readline().decode(errors='replace')
  if not chunk: continue
  pending_line+=chunk
  if not pending_line.endswith('\n'): continue
  line=pending_line.strip();pending_line=''
  if not line: continue
  log.write(line+'\n');log.flush()
  try: obj=json.loads(line)
  except ValueError:
   if any(x in line for x in ['Guru Meditation','assert failed','panic','Backtrace','ADC CONFIG ERROR']): bad.append(line)
   continue
  if obj.get('type')=='status':
   statuses.append(obj)
   keys=['acquired','missed_adc','invalid_adc','nn_dropped','nn_resets','inference_max_us','paper_overflow','printed_rows','printing','picks','heap_min']
   print(round(elapsed,1),{k:obj.get(k) for k in keys},flush=True)
  elif obj.get('type')=='inference': inference.append(obj)
  elif obj.get('type')=='printer': printer.append(obj)
  elif obj.get('type')=='samples':
   for row in obj['data']:
    if last_seq is not None and row[0]!=last_seq+1: gaps+=max(1,row[0]-last_seq-1)
    last_seq=row[0];total+=1
 s.write(b'stream off\n');s.close()
last=statuses[-1] if statuses else {}
report={'duration_s':time.monotonic()-start,'status_count':len(statuses),'last_status':last,'last_inference':inference[-1] if inference else {},'last_printer':printer[-1] if printer else {},'usb_samples':total,'usb_sequence_gaps':gaps,'faults':bad,'physical_print_tested':False,'dryrun_requested':dry,'pass':bool(statuses) and not bad and not gaps and bool(printer) and printer[-1].get('labels_dropped',1)==0 and total>10000 and last.get('printed_rows',0)>=7490 and not last.get('printing',True) and all(last.get(k,1)==0 for k in ['missed_adc','invalid_adc','nn_dropped','nn_resets','paper_overflow']) and 0<last.get('inference_max_us',0)<1280000}
(root/'local_reports/device_stress_summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
raise SystemExit(0 if report['pass'] else 1)
