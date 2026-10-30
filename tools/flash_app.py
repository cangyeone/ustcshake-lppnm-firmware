"""Build and upload app0 only; the caller must supply a recovery image and MAC."""
import argparse,hashlib,json,pathlib,re,subprocess,sys
root=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--port',required=True)
p.add_argument('--expected-mac',required=True)
p.add_argument('--backup',type=pathlib.Path,required=True)
a=p.parse_args()
if a.backup.stat().st_size!=0x400000: p.error('Expected a full 4 MB recovery image')
expected=a.expected_mac.lower()
if not re.fullmatch(r'(?:[0-9a-f]{2}:){5}[0-9a-f]{2}',expected):p.error('Invalid MAC format')
subprocess.run([sys.executable,'-m','platformio','run','--project-dir',str(root)],check=True)
identity=subprocess.check_output([sys.executable,'-m','esptool','--chip','esp32c3','--port',a.port,'read-mac'],text=True)
if expected not in [x.lower() for x in re.findall(r'MAC:\s*([0-9a-fA-F:]{17})',identity)]:raise SystemExit('Connected device MAC does not match')
binary=root/'.pio/build/esp32c3/firmware.bin'
subprocess.run([sys.executable,'-m','esptool','--chip','esp32c3','--port',a.port,'--baud','460800','write-flash','0x10000',str(binary)],check=True)
report={'port':a.port,'mac':expected,'app_offset':'0x10000','app_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'backup_sha256':hashlib.sha256(a.backup.read_bytes()).hexdigest()}
(root/'local_reports').mkdir(exist_ok=True)
(root/'local_reports/deployment.json').write_text(json.dumps(report,indent=2))
