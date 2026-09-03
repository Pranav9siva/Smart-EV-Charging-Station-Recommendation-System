import os
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
paths = [root / 'docs' / 'Documentation' / 'Task_tracking_Doc.docx', root / 'Documentation' / 'Task_tracking_Doc.docx']
for p in paths:
    print('PATH:', p)
    print('EXISTS:', p.exists())
    if p.exists():
        try:
            with zipfile.ZipFile(p, 'r') as z:
                xml = z.read('word/document.xml').decode('utf-8')
                import re
                text = re.sub(r'<.*?>', '', xml)
                print('--- BEGIN TEXT ---')
                print('\n'.join(text.splitlines()[:120]))
                print('--- END TEXT ---')
        except Exception as exc:
            print('ERROR READING DOCX:', exc)

print('SUMO PATH checks:')
for p in [r'C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe', r'C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe', r'C:\Program Files\Eclipse\Sumo\bin\sumo.exe', r'C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe']:
    print(p, os.path.exists(p))
