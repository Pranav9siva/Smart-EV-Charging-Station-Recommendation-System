from pathlib import Path
p = Path('disk_report.txt')
out = Path('disk_report_utf8.txt')
if not p.exists():
    print('missing', p)
else:
    data = p.read_bytes()
    # try utf-16-le then utf-8
    for enc in ('utf-16','utf-16-le','utf-8','latin-1'):
        try:
            text = data.decode(enc)
            out.write_text(text, encoding='utf-8')
            print('written', out)
            break
        except Exception as e:
            #print('fail',enc,e)
            pass
