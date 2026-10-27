import re, sys
from pypdf import PdfReader

path = "data/raw/AC_43.13-1B.pdf"
reader = PdfReader(path)

def compact(t):
    return re.sub(r"[\s\-–]+", "", t.lower())

needle = compact(sys.argv[1])
print(f"Pages containing '{sys.argv[1]}' (whitespace and hyphens ignored):")
for n, page in enumerate(reader.pages, start=1):
    c = compact(page.extract_text() or "")
    pos = c.find(needle)
    if pos != -1:
        print(f"  p.{n}: ...{c[max(0, pos-60):pos+120]}...")