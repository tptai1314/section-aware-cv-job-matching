import pypdf

reader = pypdf.PdfReader('paper/main.pdf')
print(f"Total pages: {len(reader.pages)}")
print("\n=== Page 15 ===")
print(reader.pages[14].extract_text()[:600])
print("\n=== Page 16 ===")
print(reader.pages[15].extract_text()[:600])
if len(reader.pages) > 16:
    print("\n=== Page 17 ===")
    print(reader.pages[16].extract_text())
