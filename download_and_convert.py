import urllib.request
import zipfile
import os
import subprocess

PANDOC_URL = "https://github.com/jgm/pandoc/releases/download/3.1.11/pandoc-3.1.11-windows-x86_64.zip"
ZIP_FILE = "pandoc.zip"
EXTRACT_DIR = "pandoc_bin"
PANDOC_EXE = os.path.join(EXTRACT_DIR, "pandoc-3.1.11", "pandoc.exe")

def download_and_extract():
    if not os.path.exists(PANDOC_EXE):
        print(f"Downloading pandoc from {PANDOC_URL}...")
        urllib.request.urlretrieve(PANDOC_URL, ZIP_FILE)
        print("Extracting pandoc...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    print("Pandoc is ready.")

def convert_files():
    print("Converting Markdown to DOCX...")
    subprocess.run([
        PANDOC_EXE, 
        "PhysioFM_Research_Dossier.md", 
        "-o", "PhysioFM_Research_Dossier.docx",
        "--toc"
    ], check=True)
    print("DOCX conversion successful.")

    print("Converting DOCX to PDF using MS Word via docx2pdf...")
    try:
        from docx2pdf import convert
        convert("PhysioFM_Research_Dossier.docx", "PhysioFM_Research_Dossier.pdf")
        print("PDF conversion successful.")
    except Exception as e:
        print(f"PDF conversion failed: {e}")
        # fallback to using fpdf or markdown-pdf if needed, but if MS Word isn't present, docx2pdf will throw an error.

if __name__ == "__main__":
    download_and_extract()
    convert_files()
