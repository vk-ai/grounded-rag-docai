"""Text extraction: pdftotext for born-digital pages, tesseract OCR for scanned pages.

Scanned pages here are *simulated*: the PDF page is rasterised (pdftoppm, 200 dpi, gray), then
degraded deterministically (small rotation, blur, Gaussian noise, JPEG artefacts) before OCR."""
import random, shutil, subprocess, tempfile
from pathlib import Path

def pdf_pages(pdf):
    out = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    pages = out.split("\f")
    return pages[:-1] if pages and not pages[-1].strip() else pages

def degrade(png_in, png_out, seed):
    from PIL import Image, ImageFilter
    rng = random.Random(seed)
    im = Image.open(png_in).convert("L")
    im = im.rotate(rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=True, fillcolor=255)
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.1)))
    px = im.load(); w, h = im.size; r2 = random.Random(seed + 1)
    for _ in range(w * h // 40):  # speckle noise on 2.5% of pixels
        x, y = r2.randrange(w), r2.randrange(h)
        px[x, y] = max(0, min(255, px[x, y] + int(r2.gauss(0, 90))))
    tmp = Path(png_out).with_suffix(".jpg")
    im.save(tmp, quality=35); Image.open(tmp).save(png_out); tmp.unlink()

def tesseract_available():
    return shutil.which("tesseract") is not None

def ocr_image(png, psm=3):
    return subprocess.run(["tesseract", str(png), "-", "--psm", str(psm), "-l", "eng"],
                          capture_output=True, text=True, check=True).stdout

def scan_and_ocr(pdf, page, seed, keep_dir=None):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "p"
        subprocess.run(["pdftoppm", "-r", "200", "-gray", "-png", "-f", str(page), "-l", str(page),
                        "-singlefile", str(pdf), str(base)], check=True)
        out = Path(keep_dir or d) / f"{Path(pdf).stem}_p{page}_scan.png"
        degrade(f"{base}.png", out, seed)
        return ocr_image(out)
