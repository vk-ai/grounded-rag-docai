import pytest
from grag.extract import tesseract_available, ocr_image, degrade
from grag.text import error_rate

@pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed")
def test_tesseract_reads_a_degraded_rendered_page(tmp_path):
    PIL = pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageFont
    text = "Retrieval augmented generation combines parametric and non parametric memory"
    im = Image.new("L", (1800, 200), 255)
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", 40)
    except OSError:
        f = ImageFont.load_default(size=40)
    ImageDraw.Draw(im).text((20, 70), text, fill=0, font=f)
    im.save(tmp_path / "a.png"); degrade(tmp_path / "a.png", tmp_path / "b.png", seed=3)
    assert error_rate(text, ocr_image(tmp_path / "b.png", psm=6)) < 0.3
