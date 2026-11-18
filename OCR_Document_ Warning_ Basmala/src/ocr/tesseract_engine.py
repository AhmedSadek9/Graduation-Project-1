"""Configurable Tesseract adapter. Confidence is mean valid word confidence (0..100)."""
import os, time
from pathlib import Path
from .standard_output import OCRResult

class TesseractEngine:
    def __init__(self, lang=None, executable=None, config="--oem 3 --psm 6"):
        self.lang=lang or os.getenv("TESSERACT_LANG","ara+eng")
        self.executable=executable or os.getenv("TESSERACT_CMD")
        self.config=config

    def extract_text(self,image_path):
        start=time.perf_counter(); path=Path(image_path)
        if not path.is_file():
            return OCRResult(status="error",engine="tesseract",processing_time=time.perf_counter()-start,error=f"Image not found: {path}").to_dict()
        try:
            import pytesseract, cv2
            from pytesseract import Output
            if self.executable: pytesseract.pytesseract.tesseract_cmd=self.executable
            image=cv2.imread(str(path))
            if image is None:
                return OCRResult(status="error",engine="tesseract",processing_time=time.perf_counter()-start,error="Unreadable/corrupt image").to_dict()
            text=pytesseract.image_to_string(image,lang=self.lang,config=self.config).strip()
            data=pytesseract.image_to_data(image,lang=self.lang,config=self.config,output_type=Output.DICT)
            vals=[]
            for x in data.get("conf",[]):
                try:
                    v=float(x)
                    if v>=0: vals.append(v)
                except (ValueError,TypeError): pass
            conf=sum(vals)/len(vals) if vals else None
            return OCRResult(text,conf,"0-100" if conf is not None else None,
                "success" if text else "no_text","tesseract",time.perf_counter()-start).to_dict()
        except Exception as e:
            msg=f"{type(e).__name__}: {e}"
            if "not installed" in msg.lower() or "not found" in msg.lower(): msg="Tesseract executable unavailable or invalid path: "+msg
            return OCRResult(status="error",engine="tesseract",processing_time=time.perf_counter()-start,error=msg).to_dict()

def extract_text(image_path, lang=None, executable=None, config="--oem 3 --psm 6"):
    return TesseractEngine(lang,executable,config).extract_text(image_path)


