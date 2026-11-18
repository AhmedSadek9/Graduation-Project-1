from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

image = Image.new("RGB", (1400, 900), "white")
draw = ImageDraw.Draw(image)

font_path = r"C:\Windows\Fonts\arial.ttf"
font = ImageFont.truetype(font_path, 38)
title_font = ImageFont.truetype(font_path, 48)

def arabic(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

draw.text((70, 40), "Basmala OCR Test Document",
          fill="navy", font=title_font)

draw.text((70, 150), "Medicine Name: Paracetamol",
          fill="black", font=font)

draw.text((70, 220), "Strength: 500 mg",
          fill="black", font=font)

draw.text((70, 290), "Language: English and Arabic",
          fill="black", font=font)

draw.text((1330, 400), arabic("\u0627\u0633\u0645 \u0627\u0644\u062f\u0648\u0627\u0621: \u0628\u0627\u0631\u0627\u0633\u064a\u062a\u0627\u0645\u0648\u0644"),
          fill="black", font=font, anchor="ra")

draw.text((1330, 480), arabic("\u0627\u0644\u062a\u0631\u0643\u064a\u0632: \u0665\u0660\u0660 \u0645\u062c\u0645"),
          fill="black", font=font, anchor="ra")

draw.text((1330, 560), arabic("\u0627\u062e\u062a\u0628\u0627\u0631 \u0627\u0633\u062a\u062e\u0631\u0627\u062c \u0627\u0644\u0646\u0635\u0648\u0635 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0648\u0627\u0644\u0625\u0646\u062c\u0644\u064a\u0632\u064a\u0629"),
          fill="black", font=font, anchor="ra")

draw.text((70, 680), "OCR Project - Basmala",
          fill="darkgreen", font=font)

image.save("data/raw_images/medicine_sample.png")

print("Test image created successfully!")
