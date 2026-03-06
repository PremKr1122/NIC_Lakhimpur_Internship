from pdf2image import convert_from_path

pdf_path = r"D:\Internship Project\images.pdf"
poppler_path = r"C:\poppler-24.08.0\Library\bin"  # update this

images = convert_from_path(pdf_path, poppler_path=poppler_path)

for i, img in enumerate(images):
    img.save(f"page_{i+1}.jpg", "JPEG")
