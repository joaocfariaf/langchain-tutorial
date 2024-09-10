import os, easyocr, fitz

DOCS_FOLDER = os.path.join('rag_manager', 'documents')
PDF_FOLDER = os.path.join(DOCS_FOLDER, 'pdf')
TXT_FOLDER = os.path.join(PDF_FOLDER, 'txt')

class pdfTextExtractor():
    def __init__(
        self,
        pdf_folder = PDF_FOLDER,
        text_folder = DOCS_FOLDER,
    ) -> None:
        self.pdf_folder = pdf_folder
        self.text_folder = text_folder
        self.content = []

    def extract_data(self):
        data_txt = os.path.join(self.text_folder, 'data.txt')
        for pdf_archive in os.listdir(self.pdf_folder):
            archive_path = os.path.join(self.pdf_folder, pdf_archive)
            img_path = os.path.join(self.text_folder, pdf_archive.split('.pdf')[0])
            try:
                os.mkdir(img_path)
            except:
                pass
            pdf_text = self.extract_text_from_pdf(archive_path, img_path)
            with open(data_txt, 'a+') as file:
                file.write(f'\n<{pdf_archive}>\n')
                for text in pdf_text:
                    file.write(f'\n{text}')
                file.write(f'\n<\\{pdf_archive}>\n')
        pass

    def extract_text_from_pdf(self, archive_path: str, img_path:str):
        doc = fitz.open(archive_path)
        zoom = 4
        mat = fitz.Matrix(zoom, zoom)
        count = 0
        for page in doc:
            count += 1        
        pdf_text = []
        reader = easyocr.Reader(['pt'])
        for i in range(count):
            page_img = os.path.join(img_path, f'image_{i+1}.png')
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat)
            pix.save(page_img)
            try:
                pdf_text.extend(reader.readtext(page_img, detail=0))
            except:
                pass
        doc.close()

        return pdf_text



    def store(self):
        pass
    
if __name__ == '__main__':
    pdfTextExtractor().extract_data()