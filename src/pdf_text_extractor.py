import os, easyocr, fitz

RAG_DOCS = os.path.join('rag-documents')
PDF_DIR = os.path.join(RAG_DOCS, 'pdf')
SCRIPT_GENERATED = os.path.join(RAG_DOCS, 'script-generated')
IMG_DIR = os.path.join(SCRIPT_GENERATED, 'img')
DATA_FILE = os.path.join(SCRIPT_GENERATED, 'data.txt')

class pdfTextExtractor():
    def __init__(
        self,
        pdf_dir = PDF_DIR,
        img_dir = IMG_DIR,
        data_file = DATA_FILE,
    ) -> None:
        self.pdf_dir = pdf_dir
        self.img_dir = img_dir
        self.data_file = data_file
        self.content = []

    def extract_data(self):
        
        with open(self.data_file, 'w+') as file:
            pass
        
        for pdf_name in os.listdir(self.pdf_dir):
            pdf_path = os.path.join(self.pdf_dir, pdf_name)
            img_subdir = os.path.join(self.img_dir, pdf_name.split('.pdf')[0])
            try:
                os.mkdir(img_subdir)
            except:
                pass
            pdf_text = self.extract_text_from_pdf(pdf_path, img_subdir)
            # Writes pdf_text
            pdf_name = pdf_name.encode('utf-8')
            with open(self.data_file, 'a+') as file:
                file.write(f'\n<{pdf_name}>\n')
                for text in pdf_text:
                    text = text.encode('utf-8')
                    file.write(f'\n{text}')
                file.write(f'\n<\\{pdf_name}>\n')
        pass

    def extract_text_from_pdf(
        self, 
        pdf_path: str, 
        img_subdir: str, 
        zoom: float = 4,
    ):
        doc = fitz.open(pdf_path)
        base_matrix = fitz.Matrix(zoom, zoom)
        n_pages = 0
        for page in doc:
            n_pages += 1        
        pdf_text = []
        
        image_reader = easyocr.Reader(['pt'])
        for i in range(n_pages):
            page_img = os.path.join(img_subdir, f'image_{i+1}.png')
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=base_matrix)
            pix.save(page_img)
            try:
                pdf_text.extend(image_reader.readtext(page_img, detail=0))
            except:
                pass
        doc.close()

        return pdf_text

    
if __name__ == '__main__':
    pdfTextExtractor().extract_data()