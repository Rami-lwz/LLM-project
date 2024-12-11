import fitz  # PyMuPDF
import json
import os
from PIL import Image
import io
import sys
import re
import logging
from pix2tex.cli import LatexOCR  # Correct import from pix2tex
import pytesseract  # Import pytesseract for fallback OCR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class OCR:
    def __init__(self, arguments=None):
        """
        Initialize the OCR class with LatexOCR and pytesseract for fallback.

        Args:
            arguments: Configuration arguments for LatexOCR.
        """
        # Initialize LatexOCR
        self.latex_ocr = LatexOCR(arguments)
        logging.info("Initialized OCR with LatexOCR.")

        # Configure pytesseract
        # If Tesseract is not in your PATH, uncomment and set the following line
        # Example for Windows:
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        logging.info("Initialized pytesseract for fallback OCR.")

    def compute_confidence(self, latex_code: str) -> float:
        """
        Compute a pseudo-confidence score based on validation checks.

        Args:
            latex_code (str): The cleaned LaTeX code.

        Returns:
            float: Confidence score between 0.0 and 1.0.
        """
        confidence = 1.0  # Start with full confidence

        # 1. Check for minimal length
        if len(latex_code) < 5:
            logging.debug("Confidence Adjustment: LaTeX code is too short.")
            confidence = 0.0

        # 2. Check for balanced braces, brackets, and parentheses
        stack = []
        brackets = {'{': '}', '[': ']', '(': ')'}
        balanced = True
        for char in latex_code:
            if char in brackets:
                stack.append(brackets[char])
            elif char in brackets.values():
                if not stack or char != stack.pop():
                    balanced = False
                    logging.debug(f"Confidence Adjustment: Unbalanced {char}")
                    break
        if not balanced or stack:
            logging.debug("Confidence Adjustment: Unbalanced braces/brackets/parentheses.")
            confidence -= 0.2

        # 3. Check for balanced LaTeX environments
        environments = re.findall(r'\\begin\{(\w+)\}', latex_code)
        for env in environments:
            if not re.search(rf'\\end\{{{env}}}', latex_code):
                logging.debug(f"Confidence Adjustment: Unbalanced environment \\begin{{{env}}}.")
                confidence -= 0.2
                break

        # 4. Check for known commands
        valid_commands = re.findall(r'\\[a-zA-Z]+', latex_code)
        known_commands = {
            'frac', 'sqrt', 'sum', 'prod', 'int', 'lim', 'cdots', 'ldots', 'dots', 'to',
            'infty', 'mathbb', 'mathcal', 'mathrm', 'mathbf', 'overline', 'underline',
            'hat', 'tilde', 'dagger', 'left', 'right', 'begin', 'end',
            'array', 'overbrace', 'underbrace', 'nabla', 'partial', 'operatorname',
            'mathfrak', 'chi', 'cdot', 'vdots', 'bullet', 'bigcap', 'dotsm', 'sim',
            'circ', 'backslash', 'vdots',
            'mathring', 'vec', 'dot', 'ddot', 'dddot', 'tilde', 'acute', 'grave',
            'hat', 'check', 'breve', 'bar', 'breve', 'dot', 'grave', 'acute',
            # Add more as needed
        }
        unknown_commands = [cmd for cmd in valid_commands if cmd.lstrip('\\') not in known_commands]
        if unknown_commands:
            logging.debug(f"Confidence Adjustment: Unknown LaTeX commands detected: {unknown_commands}")
            confidence -= 0.3

        # 5. Check for excessive characters
        if latex_code.count('\\') > 50 or latex_code.count('{') > 50 or latex_code.count('}') > 50:
            logging.debug("Confidence Adjustment: Excessive use of backslashes or braces.")
            confidence = 0.0

        # Additional heuristic adjustments
        # Increase confidence if certain mathematical terms are present
        if 'sum' in latex_code:
            logging.debug("Confidence Adjustment: 'sum' found in LaTeX code.")
            confidence += 0.3
        if 'pi' in latex_code:
            logging.debug("Confidence Adjustment: 'pi' found in LaTeX code.")
            confidence += 0.3
        if 'gamma' in latex_code:
            logging.debug("Confidence Adjustment: 'gamma' found in LaTeX code.")
            confidence += 0.3
        if 'array' in latex_code:
            confidence -= 0.3
        # if \begin{array} in latex_code more than 1 time, confidence -= 0.3:
        if latex_code.count(r'\begin{array}') > 1:
            confidence = 0.0

        # 6. Ensure confidence is within [0.0, 1.0]
        confidence = max(0.0, min(confidence, 1.0))

        logging.debug(f"Final Confidence Score: {confidence:.2f}")
        return confidence

    def clean_latex(self, latex_code: str) -> str:
        original_code = latex_code  # Keep for debugging if needed

        # 1. Remove known OCR artifacts
        artifacts = [
            r'\[object Object\]',
            r'\[\[object Object\]\]',
            r'\{\{\{',
            r'\}\}\}',
            r'\[object Object\]',
            r'\[[object Object]\]',
        ]
        for artifact in artifacts:
            if re.search(artifact, latex_code):
                logging.debug(f"Cleaning: Removing artifact {artifact}")
            latex_code = re.sub(artifact, '', latex_code)

        # 2. Replace incorrect ellipsis representations with standard LaTeX commands
        if re.search(r'(?<!\\)\.\.\.', latex_code):
            logging.debug("Cleaning: Replacing incorrect ellipsis with '\\cdots'")
        latex_code = re.sub(r'(?<!\\)\.\.\.', r'\\cdots', latex_code)

        # 3. Fix unbalanced braces by removing extra ones
        brace_diff = latex_code.count('{') - latex_code.count('}')
        if brace_diff > 0:
            logging.debug(f"Cleaning: Removing {brace_diff} extra '{{' braces from the end.")
            latex_code = latex_code.rstrip('{')
        elif brace_diff < 0:
            logging.debug(f"Cleaning: Removing {-brace_diff} extra '}}' braces from the end.")
            latex_code = latex_code.rstrip('}')

        # 4. Standardize LaTeX commands (e.g., replace common misrecognitions)
        command_corrections = {
            r'\mathfrak{v}}^{\chi^{\prime}{}^{\chi}}': r'\mathfrak{v}^{\chi^{\prime\chi}}',
            # Add more corrections as identified from OCR outputs
        }
        for wrong, correct in command_corrections.items():
            if wrong in latex_code:
                logging.debug(f"Cleaning: Correcting '{wrong}' to '{correct}'")
            latex_code = latex_code.replace(wrong, correct)

        # 5. Remove or replace any remaining suspicious patterns
        # Example: Fix malformed \sqrt commands
        if re.search(r'\\sqrt\[\w+\]\{', latex_code):
            logging.debug("Cleaning: Fixing malformed '\\sqrt' command.")
        latex_code = re.sub(r'\\sqrt\[\w+\]\{', r'\\sqrt{', latex_code)

        # 6. Trim whitespace
        latex_code = latex_code.strip()

        # 7. Remove repeated patterns of two or more characters repeated three or more times
        pattern_repeat = r'(.{2,})\1{2,}'
        while re.search(pattern_repeat, latex_code):
            logging.debug("Cleaning: Removing repeated patterns of >=2 chars repeated 3+ times.")
            latex_code = re.sub(pattern_repeat, '', latex_code)

        # 8. Detect multiple [image] occurrences close to each other (3+ times)
        # The following regex looks for the substring `[image]` followed by up to 10 characters,
        # and this entire segment repeated 3 or more times.
        # Adjust '.{0,10}' as needed to define "closeness".
        pattern_images = r'(?:\[image\].{0,10}){3,}'
        if re.search(pattern_images, latex_code):
            logging.debug("Cleaning: Detected multiple [image] occurrences close together. Removing them.")
        latex_code = re.sub(pattern_images, '', latex_code)
        latex_code = latex_code.encode('utf-8').decode('unicode_escape')

        return latex_code

    def perform_ocr(self, image: Image.Image, confidence_threshold=0.5):
        """
        Perform OCR on a single image and return the extracted LaTeX code or fallback text along with its confidence score.

        Args:
            image (PIL.Image): The image to process.
            confidence_threshold (float): Minimum confidence required to accept the LaTeX code.

        Returns:
            tuple: (Extracted text (str), confidence score (float))
        """
        try:
            # Perform LaTeX OCR using pix2tex
            raw_code = self.latex_ocr(image)
            logging.debug(f"Raw LaTeX Code: {raw_code}")
            cleaned_code = self.clean_latex(raw_code)
            logging.debug(f"Cleaned LaTeX Code: {cleaned_code}")
            
            # Compute confidence score
            confidence = self.compute_confidence(cleaned_code)
            logging.debug(f"Confidence Score: {confidence:.2f}")

            if confidence >= confidence_threshold:
                logging.info(f"OCR Success: Confidence = {confidence:.2f}")
                return cleaned_code, confidence
            else:
                logging.info(f"OCR Confidence too low: Confidence = {confidence:.2f}. Using pytesseract for fallback.")
                # Perform fallback OCR using pytesseract
                fallback_text = self.perform_pytesseract_ocr(image)
                fallback_confidence = 0.5  # Assign a default confidence for fallback
                return fallback_text, fallback_confidence
        except Exception as e:
            logging.error(f"Error performing OCR: {e}")
            # Attempt fallback OCR in case of exception
            fallback_text = self.perform_pytesseract_ocr(image)
            fallback_confidence = 0.0  # Assign a default low confidence for failure
            return fallback_text, fallback_confidence

    def perform_pytesseract_ocr(self, image: Image.Image) -> str:
        """
        Perform OCR using pytesseract as a fallback.

        Args:
            image (PIL.Image): The image to process.

        Returns:
            str: The extracted text from the image.
        """
        try:
            # Convert image to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Perform OCR using pytesseract
            ocr_text = pytesseract.image_to_string(image)
            logging.debug(f"pytesseract OCR Text: {ocr_text}")
            # Clean the extracted text
            ocr_text = ocr_text.strip()
            if not ocr_text:
                ocr_text = "[image]"
            return ocr_text
        except Exception as e:
            logging.error(f"Error performing pytesseract OCR: {e}")
            return "[image]"
class BoringPDFParser:
    def __init__(self, ocr_instance: OCR):
        """
        Initialize the PDFParser with an OCR instance.

        Args:
            ocr_instance (OCR): An instance of the OCR class.
        """
        self.ocr = ocr_instance
        logging.info("Initialized PDFParser with OCR instance.")

    def boring_extract_text_from_pdf(self, pdf_path):
        """
        Extract text from the PDF without using image-based OCR unless no text is found.
        If no text is detected (PDF likely contains only images), then extract images and
        run pytesseract OCR on them.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: The extracted text.
        """
        # Extract text directly from the PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # If we got no text, assume the PDF is image-based
        if not text.strip():
            logging.info("No text found in PDF. Attempting OCR on images.")
            images = self.extract_images_from_pdf(pdf_path)
            if not images:
                logging.info("No images found in PDF. Returning empty string.")
                return ""
            
            ocr_texts = []
            # Perform pytesseract OCR on each image
            for _, _, image, _ in images:
                extracted_text = self.ocr.perform_pytesseract_ocr(image)
                if extracted_text.strip():
                    ocr_texts.append(extracted_text)
            
            # Combine OCR-extracted texts
            return "\n".join(ocr_texts)
        
        # If some text was found, return it as is
        return text

    def boring_parse_pdf(self, pdf_path):
        """
        Boring parse method: Just extract text using boring_extract_text_from_pdf.
        If no text is found, do a pytesseract OCR on images.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: The fully extracted text (no fancy OCR or LaTeX parsing).
        """
        return self.boring_extract_text_from_pdf(pdf_path)

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts plain text from the PDF.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: The extracted plain text.
        """
        text_content = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    text_content.append(text)
            return "\n".join(text_content)
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""

    def extract_images_from_pdf(self, pdf_path):
        """
        Extracts all images from the PDF.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[Tuple[int, int, Image.Image, tuple]]: A list of tuples containing page number, image index, PIL Image object, and bounding box.
        """
        images = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    logging.info(f"Processing page {page_num}/{len(doc)} for images...")
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list, start=1):
                        xref = img[0]
                        try:
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            bbox = img[1:5]  # Bounding box (x0, y0, x1, y1)
                            images.append((page_num, img_index, image, bbox))
                            logging.info(f"Extracted image {img_index} on page {page_num}.")
                        except Exception as e:
                            logging.error(f"Error extracting image xref {xref} on page {page_num}: {e}")
            return images
        except Exception as e:
            logging.error(f"Error opening PDF {pdf_path}: {e}")
            return images

    def boring_process_directory(self, directory_path, output_json):
        """
        Processes all PDF files in the specified directory using boring_parse_pdf and outputs extracted data to JSON.

        Args:
            directory_path (str): Path to the directory containing PDF files.
            output_json (str): Path to the output JSON file.

        Returns:
            None
        """
        extracted_data = {}
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if not filename.lower().endswith(".pdf"):
                logging.info(f"Skipping non-PDF file: {filename}")
                continue
            logging.info(f"\nProcessing {filename}...")
            try:
                text = self.boring_parse_pdf(file_path)
                extracted_data[filename] = text
                logging.info("Done.")
                # Display a sample of the extracted text (first 100 characters)
                sample = text[:100].replace('\n', ' ') + "..." if len(text) > 100 else text
                logging.info(f"Sample: {sample}\n")
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")

        # Save all extracted data to a JSON file
        try:
            with open(output_json, 'w', encoding='utf-8') as json_file:
                json.dump(extracted_data, json_file, ensure_ascii=False, indent=4)
            logging.info(f"All extracted data has been saved to {output_json}.")
        except Exception as e:
            logging.error(f"Error saving extracted data to JSON: {e}")

class PDFParser:
    def __init__(self, ocr_instance: OCR):
        """
        Initialize the PDFParser with an OCR instance.

        Args:
            ocr_instance (OCR): An instance of the OCR class.
        """
        self.ocr = ocr_instance
        logging.info("Initialized PDFParser with OCR instance.")

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts plain text from the PDF.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: The extracted plain text.
        """
        text_content = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    text_content.append(text)
            return "\n".join(text_content)
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""

    def extract_images_from_pdf(self, pdf_path):
        """
        Extracts all images from the PDF.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[Tuple[int, int, Image.Image, tuple]]: A list of tuples containing page number, image index, PIL Image object, and bounding box.
        """
        images = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    logging.info(f"Processing page {page_num}/{len(doc)} for images...")
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list, start=1):
                        xref = img[0]
                        try:
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            bbox = img[1:5]  # Bounding box (x0, y0, x1, y1)
                            images.append((page_num, img_index, image, bbox))
                            logging.info(f"Extracted image {img_index} on page {page_num}.")
                        except Exception as e:
                            logging.error(f"Error extracting image xref {xref} on page {page_num}: {e}")
            return images
        except Exception as e:
            logging.error(f"Error opening PDF {pdf_path}: {e}")
            return images

    def associate_images_with_text(self, images, texts):
        """
        Associate each image with the nearest text based on bounding box proximity.

        Args:
            images (List[Tuple[int, int, Image.Image, tuple]]): Extracted images with their bounding boxes.
            texts (List[Tuple[int, str, tuple]]): Extracted texts with their bounding boxes.

        Returns:
            dict: A dictionary mapping (page_num, img_index) to their associated text.
        """
        associations = {}
        for img in images:
            page_num_img, img_index, image, bbox_img = img
            min_distance = float('inf')
            associated_text = ""
            for txt in texts:
                page_num_txt, text, bbox_txt = txt
                if page_num_img != page_num_txt:
                    continue  # Only associate images with text on the same page
                # Compute vertical distance between image and text
                # Assuming text above the image is associated
                distance = bbox_img[1] - bbox_txt[3]  # y0_img - y1_text
                if distance < min_distance and distance >= 0:
                    min_distance = distance
                    associated_text = text
            associations[(page_num_img, img_index)] = associated_text
        return associations

    def parse_pdf(self, pdf_path):
        """
        Parses the PDF to extract text and LaTeX from math regions.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: The fully extracted text including LaTeX code from images.
        """
        try:
            extracted_text = []

            # Extract plain text
            logging.info("Extracting plain text from PDF...")
            text = self.extract_text_from_pdf(pdf_path)
            extracted_text.append(text)

            # Extract images
            logging.info("Extracting images from PDF...")
            images = self.extract_images_from_pdf(pdf_path)

            # Extract text with bounding boxes for association
            logging.info("Extracting text with bounding boxes for association...")
            texts = self.extract_text_from_pdf_with_bboxes(pdf_path)

            # Associate images with surrounding text
            associations = self.associate_images_with_text(images, texts)

            # Perform OCR on images using the OCR instance
            if images:
                logging.info(f"Performing OCR on {len(images)} extracted images...")
                for img in images:
                    page_num, img_index, image, bbox = img
                    logging.info(f"OCR on image {img_index} on page {page_num}...")
                    extracted_content, confidence = self.ocr.perform_ocr(image, confidence_threshold=0.5)
                    
                    # Determine if the extracted content is LaTeX or regular text
                    if confidence >= 0.5 and extracted_content.startswith("\\"):
                        # Assume LaTeX code
                        extracted_latex = f"$$ {extracted_content} $$"
                    elif extracted_content != "[image]":
                        # Use the fallback text from pytesseract
                        extracted_latex = extracted_content
                    else:
                        # Placeholder if no valid text extracted
                        extracted_latex = "[image]"

                    # Insert the extracted content into the text
                    # Optionally, you can insert it at a specific location based on the bounding box
                    # For simplicity, we'll append it to the end of the extracted text
                    extracted_text.append(f"\n\n{extracted_latex}\n")

                    logging.info(f"Extracted Content from image {img_index} on page {page_num}: {extracted_latex}")
            else:
                logging.info("No images found in PDF.")

            # Combine all extracted text
            full_text = "\n".join(extracted_text)
            return full_text
        except Exception as e:
            raise ValueError(f"Error reading {pdf_path}: {e}")
     
    def extract_text_from_pdf_with_bboxes(self, pdf_path):
        """
        Extracts all text from a PDF file along with their bounding boxes.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[Tuple[int, str, tuple]]: A list of tuples containing page number, text, and bounding box.
        """
        texts = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text_instances = page.get_text("dict")["blocks"]
                    for block in text_instances:
                        if block['type'] == 0:  # Type 0 is text
                            for line in block.get("lines", []):
                                for span in line.get("spans", []):
                                    text = span["text"]
                                    bbox = span["bbox"]  # (x0, y0, x1, y1)
                                    texts.append((page_num, text, bbox))
            return texts
        except Exception as e:
            logging.error(f"Error extracting text with bounding boxes from PDF {pdf_path}: {e}")
            return texts

    def process_directory(self, directory_path, output_json):
        """
        Processes all PDF files in the specified directory and outputs extracted data to JSON.

        Args:
            directory_path (str): Path to the directory containing PDF files.
            output_json (str): Path to the output JSON file.

        Returns:
            None
        """
        extracted_data = {}
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if not filename.lower().endswith(".pdf"):
                logging.info(f"Skipping non-PDF file: {filename}")
                continue
            logging.info(f"\nProcessing {filename}...")
            try:
                text = self.parse_pdf(file_path)
                extracted_data[filename] = text
                logging.info("Done.")
                # Display a sample of the extracted text (first 100 characters)
                sample = text[:100].replace('\n', ' ') + "..." if len(text) > 100 else text
                logging.info(f"Sample: {sample}\n")
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")

        # Save all extracted data to a JSON file
        try:
            with open(output_json, 'w', encoding='utf-8') as json_file:
                json.dump(extracted_data, json_file, ensure_ascii=False, indent=4)
            logging.info(f"All extracted data has been saved to {output_json}.")
        except Exception as e:
            logging.error(f"Error saving extracted data to JSON: {e}")

# if __name__ == "__main__":
#     # Specify the path to the Tesseract executable if not in PATH (required for pytesseract)
#     # Example for Windows:
#     # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#     # Instantiate the OCR class
#     ocr_instance = OCR()

#     # Instantiate the PDFParser with the OCR instance
#     processor = PDFParser(ocr_instance)

#     # Define input and output paths
#     input_directory = './pdfs'  # Ensure this directory exists and contains PDF files
#     output_file = 'extracted_texts.json'

#     # Process the directory
#     processor.process_directory(input_directory, output_file)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text and LaTeX from PDFs.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing PDF files.")
    parser.add_argument("--output", type=str, default="extracted_texts.json", help="Output JSON file.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for LaTeX OCR.")
    parser.add_argument("--boring", type=bool, default=False, help="Use the boring PDF parser.")
    args = parser.parse_args()

    # Instantiate classes and process
    ocr_instance = OCR()
    if args.boring:
        print("boring")
        processor = BoringPDFParser(ocr_instance)
        processor.boring_process_directory(args.input, args.output)
    else:
        processor = PDFParser(ocr_instance)
        processor.process_directory(args.input, args.output)
        
    

