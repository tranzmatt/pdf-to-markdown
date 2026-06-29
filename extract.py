import fitz  # PyMuPDF
import pdfplumber
import re
import yaml
import pytesseract
import cv2
import io
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import logging
import traceback
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
import argparse

warnings.filterwarnings("ignore")

_CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


class PDFExtractor(ABC):
    """Abstract base class for PDF extraction."""

    def __init__(self, pdf_path, output_dir=None):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir) if output_dir else Path(pdf_path).parent
        self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.output_dir / f"{Path(self.pdf_path).stem}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def extract(self):
        """Abstract method for extracting content from PDF."""
        pass


class MarkdownPDFExtractor(PDFExtractor):
    """Class for extracting markdown-formatted content from PDF."""

    BULLET_POINTS = "•◦▪▫●○"

    def __init__(self, pdf_path, output_dir=None, skip_images=False):
        super().__init__(pdf_path, output_dir)
        self.pdf_filename = Path(pdf_path).stem
        self.skip_images = skip_images
        self.images_dir = self.output_dir / f"{self.pdf_filename}_images"
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        self.device = None

    def _ensure_captioning_model(self):
        """Lazy-load the image captioning model on first use."""
        if self.model is not None:
            return True
        try:
            self.model = VisionEncoderDecoderModel.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )
            self.feature_extractor = ViTImageProcessor.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.logger.info("Image captioning model set up successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error setting up image captioning model: {e}")
            self.logger.exception(traceback.format_exc())
            return False

    def extract(self):
        try:
            markdown_content, markdown_pages = self.extract_markdown()
            self.save_markdown(markdown_content)
            self.logger.info(
                f"Markdown content has been saved to {self.output_dir}/{self.pdf_filename}.md"
            )
            return markdown_content, markdown_pages

        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            self.logger.exception(traceback.format_exc())
            return "", []

    def extract_markdown(self):
        """Main method to extract markdown from PDF."""
        try:
            doc = fitz.open(self.pdf_path)
            markdown_content = ""
            markdown_pages = []
            tables = self.extract_tables()
            table_index = 0
            list_counter = 0
            in_code_block = False
            code_block_content = ""
            code_block_lang = None
            prev_line = ""

            for page_num, page in enumerate(doc):
                self.logger.info(f"Processing page {page_num + 1}")
                page_content = ""

                # Decide upfront whether this page has real extractable text.
                # A scanned page will have essentially no text from fitz; a text
                # PDF page will have plenty.  50 chars is conservative enough to
                # avoid false-positives from page-number/header-only pages.
                has_text = len(page.get_text().strip()) > 50

                if not has_text and page.get_images():
                    # Scanned page: render it and OCR for text content.
                    page_content = self._ocr_full_page(page)
                else:
                    blocks = page.get_text("dict")["blocks"]
                    page_height = page.rect.height
                    links = self.extract_links(page)

                    for block in blocks:
                        if block["type"] == 0:  # Text
                            page_content += self.process_text_block(
                                block,
                                page_height,
                                links,
                                list_counter,
                                in_code_block,
                                code_block_content,
                                code_block_lang,
                                prev_line,
                            )
                        elif block["type"] == 1 and not self.skip_images:
                            page_content += self.process_image_block(page, block)

                # Insert tables at their approximate positions
                while (
                    table_index < len(tables)
                    and tables[table_index]["page"] == page.number
                ):
                    page_content += (
                        "\n\n"
                        + self.table_to_markdown(tables[table_index]["content"])
                        + "\n\n"
                    )
                    table_index += 1

                markdown_pages.append(self.post_process_markdown(page_content))
                markdown_content += page_content + config["PAGE_DELIMITER"]

            markdown_content = self.post_process_markdown(markdown_content)
            return markdown_content, markdown_pages
        except Exception as e:
            self.logger.error(f"Error extracting markdown: {e}")
            self.logger.exception(traceback.format_exc())
            return "", []

    def extract_tables(self):
        """Extract tables from PDF using pdfplumber."""
        tables = []
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    if len(page_tables) > 128:
                        continue
                    for table in page_tables:
                        if self._is_data_table(table):
                            tables.append({"page": page_number, "content": table})
            self.logger.info(f"Extracted {len(tables)} tables from the PDF.")
        except Exception as e:
            self.logger.error(f"Error extracting tables: {e}")
            self.logger.exception(traceback.format_exc())
        return tables

    def table_to_markdown(self, table):
        """Convert a table to markdown format."""
        if not table:
            return ""

        try:
            table = [
                ["" if cell is None else str(cell).strip() for cell in row]
                for row in table
            ]
            col_widths = [max(len(cell) for cell in col) for col in zip(*table)]

            markdown = ""
            for i, row in enumerate(table):
                formatted_row = [
                    cell.ljust(col_widths[j]) for j, cell in enumerate(row)
                ]
                markdown += "| " + " | ".join(formatted_row) + " |\n"

                if i == 0:
                    markdown += (
                        "|"
                        + "|".join(["-" * (width + 2) for width in col_widths])
                        + "|\n"
                    )

            return markdown
        except Exception as e:
            self.logger.error(f"Error converting table to markdown: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def _ocr_full_page(self, page):
        """Render a scanned page at 2x and return its full OCR text as markdown."""
        try:
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            text = pytesseract.image_to_string(opencv_image)
            self.logger.info(f"OCR page {page.number + 1}: {len(text)} chars extracted")
            return text + "\n"
        except Exception as e:
            self.logger.error(f"Error OCR-ing page {page.number + 1}: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def _neural_caption(self, image):
        """Generate a caption using the VIT-GPT2 model. Returns empty string on failure."""
        try:
            if not self._ensure_captioning_model():
                return ""
            img = image.convert("RGB") if image.mode != "RGB" else image
            arr = np.array(img).transpose(2, 0, 1)
            inputs = self.feature_extractor(images=arr, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(inputs.pixel_values, max_length=30)
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            self.logger.error(f"Error generating neural caption: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def _merge_letterspaced_spans(self, spans):
        """
        Merge consecutive all-uppercase spans whose horizontal gap is smaller
        than a word space.  Fixes three related rendering artifacts:
          1. Tracked/letter-spaced bold headings: **E** **DITION** → **EDITION**
          2. Small-caps section titles: W HAT → WHAT
          3. Heading lines where every letter is its own span: H  OW → HOW

        Word space in most fonts is ~1/3 em; we use 0.35× the larger font size
        as the threshold.  Spans with a gap below that are within the same word.
        """
        if len(spans) <= 1:
            return spans

        result = []
        current = dict(spans[0])

        for nxt in spans[1:]:
            c_text = current["text"]
            n_text = nxt["text"]

            c_upper = bool(c_text.strip()) and c_text == c_text.upper()
            n_upper = bool(n_text.strip()) and n_text == n_text.upper()

            if c_upper and n_upper:
                gap = nxt["bbox"][0] - current["bbox"][2]
                word_space = max(current["size"], nxt["size"]) * 0.35
                if gap < word_space:
                    current = {
                        **current,
                        "text": c_text + n_text,
                        "bbox": (
                            current["bbox"][0],
                            min(current["bbox"][1], nxt["bbox"][1]),
                            nxt["bbox"][2],
                            max(current["bbox"][3], nxt["bbox"][3]),
                        ),
                    }
                    continue

            result.append(current)
            current = dict(nxt)

        result.append(current)
        return result

    def _is_data_table(self, table):
        """
        Return True only when the table looks like real tabular data.
        pdfplumber misidentifies multi-column body text as tables; those
        produce cells with hundreds of words.  Any cell exceeding 200 chars
        is a strong signal that this is a column layout, not a table.
        """
        for row in table:
            for cell in row:
                if cell and len(str(cell)) > 200:
                    return False
        return True

    def clean_text(self, text):
        """Clean the given text by removing extra spaces."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def apply_formatting(self, text, flags):
        """Apply markdown formatting to the given text based on flags."""
        text = text.strip()
        if not text:
            return text

        is_bold = flags & 2**4
        is_italic = flags & 2**1
        is_monospace = flags & 2**3
        is_superscript = flags & 2**0
        is_subscript = flags & 2**5

        if is_monospace:
            text = f"`{text}`"
        elif is_superscript and not bool(re.search(r"\s+", text)):
            text = f"^{text}^"
        elif is_subscript and not bool(re.search(r"\s+", text)):
            text = f"~{text}~"

        if is_bold and is_italic:
            text = f"***{text}***"
        elif is_bold:
            text = f"**{text}**"
        elif is_italic:
            text = f"*{text}*"

        return f" {text} "

    def is_bullet_point(self, text):
        """Check if the given text is a bullet point."""
        return text.strip().startswith(tuple(self.BULLET_POINTS))

    def convert_bullet_to_markdown(self, text):
        """Convert a bullet point to markdown format."""
        text = re.sub(r"^\s*", "", text)
        return re.sub(f"^[{re.escape(self.BULLET_POINTS)}]\s*", "- ", text)

    def is_numbered_list_item(self, text):
        """Check if the given text is a numbered list item."""
        return bool(re.match(r"^\d+\s{0,3}[.)]", text.strip()))

    def convert_numbered_list_to_markdown(self, text, list_counter):
        """Convert a numbered list item to markdown format."""
        text = re.sub(r"^\s*", "", text)
        return re.sub(r"^\d+\s{0,3}[.)]", f"{list_counter}. ", text)

    def is_horizontal_line(self, text):
        """Check if the given text represents a horizontal line."""
        return bool(re.match(r"^[_-]+$", text.strip()))

    def extract_links(self, page):
        """Extract links from the given page."""
        links = []
        try:
            for link in page.get_links():
                if link["kind"] == 2:  # URI link
                    links.append({"rect": link["from"], "uri": link["uri"]})
            self.logger.info(f"Extracted {len(links)} links from the page.")
        except Exception as e:
            self.logger.error(f"Error extracting links: {e}")
            self.logger.exception(traceback.format_exc())
        return links

    def detect_code_block(self, prev_line, current_line):
        """Detect if the current line starts a code block."""
        patterns = {
            "python": [
                (
                    r"^(?:from|import)\s+\w+",
                    r"^(?:from|import|def|class|if|for|while|try|except|with)\s",
                ),
                (r"^(?:def|class)\s+\w+", r"^\s{4}"),
                (r"^\s{4}", r"^\s{4,}"),
            ],
            "javascript": [
                (
                    r"^(?:function|const|let|var)\s+\w+",
                    r"^(?:function|const|let|var|if|for|while|try|catch|class)\s",
                ),
                (r"^(?:if|for|while)\s*\(", r"^\s{2,}"),
                (r"^\s{2,}", r"^\s{2,}"),
            ],
            "html": [
                (
                    r"^<(!DOCTYPE|html|head|body|div|p|a|script|style)",
                    r"^<(!DOCTYPE|html|head|body|div|p|a|script|style)",
                ),
                (r"^<\w+.*>$", r"^\s{2,}<"),
                (r"^\s{2,}<", r"^\s{2,}<"),
            ],
            "shell": [
                (r"^(?:\$|\#)\s", r"^(?:\$|\#)\s"),
                (r"^[a-z_]+\s*=", r"^[a-z_]+\s*="),
            ],
            "bash": [
                (
                    r"^(?:#!/bin/bash|alias|export|source)\s",
                    r"^(?:#!/bin/bash|alias|export|source|echo|read|if|for|while|case|function)\s",
                ),
                (r"^(?:if|for|while|case|function)\s", r"^\s{2,}"),
                (r"^\s{2,}", r"^\s{2,}"),
            ],
            "cpp": [
                (
                    r"^#include\s*<",
                    r"^(?:#include|using|namespace|class|struct|enum|template|typedef)\s",
                ),
                (r"^(?:class|struct|enum)\s+\w+", r"^\s{2,}"),
                (r"^\s{2,}", r"^\s{2,}"),
            ],
            "java": [
                (
                    r"^(?:import|package)\s+\w+",
                    r"^(?:import|package|public|private|protected|class|interface|enum)\s",
                ),
                (r"^(?:public|private|protected)\s+class\s+\w+", r"^\s{4,}"),
                (r"^\s{4,}", r"^\s{4,}"),
            ],
            "json": [
                (r"^\s*{", r'^\s*["{[]'),
                (r'^\s*"', r'^\s*["}],?$'),
                (r"^\s*\[", r"^\s*[}\]],?$"),
            ],
        }

        for lang, pattern_pairs in patterns.items():
            for prev_pattern, curr_pattern in pattern_pairs:
                if re.match(prev_pattern, prev_line.strip()) and re.match(
                    curr_pattern, current_line.strip()
                ):
                    return lang

        return None

    def process_text_block(
        self,
        block,
        page_height,
        links,
        list_counter,
        in_code_block,
        code_block_content,
        code_block_lang,
        prev_line,
    ):
        """Process a text block and convert it to markdown."""
        try:
            block_rect = block["bbox"]
            if block_rect[1] < 50 or block_rect[3] > page_height - 50:
                return ""  # Skip headers and footers

            block_text = ""
            last_y1 = None
            last_font_size = None

            for line in block["lines"]:
                line_text = ""
                merged_spans = self._merge_letterspaced_spans(line["spans"])
                curr_font_size = [span["size"] for span in merged_spans]

                for span in merged_spans:
                    text = span["text"]
                    font_size = span["size"]
                    flags = span["flags"]
                    span_rect = span["bbox"]

                    if self.is_horizontal_line(text):
                        line_text += "\n---\n"
                        continue

                    text = self.clean_text(text)

                    if text.strip():
                        header_level = self.get_header_level(font_size)
                        if header_level > 0:
                            text = f"\n{'#' * header_level} {text}\n\n"

                        else:
                            is_list_item = self.is_bullet_point(
                                text
                            ) or self.is_numbered_list_item(text)

                            if is_list_item:
                                marker, content = re.split(
                                    r"(?<=^[•◦▪▫●○\d.)])\s*", text, 1
                                )
                                formatted_content = self.apply_formatting(
                                    content, flags
                                )
                                text = f"{marker} {formatted_content}"
                            else:
                                text = self.apply_formatting(text, flags)

                    for link in links:
                        if fitz.Rect(span_rect).intersects(link["rect"]):
                            text = f"[{text.strip()}]({link['uri']})"
                            break

                    line_text += text

                if last_y1 is not None:
                    avg_last_font_size = (
                        sum(last_font_size) / len(last_font_size)
                        if last_font_size
                        else 0
                    )
                    avg_current_font_size = sum(curr_font_size) / len(curr_font_size)
                    font_size_changed = (
                        abs(avg_current_font_size - avg_last_font_size) > 1
                    )

                    if abs(line["bbox"][3] - last_y1) > 2 or font_size_changed:
                        block_text += "\n"

                block_text += self.clean_text(line_text) + " "
                last_font_size = curr_font_size
                last_y1 = line["bbox"][3]

            markdown_content = ""
            lines = block_text.split("\n")
            for i, line in enumerate(lines):
                clean_line = self.clean_text(line)

                if not in_code_block:
                    code_lang = self.detect_code_block(prev_line, clean_line)
                    if code_lang:
                        in_code_block = True
                        code_block_lang = code_lang
                        code_block_content = prev_line + "\n" + clean_line + "\n"
                        prev_line = clean_line
                        continue

                if in_code_block:
                    code_block_content += clean_line + "\n"
                    if (
                        i == len(lines) - 1
                        or self.detect_code_block(clean_line, lines[i + 1])
                        != code_block_lang
                    ):
                        markdown_content += (
                            f"```{code_block_lang}\n{code_block_content}```\n\n"
                        )
                        in_code_block = False
                        code_block_content = ""
                        code_block_lang = None
                else:
                    if self.is_bullet_point(clean_line):
                        markdown_content += "\n" + self.convert_bullet_to_markdown(
                            clean_line
                        )
                        list_counter = 0
                    elif self.is_numbered_list_item(clean_line):
                        list_counter += 1
                        markdown_content += (
                            "\n"
                            + self.convert_numbered_list_to_markdown(
                                clean_line, list_counter
                            )
                        )
                    else:
                        markdown_content += f"{clean_line}\n"
                        list_counter = 0

                prev_line = clean_line

            return markdown_content + "\n"
        except Exception as e:
            self.logger.error(f"Error processing text block: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def process_image_block(self, page, block):
        """Process an image block and convert it to markdown."""
        try:
            # Pull the image bytes that fitz already decoded from the PDF stream.
            # This avoids re-rendering the page region through a pixmap.
            image = None
            raw = block.get("image")
            if raw:
                try:
                    image = Image.open(io.BytesIO(raw))
                    if image.mode not in ("RGB", "RGBA", "L", "P"):
                        image = image.convert("RGB")
                except Exception:
                    image = None

            if image is None:
                # Fallback: render the page region (e.g. for vector graphics).
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(clip=fitz.Rect(block["bbox"]), matrix=mat, alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            if image.width < 20 or image.height < 20:
                return ""

            self.images_dir.mkdir(parents=True, exist_ok=True)
            image_filename = (
                f"{self.pdf_filename}_image_{int(page.number)+1}_{block['number']}.png"
            )
            image_path = self.images_dir / image_filename
            save_img = image.convert("RGB") if image.mode not in ("RGB", "L") else image
            save_img.save(image_path, "PNG", optimize=True)

            # For embedded images in a text PDF, the image is a figure/diagram —
            # OCR on it produces garbage.  Use the neural captioner instead, and
            # only fall back to OCR when the page itself was identified as a scan
            # (that path is handled by _ocr_full_page, not here).
            caption = self._neural_caption(image) or (
                f"{self.pdf_filename}_image_{int(page.number)+1}_{block['number']}"
            )

            rel_path = image_path.relative_to(self.output_dir)
            return f"![{caption}]({rel_path})\n\n"
        except Exception as e:
            self.logger.error(f"Error processing image block: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def get_header_level(self, font_size):
        """Determine header level based on font size."""
        if font_size > 24:
            return 1
        elif font_size > 20:
            return 2
        elif font_size > 18:
            return 3
        elif font_size > 16:
            return 4
        elif font_size > 14:
            return 5
        elif font_size > 12:
            return 6
        else:
            return 0

    def post_process_markdown(self, markdown_content):
        """Post-process the markdown content."""
        try:
            markdown_content = re.sub(
                r"\n{3,}", "\n\n", markdown_content
            )  # Remove excessive newlines
            markdown_content = re.sub(
                r"(\d+)\s*\n", "", markdown_content
            )  # Remove page numbers
            markdown_content = re.sub(
                r" +", " ", markdown_content
            )  # Remove multiple spaces
            markdown_content = re.sub(
                r"\s*(---\n)+", "\n\n---\n", markdown_content
            )  # Remove duplicate horizontal lines

            # Safety net for bold letter-spaced fragments the span merger missed.
            # A single uppercase char in its own **bold** span almost certainly
            # belongs to the next all-caps bold word (e.g. **E** **DITION**).
            # Apply repeatedly until the pattern is gone.
            bold_frag = re.compile(r'\*\*([A-Z0-9])\*\* \*\*([A-Z][A-Z0-9]*)\*\*')
            while bold_frag.search(markdown_content):
                markdown_content = bold_frag.sub(r'**\1\2**', markdown_content)

            # Safety net for small-caps headers the span merger missed.
            # Restrict to heading lines (start with #) to avoid false merges
            # in body text where a single capital might be a word on its own.
            def fix_smallcaps_heading(m):
                hashes = m.group(1)
                text = m.group(2)
                # Merge: single uppercase letter + space + uppercase word(s)
                fixed = re.sub(r'\b([A-Z]) ([A-Z]{2,})\b', r'\1\2', text)
                return hashes + fixed

            markdown_content = re.sub(
                r'^(#{1,6} )(.+)$',
                fix_smallcaps_heading,
                markdown_content,
                flags=re.MULTILINE,
            )

            def remove_middle_headers(match):
                line = match.group(0)
                # Keep the initial header and remove all subsequent '#' characters
                return re.sub(
                    r"(^#{1,6}\s).*?(?=\n)",
                    lambda m: m.group(1)
                    + re.sub(r"#", "", m.group(0)[len(m.group(1)) :]),
                    line,
                )

            markdown_content = re.sub(
                r"^#{1,6}\s.*\n",
                remove_middle_headers,
                markdown_content,
                flags=re.MULTILINE,
            )  # Remove headers in the middle of lines
            return markdown_content
        except Exception as e:
            self.logger.error(f"Error post-processing markdown: {e}")
            self.logger.exception(traceback.format_exc())
            return markdown_content

    def save_markdown(self, markdown_content):
        """Save the markdown content to a file."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_file = self.output_dir / f"{self.pdf_filename}.md"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            self.logger.info("Markdown content saved successfully.")
        except Exception as e:
            self.logger.error(f"Error saving markdown content: {e}")
            self.logger.exception(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(
        description="Extract markdown-formatted content from a PDF file."
    )
    parser.add_argument("--pdf_path", help="Path to the input PDF file", required=True)
    parser.add_argument(
        "--output_dir",
        help="Directory for output files (default: same directory as the PDF)",
        default=None,
    )
    parser.add_argument(
        "--skip_images",
        help="Skip image extraction and captioning (faster for text-only PDFs)",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    extractor = MarkdownPDFExtractor(
        args.pdf_path,
        output_dir=args.output_dir,
        skip_images=args.skip_images,
    )
    markdown_pages = extractor.extract()
    return markdown_pages


if __name__ == "__main__":
    main()
