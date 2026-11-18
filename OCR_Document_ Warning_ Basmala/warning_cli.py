
import argparse
import json
from pathlib import Path

from src.pipeline.ocr_pipeline import OCRPipeline


def main():
    parser = argparse.ArgumentParser(description="Basmala Document Warning CLI")
    parser.add_argument("image", help="Path to document image")
    parser.add_argument(
        "--engine",
        choices=["paddle", "tesseract"],
        default="paddle",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the result as JSON",
    )
    args = parser.parse_args()

    result = OCRPipeline(engine=args.engine).run(args.image)

    print("\n" + "=" * 45)
    print("BASMALA - DOCUMENT WARNING")
    print("=" * 45)
    print(f"Image: {args.image}")
    print(f"OCR Status: {result['status']}")
    print(f"Document Type: {result['document_type']}")
    print(f"Extracted Text: {result['extracted_text'] or '(empty)'}")
    print(f"Confidence: {result['confidence']}")
    print(f"Warning Status: {result['warning_status']}")
    print(f"Warning Category: {result['warning_category']}")
    print(f"Warnings: {', '.join(result['warnings']) or 'None'}")
    print(f"Manual Review Required: {result['review_required']}")
    print("=" * 45)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        print(f"JSON result saved to: {output_path}")


if __name__ == "__main__":
    main()
