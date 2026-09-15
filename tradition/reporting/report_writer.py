from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tradition.evaluation.radarocc_metrics import (
    MetricResult,
    radarocc_metric_dict,
)


def _format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


class MetricsReportWriter:
    """Write only human-facing results: CSV, Markdown, PNG and console text."""

    def write(self, results: list[MetricResult], output_dir: str | Path) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "traditional_metrics.csv"
        md_path = output_dir / "traditional_metrics.md"
        png_path = output_dir / "traditional_metrics.png"

        metrics = radarocc_metric_dict(results)
        rows = [(name, _format_metric(value)) for name, value in metrics.items()]

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Metric", "Traditional"])
            writer.writerows(rows)

        with md_path.open("w", encoding="utf-8") as handle:
            handle.write("| Metric | Traditional |\n")
            handle.write("| --- | ---: |\n")
            for row in rows:
                handle.write(f"| {row[0]} | {row[1]} |\n")

        self._write_png(rows, png_path)
        self._print(rows)
        return {"csv": csv_path, "markdown": md_path, "png": png_path}

    @staticmethod
    def _print(rows: list[tuple[str, str]]) -> None:
        values = {
            name: float(value) if value != "nan" else float("nan")
            for name, value in rows
        }
        print("\nTraditional RadarOcc-compatible results")
        print(values)

    @staticmethod
    def _write_png(rows: list[tuple[str, str]], path: Path) -> None:
        widths = (280, 180)
        row_h = 46
        header_h = 54
        margin = 20
        image = Image.new(
            "RGB",
            (sum(widths) + 2 * margin, header_h + row_h * len(rows) + 2 * margin),
            "white",
        )
        draw = ImageDraw.Draw(image)
        header_font = _font(20, bold=True)
        cell_font = _font(18, bold=False)
        x0, y0 = margin, margin
        headers = ("Metric", "Traditional")

        x = x0
        for width, header in zip(widths, headers):
            draw.rectangle((x, y0, x + width, y0 + header_h), outline=(180, 180, 180), width=1)
            box = draw.textbbox((0, 0), header, font=header_font)
            draw.text((x + (width - (box[2] - box[0])) / 2, y0 + 14), header, fill="black", font=header_font)
            x += width

        for row_index, row in enumerate(rows):
            y = y0 + header_h + row_index * row_h
            x = x0
            for width, value in zip(widths, row):
                draw.rectangle((x, y, x + width, y + row_h), outline=(210, 210, 210), width=1)
                box = draw.textbbox((0, 0), value, font=cell_font)
                draw.text((x + (width - (box[2] - box[0])) / 2, y + 12), value, fill="black", font=cell_font)
                x += width
        image.save(path)
