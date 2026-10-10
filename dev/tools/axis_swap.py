
#written by gemini on 27 sep 2026

import argparse
import glob
from pathlib import Path
import numpy as np
from PIL import Image


def load_png_sequence(input_dir: Path) -> tuple[np.ndarray, Image.Image]:
    """Loads a directory of PNG files into a 3D/4D NumPy array with shape (T, Y, X, [C])."""
    png_files = sorted(
        input_dir.glob("*.png"),
        key=lambda p: [
            int(c) if c.isdigit() else c for c in p.stem.replace("_", " ").split()
        ],
    )

    if not png_files:
        raise FileNotFoundError(f"No PNG images found in {input_dir}")

    print(f"Loading {len(png_files)} PNG frames from '{input_dir}'...")

    sample_img = Image.open(png_files[0])

    frames = []
    for filepath in png_files:
        with Image.open(filepath) as img:
            frames.append(np.array(img))

    # Array structure in memory: axis 0 = T, axis 1 = Y, axis 2 = X
    volume = np.stack(frames, axis=0)
    print(f"Loaded volume with shape (T, Y, X, C): {volume.shape}")

    return volume, sample_img


def save_png_sequence(
    volume: np.ndarray, output_dir: Path, template_img: Image.Image
) -> None:
    """Saves a 3D/4D NumPy array as PNGs with 5-digit zero-padded filenames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    num_frames = volume.shape[0]

    print(f"Saving {num_frames} frames to '{output_dir}'...")

    for t in range(num_frames):
        frame_data = volume[t]
        img = Image.fromarray(frame_data)

        if img.mode != template_img.mode and hasattr(template_img, "mode"):
            img = img.convert(template_img.mode)

        # Output filename left-padded with zeroes to 5 digits
        output_path = output_dir / f"frame_{t:05d}.png"
        img.save(output_path)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Swap axes (X, Y, T) of a PNG image sequence."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing source PNG sequence",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where transposed sequence will be saved",
    )
    parser.add_argument(
        "--xyt-to",
        type=str,
        required=True,
        choices=[
            "xyt",
            "xty",
            "yxt",
            "ytx",
            "txy",
            "tyx",
        ],
        help="Target axis arrangement mapping X, Y, T to destination slots (0, 1, 2)",
    )

    args = parser.parse_args()

    volume, sample_img = load_png_sequence(args.input_dir)

    # In NumPy memory space:
    # Axis 0 = T, Axis 1 = Y, Axis 2 = X
    source_axes = {"t": 0, "y": 1, "x": 2}

    # --xyt-to specifies what goes into slot 0 (New T/Output Frame),
    # slot 1 (New Y/Height), and slot 2 (New X/Width).
    #
    # Example for --xyt-to xty:
    # Target Slot 0 (Output Frame) = source X (axis 2)
    # Target Slot 1 (Output Height)= source T (axis 0)
    # Target Slot 2 (Output Width) = source Y (axis 1)
    target_order = [source_axes[char] for char in args.xyt_to.lower()]

    # Preserve color/alpha channel dimension (axis 3) if present
    if volume.ndim == 4:
        target_order.append(3)

    print(f"Permuting axes using permutation vector {target_order} for {args.xyt_to.upper()}...")
    transposed_volume = np.transpose(volume, axes=target_order)
    print(f"Output volume shape (Frames, Height, Width): {transposed_volume.shape[:3]}")

    save_png_sequence(transposed_volume, args.output_dir, sample_img)


if __name__ == "__main__":
    main()