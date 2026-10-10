#!/usr/bin/env python3

# written by chatgpt on 26 Sep 2026


import argparse
import glob
import os

from PIL import Image
from scipy.spatial import cKDTree


def replace_black_pixels(input_path, output_path, channelwise=False):
    image = Image.open(input_path).convert("RGBA")
    pixels = image.load()
    width, height = image.size

    if channelwise:
        process_channelwise(pixels, width, height)
    else:
        process_rgb(pixels, width, height)

    image.save(output_path)


def process_rgb(pixels, width, height):
    """Replace black pixels with the RGB(A) color of the nearest non-black pixel."""
    non_black_positions = []
    non_black_colors = []

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]

            if (r, g, b) != (0, 0, 0):
                non_black_positions.append((x, y))
                non_black_colors.append((r, g, b, a))

    if not non_black_positions:
        raise ValueError("image contains only black pixels")

    tree = cKDTree(non_black_positions)

    black_positions = []

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]

            if (r, g, b) == (0, 0, 0):
                black_positions.append((x, y))

    if black_positions:
        _, nearest_indices = tree.query(black_positions)

        for position, index in zip(black_positions, nearest_indices):
            pixels[position[0], position[1]] = non_black_colors[index]


def process_channelwise(pixels, width, height):
    """Process R, G, and B independently."""
    for channel in range(3):
        nonzero_positions = []
        nonzero_values = []

        for y in range(height):
            for x in range(width):
                value = pixels[x, y][channel]

                if value != 0:
                    nonzero_positions.append((x, y))
                    nonzero_values.append(value)

        if not nonzero_positions:
            raise ValueError(
                f"channel {channel} contains no nonzero pixels"
            )

        tree = cKDTree(nonzero_positions)

        zero_positions = []

        for y in range(height):
            for x in range(width):
                if pixels[x, y][channel] == 0:
                    zero_positions.append((x, y))

        if zero_positions:
            _, nearest_indices = tree.query(zero_positions)

            for position, index in zip(zero_positions, nearest_indices):
                x, y = position
                rgba = list(pixels[x, y])
                rgba[channel] = nonzero_values[index]
                pixels[x, y] = tuple(rgba)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replace zero-valued pixels in PNG files with values from "
            "the nearest nonzero pixel."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input PNG file or glob pattern",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output file or output directory",
    )

    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input files",
    )

    parser.add_argument(
        "--channelwise",
        action="store_true",
        help=(
            "Process the red, green, and blue channels independently "
            "instead of processing RGB colors as a whole"
        ),
    )

    args = parser.parse_args()

    if args.in_place and args.output:
        parser.error("--in-place and --output cannot be used together")

    if not args.in_place and not args.output:
        parser.error("one of --output or --in-place is required")

    input_paths = glob.glob(args.input)

    if not input_paths:
        parser.error(f"no files matched: {args.input}")

    if args.in_place:
        for input_path in input_paths:
            print(f"Processing {input_path}")
            replace_black_pixels(
                input_path,
                input_path,
                channelwise=args.channelwise,
            )
    else:
        if len(input_paths) == 1:
            replace_black_pixels(
                input_paths[0],
                args.output,
                channelwise=args.channelwise,
            )
        else:
            os.makedirs(args.output, exist_ok=True)

            for input_path in input_paths:
                output_path = os.path.join(
                    args.output,
                    os.path.basename(input_path),
                )

                print(f"Processing {input_path} -> {output_path}")
                replace_black_pixels(
                    input_path,
                    output_path,
                    channelwise=args.channelwise,
                )


if __name__ == "__main__":
    main()