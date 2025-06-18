#!/usr/bin/env python3
"""
OCR Metrics Evaluation CLI
Evaluates OCR/text recognition performance using various metrics including
accuracy, character accuracy, normalized edit distance, correct rate, and accurate rate.
"""

import argparse
import os
import sys
import json
from ppocr.metrics.rec_metric import RecMetric


def load_ground_truth(gt_file):
    """Load ground truth labels from file."""
    gt_data = {}
    try:
        with open(gt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    image_path = parts[0]
                    text = parts[1]
                    # Extract just the filename for matching
                    filename = os.path.basename(image_path)
                    gt_data[filename] = text
                else:
                    print(f"Warning: Invalid line format in ground truth: {line}")
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        sys.exit(1)
    return gt_data


def load_predictions(pred_file):
    """Load predictions from file."""
    pred_data = {}
    try:
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    image_path = parts[0]
                    text = parts[1]
                    confidence = float(parts[2])
                    # Extract just the filename for matching
                    filename = os.path.basename(image_path)
                    pred_data[filename] = (text, confidence)
                else:
                    print(f"Warning: Invalid line format in predictions: {line}")
    except Exception as e:
        print(f"Error loading predictions file: {e}")
        sys.exit(1)
    return pred_data


def match_data(gt_data, pred_data):
    """Match ground truth and predictions by filename."""
    matched_pairs = []
    missing_gt = []
    missing_pred = []

    for filename in gt_data:
        if filename in pred_data:
            gt_text = gt_data[filename]
            pred_text, pred_conf = pred_data[filename]
            matched_pairs.append(((pred_text, pred_conf), (gt_text, None)))
        else:
            missing_pred.append(filename)

    for filename in pred_data:
        if filename not in gt_data:
            missing_gt.append(filename)

    return matched_pairs, missing_gt, missing_pred


def print_results(metrics, total_samples, missing_gt, missing_pred):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print("OCR EVALUATION RESULTS")
    print("=" * 60)

    print(f"Total matched samples: {total_samples}")
    if missing_pred:
        print(f"Missing predictions: {len(missing_pred)} files")
    if missing_gt:
        print(f"Missing ground truth: {len(missing_gt)} files")

    print("\nMetrics:")
    print("-" * 40)
    print(
        f"Accuracy (exact match):     {metrics['acc']:.4f} ({metrics['acc'] * 100:.2f}%)"
    )
    print(
        f"Character Accuracy:         {metrics['char_acc']:.4f} ({metrics['char_acc'] * 100:.2f}%)"
    )
    print(f"Normalized Edit Distance:   {metrics['norm_edit_dis']:.4f}")
    print(
        f"Correct Rate (CR):          {metrics['corr_rate']:.4f} ({metrics['corr_rate'] * 100:.2f}%)"
    )
    print(
        f"Accurate Rate (AR):         {metrics['acc_rate']:.4f} ({metrics['acc_rate'] * 100:.2f}%)"
    )

    print("\nMetric Definitions:")
    print("-" * 40)
    print("• Accuracy: Percentage of exactly matching predictions")
    print("• Character Accuracy: Percentage of correctly predicted characters")
    print("• Normalized Edit Distance: 1 - (average normalized Levenshtein distance)")
    print("• Correct Rate: Accounts for deletions and substitutions only")
    print("• Accurate Rate: Accounts for insertions, deletions, and substitutions")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR/text recognition performance using comprehensive metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ocr_eval.py -g ground_truth.txt -p predictions.txt
  python ocr_eval.py -g gt.txt -p pred.txt --ignore-space --filter
  python ocr_eval.py -g gt.txt -p pred.txt --output results.json
        """,
    )

    parser.add_argument(
        "-g",
        "--ground-truth",
        required=True,
        help="Path to ground truth file (format: filename<tab>text)",
    )

    parser.add_argument(
        "-p",
        "--predictions",
        required=True,
        help="Path to predictions file (format: filepath<tab>text<tab>confidence)",
    )

    parser.add_argument(
        "--ignore-space",
        action="store_true",
        default=True,
        help="Ignore spaces when comparing texts (default: True)",
    )

    parser.add_argument(
        "--no-ignore-space",
        action="store_true",
        help="Do not ignore spaces when comparing texts",
    )

    parser.add_argument(
        "--filter",
        action="store_true",
        help="Apply text normalization (keep only alphanumeric characters)",
    )

    parser.add_argument("-o", "--output", help="Output results to JSON file")

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information about missing files",
    )

    args = parser.parse_args()

    # Handle space ignoring logic
    ignore_space = args.ignore_space and not args.no_ignore_space

    # Validate input files
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        sys.exit(1)

    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)

    print("Loading data...")
    gt_data = load_ground_truth(args.ground_truth)
    pred_data = load_predictions(args.predictions)

    print(f"Loaded {len(gt_data)} ground truth samples")
    print(f"Loaded {len(pred_data)} prediction samples")

    # Match data
    matched_pairs, missing_gt, missing_pred = match_data(gt_data, pred_data)

    if not matched_pairs:
        print("Error: No matching samples found between ground truth and predictions!")
        print("Please check that the filenames match between the two files.")
        sys.exit(1)

    # Show verbose information if requested
    if args.verbose:
        if missing_pred:
            print(f"\nFiles missing predictions ({len(missing_pred)}):")
            for f in missing_pred[:10]:  # Show first 10
                print(f"  {f}")
            if len(missing_pred) > 10:
                print(f"  ... and {len(missing_pred) - 10} more")

        if missing_gt:
            print(f"\nFiles missing ground truth ({len(missing_gt)}):")
            for f in missing_gt[:10]:  # Show first 10
                print(f"  {f}")
            if len(missing_gt) > 10:
                print(f"  ... and {len(missing_gt) - 10} more")

    # Initialize metric calculator
    metric_calc = RecMetric(
        main_indicator="acc", is_filter=args.filter, ignore_space=ignore_space
    )

    print(f"\nEvaluating {len(matched_pairs)} matched samples...")
    print(f"Configuration: ignore_space={ignore_space}, filter={args.filter}")

    # Process in batches to show progress for large datasets
    batch_size = 1000
    for i in range(0, len(matched_pairs), batch_size):
        end_idx = min(i + batch_size, len(matched_pairs))
        batch_pairs = matched_pairs[i:end_idx]
        batch_preds = [pair[0] for pair in batch_pairs]
        batch_labels = [pair[1] for pair in batch_pairs]

        metric_calc((batch_preds, batch_labels))

        if len(matched_pairs) > batch_size:
            print(f"Processed {end_idx}/{len(matched_pairs)} samples...")

    # Get final metrics
    final_metrics = metric_calc.get_metric()

    # Print results
    print_results(final_metrics, len(matched_pairs), missing_gt, missing_pred)

    # Save to JSON if requested
    if args.output:
        output_data = {
            "metrics": final_metrics,
            "total_samples": len(matched_pairs),
            "missing_predictions": len(missing_pred),
            "missing_ground_truth": len(missing_gt),
            "configuration": {"ignore_space": ignore_space, "filter": args.filter},
        }

        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {args.output}")
        except Exception as e:
            print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()
