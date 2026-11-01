import argparse
from pathlib import Path
from part1.data_loader import load_all
from part1.data_analysis import run_data_analysis
from part1.filter_split import run_filter_split


def parse_args():
    parser = argparse.ArgumentParser(description="Recommender Systems Project - MovieLens 1M Pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ml-1m",
        help="Path to directory containing MovieLens 1M .dat files (default: ml-1m)",
    )
    parser.add_argument(
        "--part",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Project part to execute (default: 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print(f"=== Running Recommender Systems Pipeline: Part {args.part} ===")
    print(f"Dataset path: {data_dir.resolve()}\n")

    # Step 1: Ingestion
    ratings, movies, users = load_all(data_dir)

    # Ensure output directory exists for Part 1
    outputs_dir = Path("part1") / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.part == 1:
        print("\nPart 1: Running data analysis...")
        run_data_analysis(ratings, movies, users, outputs_dir)

        print("\nPart 1: Running filtering & temporal split...")
        train_df, val_df, test_df = run_filter_split(ratings, min_item_ratings=5, min_user_ratings=20)


if __name__ == "__main__":
    main()
