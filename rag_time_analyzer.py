import pandas as pd
import glob
import argparse
import sys
from pathlib import Path


def process_rag_files(input_pattern, output_dir="."):
    """
    Process multiple RAG CSV files and generate time analysis reports.

    Args:
        input_pattern (str): File pattern to match input CSV files (e.g., "*.csv" or "rag_*.csv")
        output_dir (str): Directory to save output files
    """

    # Find all matching CSV files
    csv_files = glob.glob(input_pattern)

    if not csv_files:
        print(f"No CSV files found matching pattern: {input_pattern}")
        return

    print(f"Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {file}")

    # Read and combine all CSV files
    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Ensure required columns exist
            required_cols = ['query', 'pipeline1_time_seconds', 'pipeline4_time_seconds']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {file} missing required columns. Skipping.")
                continue
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    if not all_data:
        print("No valid CSV files found to process.")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined data: {len(combined_df)} total rows")

    # Debug: Show unique queries and their counts
    query_counts = combined_df['query'].value_counts()
    print(f"Number of unique queries: {len(query_counts)}")
    print("Query occurrence counts:")
    for query, count in query_counts.head(5).items():
        print(f"  '{query}': {count} times")
    if len(query_counts) > 5:
        print(f"  ... and {len(query_counts) - 5} more")

    # Create long format data for rag_time_long.csv
    long_data = []

    # Get unique queries and sort them for consistent qid assignment
    unique_queries = sorted(combined_df['query'].unique())
    print(f"Generating long format data for {len(unique_queries)} unique queries...")

    for i, query in enumerate(unique_queries, 1):
        query_data = combined_df[combined_df['query'] == query]

        # Calculate mean times for Pipeline 1 and Pipeline 4
        pipeline1_times = query_data['pipeline1_time_seconds'].dropna()
        pipeline4_times = query_data['pipeline4_time_seconds'].dropna()

        # Always add both pipelines if they have data, with the same qid
        if len(pipeline1_times) > 0:
            long_data.append({
                'qid': i,
                'query': query,
                'n_votes': len(pipeline1_times),
                'pipeline': 'Pipeline 1',
                'mean_time_seconds': pipeline1_times.mean()
            })

        if len(pipeline4_times) > 0:
            long_data.append({
                'qid': i,
                'query': query,
                'n_votes': len(pipeline4_times),
                'pipeline': 'Pipeline 4',
                'mean_time_seconds': pipeline4_times.mean()
            })

    # Create long format DataFrame and sort properly
    long_df = pd.DataFrame(long_data)

    # Sort by pipeline first, then by qid to match the expected format
    # This gives us: Pipeline 1 (qid 1-20), then Pipeline 4 (qid 1-20)
    long_df = long_df.sort_values(['pipeline', 'qid']).reset_index(drop=True)

    # Create overall summary for rag_time_overall.csv
    overall_data = []

    # Calculate overall mean for Pipeline 1
    all_pipeline1_times = combined_df['pipeline1_time_seconds'].dropna()
    if len(all_pipeline1_times) > 0:
        overall_data.append({
            'pipeline': 'Pipeline 1',
            'overall_mean_seconds': all_pipeline1_times.mean()
        })

    # Calculate overall mean for Pipeline 4
    all_pipeline4_times = combined_df['pipeline4_time_seconds'].dropna()
    if len(all_pipeline4_times) > 0:
        overall_data.append({
            'pipeline': 'Pipeline 4',
            'overall_mean_seconds': all_pipeline4_times.mean()
        })

    overall_df = pd.DataFrame(overall_data)

    # Save output files
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    long_output = output_path / "rag_time_long.csv"
    overall_output = output_path / "rag_time_overall.csv"

    long_df.to_csv(long_output, index=False)
    overall_df.to_csv(overall_output, index=False)

    print(f"\nOutput files created:")
    print(f"  - {long_output} ({len(long_df)} rows)")
    print(f"  - {overall_output} ({len(overall_df)} rows)")

    # Display summary statistics
    print(f"\nSummary:")
    print(f"  - Processed {len(unique_queries)} unique queries")
    print(f"  - Pipeline 1 overall mean: {all_pipeline1_times.mean():.6f} seconds")
    print(f"  - Pipeline 4 overall mean: {all_pipeline4_times.mean():.6f} seconds")


def calculate_winrate_with_dont_care(input_pattern):
    """
    Calculate and display pipeline win rates including "Don't Care" as a third category.

    Args:
        input_pattern (str): File pattern to match input CSV files
    """
    # Find all matching CSV files
    csv_files = glob.glob(input_pattern)

    if not csv_files:
        print(f"No CSV files found matching pattern: {input_pattern}")
        return

    print(f"Calculating win rates (including Don't Care) from {len(csv_files)} files...")

    # Read and combine all CSV files
    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Ensure winner column exists
            if 'winner' not in df.columns:
                print(f"Warning: {file} missing 'winner' column. Skipping.")
                continue
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    if not all_data:
        print("No valid CSV files found to process.")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    total_entries = len(combined_df)

    # Count wins for the three categories
    pipeline1_wins = len(combined_df[combined_df['winner'] == 'Pipeline 1'])
    pipeline4_wins = len(combined_df[combined_df['winner'] == 'Pipeline 4'])
    dont_care_wins = len(combined_df[combined_df['winner'] == "Don't Care"])

    # Calculate percentages
    pipeline1_pct = (pipeline1_wins / total_entries) * 100
    pipeline4_pct = (pipeline4_wins / total_entries) * 100
    dont_care_pct = (dont_care_wins / total_entries) * 100

    print(f"\n=== PIPELINE WIN RATE ANALYSIS (INCLUDING DON'T CARE) ===")
    print(f"Total files processed: {len(csv_files)}")
    print(f"Total entries: {total_entries}")
    print(f"\nWin Rate Results:")
    print("-" * 50)
    print(f"Pipeline 1: {pipeline1_wins:3d} wins ({pipeline1_pct:5.1f}%)")
    print(f"Pipeline 4: {pipeline4_wins:3d} wins ({pipeline4_pct:5.1f}%)")
    print(f"Don't Care: {dont_care_wins:3d} wins ({dont_care_pct:5.1f}%)")
    print("-" * 50)
    print(
        f"Total: {pipeline1_wins + pipeline4_wins + dont_care_wins:3d} entries ({pipeline1_pct + pipeline4_pct + dont_care_pct:5.1f}%)")

    # Check for any unaccounted entries
    accounted_entries = pipeline1_wins + pipeline4_wins + dont_care_wins
    if accounted_entries != total_entries:
        other_entries = total_entries - accounted_entries
        print(f"Other/Unrecognized: {other_entries:3d} entries")
        print(f"  (These entries have winner values other than 'Pipeline 1', 'Pipeline 4', or 'Don't Care')")

    print(
        f"\nDecisive outcomes: {pipeline1_wins + pipeline4_wins} ({((pipeline1_wins + pipeline4_wins) / total_entries) * 100:.1f}%)")
    print(f"Indecisive outcomes: {dont_care_wins} ({dont_care_pct:.1f}%)")


def calculate_winrate(input_pattern):
    """
    Calculate and display pipeline win rates from CSV files.

    Args:
        input_pattern (str): File pattern to match input CSV files
    """
    # Find all matching CSV files
    csv_files = glob.glob(input_pattern)

    if not csv_files:
        print(f"No CSV files found matching pattern: {input_pattern}")
        return

    print(f"Calculating win rates from {len(csv_files)} files...")

    # Read and combine all CSV files
    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Ensure winner column exists
            if 'winner' not in df.columns:
                print(f"Warning: {file} missing 'winner' column. Skipping.")
                continue
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    if not all_data:
        print("No valid CSV files found to process.")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Filter out "Don't Care" entries for win rate calculation
    valid_winners = combined_df[combined_df['winner'] != "Don't Care"]

    if len(valid_winners) == 0:
        print("No valid winner data found (all entries are 'Don't Care').")
        return

    # Count wins for each pipeline
    win_counts = valid_winners['winner'].value_counts()
    total_valid_entries = len(valid_winners)

    print(f"\n=== PIPELINE WIN RATE ANALYSIS ===")
    print(f"Total files processed: {len(csv_files)}")
    print(f"Total entries: {len(combined_df)}")
    print(f"Valid entries (excluding 'Don't Care'): {total_valid_entries}")
    print(f"'Don't Care' entries: {len(combined_df) - total_valid_entries}")
    print(f"\nWin Rate Results:")
    print("-" * 40)

    for pipeline in sorted(win_counts.index):
        wins = win_counts[pipeline]
        win_rate = (wins / total_valid_entries) * 100
        print(f"{pipeline}: {wins:3d} wins ({win_rate:5.1f}%)")

    print("-" * 40)
    print(f"Total valid competitions: {total_valid_entries}")


def main():
    parser = argparse.ArgumentParser(description="Process RAG timing CSV files and generate analysis reports")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Process command (original functionality)
    process_parser = subparsers.add_parser('process', help='Process CSV files and generate timing reports')
    process_parser.add_argument("input_pattern",
                                help="File pattern to match input CSV files (e.g., '*.csv', '20250816_*.csv')")
    process_parser.add_argument("-o", "--output",
                                default=".",
                                help="Output directory for generated files (default: current directory)")

    # Winrate command
    winrate_parser = subparsers.add_parser('winrate', help='Calculate pipeline win rates (excluding Don\'t Care)')
    winrate_parser.add_argument("input_pattern",
                                help="File pattern to match input CSV files (e.g., '*.csv', '20250816_*.csv')")

    # Winrate with Don't Care command
    winrate_dc_parser = subparsers.add_parser('winrate_dont_care',
                                              help='Calculate pipeline win rates including Don\'t Care as third category')
    winrate_dc_parser.add_argument("input_pattern",
                                   help="File pattern to match input CSV files (e.g., '*.csv', '20250816_*.csv')")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'process':
            process_rag_files(args.input_pattern, args.output)
        elif args.command == 'winrate':
            calculate_winrate(args.input_pattern)
        elif args.command == 'winrate_dont_care':
            calculate_winrate_with_dont_care(args.input_pattern)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()