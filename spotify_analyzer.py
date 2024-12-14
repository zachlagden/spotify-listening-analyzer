"""
(c) 2024, Zachariah Michael Lagden

Spotify Listening History Analyzer

This script processes Spotify listening history data exported from the Spotify
web interface and provides detailed analysis of listening patterns, artist
preferences, track statistics, and temporal patterns.

Once you export your Spotify data, you can run this script to analyze your
listening history and generate a comprehensive report. Any gaps in the data
can be filled by connecting to the Spotify API to fetch recent plays.

Features:
- Load and process multiple Spotify JSON history files
- Connect to Spotify API to fetch recent plays
- Generate comprehensive listening statistics and patterns
- Analyze artist, track, and album preferences
- Examine temporal listening patterns
- Calculate advanced listening metrics
- Export combined history to JSON

This script requires the following libraries:
- pandas
- spotipy
- tqdm
- python-dotenv

You can install the required libraries using pip:
- pip install -r requirements.txt

Environment Setup:
1. Create a Spotify Developer account
2. Create an application at https://developer.spotify.com/dashboard/create
3. Get your client ID and client secret
4. Create a .env file in the script directory with:
   SPOTIPY_CLIENT_ID=your_client_id
   SPOTIPY_CLIENT_SECRET=your_client_secret

Usage:
1. Export your Spotify listening history from account settings
2. Run this script and point it to your JSON files
3. Follow the prompts to analyze your listening history
"""

# Standard library imports
from pathlib import Path
from typing import List, Dict, Union, Tuple
import json
import os
import sys

# Third-party library imports
from datetime import datetime
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm
import pandas as pd
import spotipy


def format_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human readable format.

    Args:
        size_bytes (int): Size in bytes to format

    Returns:
        str: Formatted string with appropriate unit (B, KB, MB, GB, TB)
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def format_duration(ms: float) -> str:
    """
    Format milliseconds into human readable duration.

    Args:
        ms (float): Duration in milliseconds

    Returns:
        str: Formatted string in the format "Xh Ym Zs"
    """
    seconds = ms / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def confirm_continue(message: str, show_divider: bool = True) -> bool:
    """
    Ask user if they want to continue processing.

    Args:
        message (str): The prompt message to show
        show_divider (bool): Whether to show a divider line before the prompt

    Returns:
        bool: True if user wants to continue, False otherwise
    """
    if show_divider:
        print("\n" + "=" * 80)

    while True:
        choice = input(f"\n{message} (y/n): ").lower().strip()
        if choice in ["y", "yes"]:
            return True
        elif choice in ["n", "no"]:
            return False
        print("Please enter 'y' or 'n'")


def print_section(title: str):
    """
    Print a section title with decorative formatting.

    Args:
        title (str): The title to display
    """
    print("\n" + "=" * 80)
    print(f"🔍 {title}")
    print("=" * 80)


class SpotifyAnalyzer:
    """
    A class to analyze Spotify listening history data.

    This class handles loading, processing, and analyzing Spotify listening history
    data from JSON files and optionally fetching recent plays via the Spotify API.

    Attributes:
        file_paths (List[str]): Paths to JSON history files
        df (pd.DataFrame): Processed listening history data
        cache (Dict): Cache for computed statistics
        client_id (str): Spotify API client ID
        client_secret (str): Spotify API client secret
        sp (spotipy.Spotify): Spotify API client
        total_entries_processed (int): Count of processed entries
        processing_stats (Dict): Statistics about the processing
        original_data (List): Original format of the data
    """

    def __init__(
        self,
        file_paths: Union[str, List[str]],
        client_id: str = None,
        client_secret: str = None,
    ):
        """
        Initialize the Spotify analyzer.

        Args:
            file_paths: Path(s) to JSON history files
            client_id: Optional Spotify API client ID
            client_secret: Optional Spotify API client secret
        """
        self.file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        self.df = None
        self.cache = {}
        self.client_id = client_id
        self.client_secret = client_secret
        self.sp = None
        self.total_entries_processed = 0
        self.processing_stats = {
            "files_processed": 0,
            "total_entries": 0,
            "duplicates_removed": 0,
            "api_entries_added": 0,
        }
        self.original_data = []

    def load_file(self, file_path: str) -> Tuple[List[Dict], Dict]:
        """
        Load and analyze a single JSON history file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Tuple containing:
            - List of history entries
            - Dictionary of file statistics
        """
        stats = {
            "file_size": os.path.getsize(file_path),
            "entries": 0,
            "earliest_entry": None,
            "latest_entry": None,
            "unique_tracks": set(),
            "unique_artists": set(),
        }

        # Read file in binary mode for better performance
        with open(file_path, "rb") as file:
            data = json.load(file)

        stats["entries"] = len(data)

        # Single pass through data to collect all stats
        min_ts = float("inf")
        max_ts = float("-inf")

        for entry in data:
            # Parse timestamp once and store as integer for faster comparison
            ts = int(
                datetime.fromisoformat(entry["ts"].replace("Z", "+00:00")).timestamp()
            )
            if ts < min_ts:
                min_ts = ts
            if ts > max_ts:
                max_ts = ts

            # Add to sets directly without get()
            if track := entry.get("master_metadata_track_name"):
                stats["unique_tracks"].add(track)
            if artist := entry.get("master_metadata_album_artist_name"):
                stats["unique_artists"].add(artist)

        # Convert timestamps back to datetime objects at the end
        stats["earliest_entry"] = datetime.fromtimestamp(min_ts)
        stats["latest_entry"] = datetime.fromtimestamp(max_ts)

        return data, stats

    def connect_to_spotify(self) -> bool:
        """
        Initialize connection to the Spotify API.

        Returns:
            bool: True if connection successful, False otherwise
        """
        print_section("Spotify API Connection")

        if not (self.client_id and self.client_secret):
            print("⚠️ No Spotify API credentials provided - skipping live data fetch")
            return False

        try:
            auth_manager = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri="http://localhost:8888/callback",
                scope="user-read-recently-played",
            )

            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            self.sp.current_user()  # Test the connection
            print("✅ Successfully connected to Spotify API")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Spotify API: {str(e)}")
            return False

    def fetch_recent_plays(self, last_timestamp: pd.Timestamp) -> List[Dict]:
        """
        Fetch plays since the last entry in the JSON data.

        Args:
            last_timestamp: Timestamp of most recent play in JSON data

        Returns:
            List of play history entries from the API
        """
        if not self.sp:
            return []

        print_section("Fetching Recent Plays")

        # Ensure timestamp is UTC
        if last_timestamp.tz is None:
            last_timestamp = last_timestamp.tz_localize("UTC")

        current_time = pd.Timestamp.now(tz="UTC")
        print(f"📅 Fetching plays from {last_timestamp:%Y-%m-%d %H:%M:%S %Z} to now")

        # Convert timestamps to milliseconds for API
        after_timestamp = int(last_timestamp.timestamp() * 1000)
        current_timestamp = int(current_time.timestamp() * 1000)
        initial_gap = current_timestamp - after_timestamp

        all_plays = []
        with tqdm(
            total=100, desc="Fetching plays", bar_format="{l_bar}{bar}| {n_fmt}%"
        ) as pbar:
            while after_timestamp < current_timestamp:
                try:
                    results = self.sp.current_user_recently_played(
                        limit=50, after=after_timestamp
                    )

                    if not results["items"]:
                        break

                    batch_plays = []
                    earliest_ts = None

                    # Process batch of plays
                    for item in results["items"]:
                        play = {
                            "ts": item["played_at"],
                            "master_metadata_track_name": item["track"]["name"],
                            "master_metadata_album_artist_name": item["track"][
                                "artists"
                            ][0]["name"],
                            "master_metadata_album_album_name": item["track"]["album"][
                                "name"
                            ],
                            "spotify_track_uri": item["track"]["uri"],
                            "ms_played": item["track"]["duration_ms"],
                        }
                        batch_plays.append(play)

                        ts = pd.to_datetime(item["played_at"]).tz_convert("UTC")
                        if earliest_ts is None or ts < earliest_ts:
                            earliest_ts = ts

                    all_plays.extend(batch_plays)

                    # Update progress bar
                    if earliest_ts:
                        after_timestamp = int(earliest_ts.timestamp() * 1000)
                        progress = min(
                            (
                                (
                                    after_timestamp
                                    - int(last_timestamp.timestamp() * 1000)
                                )
                                / initial_gap
                            )
                            * 100,
                            100,
                        )
                        pbar.n = int(progress)
                        pbar.refresh()

                except Exception as e:
                    print(f"\n❌ Error fetching plays: {str(e)}")
                    break

        print(f"✅ Fetched {len(all_plays):,} new plays")
        return all_plays

    def _process_recent_plays(self, recent_plays: List[Dict]) -> None:
        """
        Process and add recent plays to the dataset.

        Args:
            recent_plays: List of play history entries from API
        """
        if not recent_plays:
            return

        print("\n📥 Processing recent plays...")

        # Convert to DataFrame and process
        recent_df = pd.DataFrame(recent_plays)
        recent_df["ts"] = pd.to_datetime(recent_df["ts"]).dt.tz_convert("UTC")

        # Merge with existing data and deduplicate
        self.df = pd.concat([self.df, recent_df], ignore_index=True)
        self.df = self.df.drop_duplicates(
            subset=["ts", "spotify_track_uri", "ms_played"]
        )

        self._process_timestamps()

        print(f"✨ Added {len(recent_plays)} recent plays to the analysis")
        print(
            f"📊 Updated date range: {self.df['ts'].min():%Y-%m-%d %H:%M:%S %Z} to {self.df['ts'].max():%Y-%m-%d %H:%M:%S %Z}"
        )

    def load_and_process_data(self) -> None:
        """
        Load and process all data files with progress tracking.

        This method handles:
        - Loading JSON files
        - Converting to DataFrame
        - Removing duplicates
        - Processing timestamps
        - Fetching recent plays from API
        - Offering to export processed data
        """
        print_section("Data Loading and Processing")

        all_history = []
        file_stats = []
        total_size = sum(os.path.getsize(fp) for fp in self.file_paths)

        print("📂 Initial File Analysis:")
        print(f"Found {len(self.file_paths)} files totaling {format_size(total_size)}")

        # Process each file
        for file_path in tqdm(self.file_paths, desc="Loading files"):
            print(f"\n📄 Processing {file_path}...")
            data, stats = self.load_file(file_path)

            # Store original format data
            self.original_data.extend(data)

            print(f"• Size: {format_size(stats['file_size'])}")
            print(f"• Entries: {stats['entries']:,}")
            print(
                f"• Date range: {stats['earliest_entry']:%Y-%m-%d} to {stats['latest_entry']:%Y-%m-%d}"
            )
            print(f"• Unique tracks: {len(stats['unique_tracks']):,}")
            print(f"• Unique artists: {len(stats['unique_artists']):,}")

            all_history.extend(data)
            file_stats.append(stats)
            self.processing_stats["files_processed"] += 1
            self.processing_stats["total_entries"] += stats["entries"]

        if not confirm_continue(
            f"Loaded {len(all_history):,} total entries. Continue with processing?"
        ):
            print("🛑 Processing cancelled")
            sys.exit(0)

        # Convert to DataFrame and process
        print("\n🔄 Converting to DataFrame...")
        self.df = pd.DataFrame(all_history)

        # Remove duplicates
        print("\n🔍 Checking for duplicates...")
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(
            subset=["ts", "spotify_track_uri", "ms_played"]
        )
        dupes_removed = initial_rows - len(self.df)
        self.processing_stats["duplicates_removed"] = dupes_removed

        if dupes_removed > 0:
            print(f"✨ Cleaned {dupes_removed:,} duplicate entries")
            print(f"Reduced dataset from {initial_rows:,} to {len(self.df):,} entries")

        # Process timestamps
        print("\n⏰ Processing timestamps and generating derived columns...")
        self._process_timestamps()

        # Handle gap filling if API credentials available
        if self.client_id and self.client_secret:
            last_timestamp = self.df["ts"].max()
            current_time = pd.Timestamp.now(tz="UTC")
            gap_days = (current_time - last_timestamp).days

            if gap_days < 1:
                gap_minutes = (current_time - last_timestamp).seconds // 60

            print("\n📊 Time Gap Analysis")
            print(f"Last entry in data: {last_timestamp:%Y-%m-%d %H:%M:%S %Z}")
            print(f"Current time: {current_time:%Y-%m-%d %H:%M:%S %Z}")
            print(
                f"Gap to fill: {str(gap_days) +' days' if gap_days > 0 else str(gap_minutes) + ' minutes'}"
            )

            if gap_days > 0 or gap_minutes > 0:
                print("\nGap Filling Options:")
                print("1. Fill entire gap")
                print("2. Fill last month only")
                print("3. Fill custom time period")
                print("4. Skip gap filling")

                while True:
                    choice = input("\nSelect an option (1-4): ").strip()
                    if choice in ["1", "2", "3", "4"]:
                        break
                    print("Please enter a number between 1 and 4")

                if choice == "1":
                    if self.connect_to_spotify():
                        recent_plays = self.fetch_recent_plays(last_timestamp)
                        if recent_plays:
                            self._process_recent_plays(recent_plays)
                            self.original_data.extend(recent_plays)

                elif choice == "2":
                    month_ago = current_time - pd.Timedelta(days=30)
                    start_time = max(last_timestamp, month_ago)
                    if self.connect_to_spotify():
                        recent_plays = self.fetch_recent_plays(start_time)
                        if recent_plays:
                            self._process_recent_plays(recent_plays)
                            self.original_data.extend(recent_plays)

                elif choice == "3":
                    days = input(
                        "Enter number of days to fill (or press Enter for custom date): "
                    ).strip()
                    if days.isdigit():
                        start_time = current_time - pd.Timedelta(days=int(days))
                        start_time = max(last_timestamp, start_time)
                    else:
                        while True:
                            date_str = input("Enter start date (YYYY-MM-DD): ").strip()
                            try:
                                start_time = pd.to_datetime(date_str)
                                if start_time < last_timestamp:
                                    start_time = last_timestamp
                                break
                            except ValueError:
                                print("Invalid date format. Please use YYYY-MM-DD")

                    if self.connect_to_spotify():
                        recent_plays = self.fetch_recent_plays(start_time)
                        if recent_plays:
                            self._process_recent_plays(recent_plays)
                            self.original_data.extend(recent_plays)

        # Print processing summary
        print_section("Processing Summary")
        print(f"✅ Files processed: {self.processing_stats['files_processed']}")
        print(f"📊 Total entries processed: {self.processing_stats['total_entries']:,}")
        print(f"🧹 Duplicates removed: {self.processing_stats['duplicates_removed']:,}")
        print(
            f"📅 Date range: {self.df['ts'].min():%Y-%m-%d} to {self.df['ts'].max():%Y-%m-%d}"
        )
        print(
            f"👥 Unique artists: {self.df['master_metadata_album_artist_name'].nunique():,}"
        )
        print(f"🎵 Unique tracks: {self.df['master_metadata_track_name'].nunique():,}")
        print(
            f"💿 Unique albums: {self.df['master_metadata_album_album_name'].nunique():,}"
        )

        print("\n✅ Data processing complete!")

        # Offer to export combined data
        if confirm_continue(
            "\nWould you like to export the complete processed history to a JSON file?"
        ):
            self.export_full_history()

    def export_full_history(self) -> None:
        """
        Export the complete processed history to a JSON file.

        This method handles:
        - Converting DataFrame back to original format
        - Deduplicating entries
        - Preserving original data structure
        - Writing to JSON file
        """
        print("\n📤 Preparing to export processed history...")

        # Convert DataFrame back to original format if we used the DataFrame
        if self.df is not None:
            # Define original columns to preserve structure
            original_columns = [
                "ts",
                "spotify_track_uri",
                "ms_played",
                "master_metadata_track_name",
                "master_metadata_album_artist_name",
                "master_metadata_album_album_name",
                "platform",
                "conn_country",
                "ip_addr_decrypted",
                "user_agent_decrypted",
                "episode_name",
                "episode_show_name",
                "spotify_episode_uri",
                "reason_start",
                "reason_end",
                "shuffle",
                "skipped",
                "offline",
                "offline_timestamp",
                "incognito_mode",
            ]

            # Select only columns that exist in the DataFrame
            export_columns = [col for col in original_columns if col in self.df.columns]

            # Get unique entries based on core identifying fields
            unique_entries = self.df[export_columns].drop_duplicates(
                subset=["ts", "spotify_track_uri", "ms_played"]
            )

            # Convert to list of dicts while preserving original format
            export_data = unique_entries.to_dict("records")

            # Ensure proper datetime format and handle missing fields
            for entry in export_data:
                # Convert timestamp to proper format
                if isinstance(entry["ts"], pd.Timestamp):
                    entry["ts"] = entry["ts"].strftime("%Y-%m-%dT%H:%M:%S.%fZ")

                # Ensure all original fields exist (even if null)
                for col in original_columns:
                    if col not in entry:
                        entry[col] = None
        else:
            # Use original_data if DataFrame wasn't created
            export_data = self.original_data

        # Remove duplicates while preserving order
        seen = set()
        deduplicated_data = []
        for entry in export_data:
            # Create a tuple of identifying fields
            entry_key = (
                entry["ts"],
                entry.get("spotify_track_uri"),
                entry.get("ms_played"),
            )
            if entry_key not in seen:
                seen.add(entry_key)
                deduplicated_data.append(entry)

        # Sort by timestamp
        deduplicated_data.sort(key=lambda x: x["ts"])

        output_file = "history_full.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(deduplicated_data, f, ensure_ascii=False, indent=2)

        size = os.path.getsize(output_file)
        print(f"\n✅ Exported {len(deduplicated_data):,} entries to {output_file}")
        print(f"📁 File size: {format_size(size)}")

    def _process_timestamps(self) -> None:
        """
        Process timestamps and add derived columns to the DataFrame.

        Adds columns for:
        - Duration in hours and minutes
        - Year, month, hour
        - Day of week
        - Week number
        - Weekend flag
        - Time of day category
        - Month-year period
        - Day
        - Quarter
        - Season
        """
        self.df["ts"] = pd.to_datetime(self.df["ts"]).dt.tz_convert("UTC")

        ts_naive = self.df["ts"].dt.tz_localize(None)

        # Duration columns
        self.df["duration_hours"] = self.df["ms_played"] / (1000 * 60 * 60)
        self.df["duration_minutes"] = self.df["ms_played"] / (1000 * 60)

        # Time-based columns
        self.df["year"] = self.df["ts"].dt.year
        self.df["month"] = self.df["ts"].dt.month
        self.df["hour"] = self.df["ts"].dt.hour
        self.df["day_of_week"] = self.df["ts"].dt.day_name()
        self.df["week_number"] = self.df["ts"].dt.isocalendar().week
        self.df["is_weekend"] = self.df["ts"].dt.dayofweek.isin([5, 6])

        # Time of day categories
        self.df["time_of_day"] = pd.cut(
            self.df["hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["Night", "Morning", "Afternoon", "Evening"],
        )

        # Additional time periods
        self.df["month_year"] = ts_naive.dt.to_period("M")
        self.df["day"] = ts_naive.dt.date
        self.df["quarter"] = self.df["ts"].dt.quarter
        self.df["season"] = pd.cut(
            self.df["month"],
            bins=[0, 3, 6, 9, 12],
            labels=["Winter", "Spring", "Summer", "Fall"],
        )

    def _calculate_time_difference(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        """
        Calculate and format time difference between two timestamps.

        Args:
            start: Starting timestamp
            end: Ending timestamp

        Returns:
            str: Formatted time difference
        """
        if start.tz is None:
            start = start.tz_localize("UTC")
        if end.tz is None:
            end = end.tz_localize("UTC")

        diff = end - start
        days = diff.days
        hours = diff.components.hours
        minutes = diff.components.minutes

        if days > 0:
            return f"{days} days, {hours} hours"
        elif hours > 0:
            return f"{hours} hours, {minutes} minutes"
        else:
            return f"{minutes} minutes"

    def generate_report(self) -> None:
        """
        Generate a comprehensive analysis report.

        This method coordinates the generation of various analysis sections:
        - Overall statistics
        - Artist analysis
        - Track analysis
        - Album analysis
        - Temporal patterns
        - Advanced metrics
        """
        if self.df is None:
            print("❌ No data loaded. Please run load_and_process_data() first.")
            return

        print_section("COMPREHENSIVE MUSIC LISTENING ANALYSIS")

        sections = [
            ("overall_stats", "Overall Statistics"),
            ("artist_analysis", "Artist Analysis"),
            ("track_analysis", "Track Analysis"),
            ("album_analysis", "Album Analysis"),
            ("temporal_patterns", "Temporal Patterns"),
            ("advanced_metrics", "Advanced Metrics"),
        ]

        for method_name, section_title in sections:
            method = f"_analyze_{method_name}"
            if hasattr(self, method):
                print_section(section_title)
                getattr(self, method)()
            else:
                print(f"\nWarning: Analysis method {method} not found")

    def _analyze_overall_stats(self) -> None:
        """
        Analyze overall listening statistics.

        Calculates and displays:
        - Date range and period coverage
        - Total listening time and daily averages
        - Track and artist counts
        - Session statistics
        - Weekend vs weekday patterns
        """
        date_range = f"{self.df['ts'].min():%Y-%m-%d} to {self.df['ts'].max():%Y-%m-%d}"
        days_covered = (self.df["ts"].max() - self.df["ts"].min()).days
        total_hours = self.df["duration_hours"].sum()
        active_days = self.df["day"].nunique()

        stats = {
            "📅 Period": f"{date_range} ({days_covered:,} days)",
            "⏱️ Total listening time": f"{total_hours:.1f} hours ({(total_hours/24):.1f} days)",
            "📊 Daily average": f"{(total_hours*60/days_covered):.1f} minutes",
            "🎵 Total tracks played": f"{len(self.df):,}",
            "🎸 Unique tracks": f"{self.df['master_metadata_track_name'].nunique():,}",
            "👥 Unique artists": f"{self.df['master_metadata_album_artist_name'].nunique():,}",
            "💿 Unique albums": f"{self.df['master_metadata_album_album_name'].nunique():,}",
            "📅 Active listening days": f"{active_days} ({(active_days/days_covered)*100:.1f}% of period)",
            "🏁 Weekend vs Weekday ratio": f"{self.df[self.df['is_weekend']]['duration_hours'].sum() / self.df[~self.df['is_weekend']]['duration_hours'].sum():.2f}",
            "🌙 Night listening": f"{(len(self.df[self.df['time_of_day'] == 'Night']) / len(self.df) * 100):.1f}% of plays",
        }

        for key, value in stats.items():
            print(f"{key}: {value}")

    def _analyze_artist_analysis(self) -> None:
        """
        Perform detailed artist analysis.

        Analyzes and displays:
        - Top 15 artists by listening time
        - Detailed statistics for each artist
        - Listening patterns and trends
        """
        if "artist_stats" not in self.cache:
            self.cache["artist_stats"] = (
                self.df.groupby("master_metadata_album_artist_name", observed=False)
                .agg(
                    {
                        "ms_played": ["sum", "mean", "count"],
                        "master_metadata_track_name": ["count", "nunique"],
                        "master_metadata_album_album_name": "nunique",
                        "is_weekend": "mean",
                        "ts": ["min", "max"],
                    }
                )
                .sort_values(("ms_played", "sum"), ascending=False)
            )

        artist_stats = self.cache["artist_stats"]

        print("\n🎸 Top 15 Artists by Listening Time:")
        for artist, data in artist_stats.head(15).iterrows():
            print("\n" + "─" * 40)
            hours = data[("ms_played", "sum")] / (1000 * 60 * 60)
            avg_time = data[("ms_played", "mean")] / (1000 * 60)
            total_plays = data[("master_metadata_track_name", "count")]
            unique_tracks = data[("master_metadata_track_name", "nunique")]
            unique_albums = data[("master_metadata_album_album_name", "nunique")]
            weekend_percent = data[("is_weekend", "mean")] * 100
            first_played = data[("ts", "min")]
            last_played = data[("ts", "max")]

            print(f"🎤 {artist}")
            print(f"  • Total Time: {hours:.1f} hours")
            print(f"  • Average Play Duration: {avg_time:.1f} minutes")
            print(f"  • Tracks: {unique_tracks} unique across {unique_albums} albums")
            print(
                f"  • Plays: {total_plays:,} total (avg {(total_plays/unique_tracks):.1f} per track)"
            )
            print(f"  • Weekend Listening: {weekend_percent:.1f}% of plays")
            print(f"  • First Played: {first_played:%Y-%m-%d}")
            print(f"  • Last Played: {last_played:%Y-%m-%d}")
            print(f"  • Active Period: {(last_played - first_played).days} days")

    def _analyze_track_analysis(self) -> None:
        """
        Perform detailed track analysis.

        Analyzes and displays:
        - Top 15 most played tracks
        - Play counts and durations
        - Listening patterns
        - First/last played dates
        """
        if "track_stats" not in self.cache:
            self.cache["track_stats"] = (
                self.df.groupby(
                    [
                        "master_metadata_track_name",
                        "master_metadata_album_artist_name",
                        "master_metadata_album_album_name",
                    ],
                    observed=False,
                )
                .agg(
                    {
                        "ms_played": ["sum", "count", "mean"],
                        "ts": ["min", "max"],
                        "is_weekend": "mean",
                        "time_of_day": lambda x: (
                            x.mode().iloc[0] if len(x.mode()) > 0 else "Various"
                        ),
                    }
                )
                .sort_values(("ms_played", "sum"), ascending=False)
            )

        track_stats = self.cache["track_stats"]

        print("\n🎵 Top 15 Most Played Tracks:")
        for (track, artist, album), data in track_stats.head(15).iterrows():
            if pd.isna(track) or pd.isna(artist) or pd.isna(album):
                continue

            print("\n" + "─" * 40)
            hours = data[("ms_played", "sum")] / (1000 * 60 * 60)
            plays = data[("ms_played", "count")]
            avg_completion = data[("ms_played", "mean")] / 1000
            weekend_ratio = data[("is_weekend", "mean")] * 100
            most_common_time = data[("time_of_day", "<lambda>")]

            print(f"🎵 {track}")
            print(f"👤 by {artist}")
            print(f"💿 from {album}")
            print(f"  • Listening Time: {hours:.1f} hours")
            print(f"  • Total Plays: {plays:,}")
            print(f"  • Average Duration: {avg_completion:.1f} seconds")
            print(f"  • Weekend Plays: {weekend_ratio:.1f}%")
            print(f"  • Most Common Time: {most_common_time}")
            print(f"  • First Played: {data[('ts', 'min')].strftime('%Y-%m-%d')}")
            print(f"  • Last Played: {data[('ts', 'max')].strftime('%Y-%m-%d')}")
            print(
                f"  • Days Between First/Last: {(data[('ts', 'max')] - data[('ts', 'min')]).days}"
            )

    def _analyze_album_analysis(self) -> None:
        """
        Perform detailed album analysis.

        Analyzes and displays:
        - Top 10 most played albums
        - Listening statistics per album
        - Track counts and play patterns
        - Album lifecycle information
        """
        if "album_stats" not in self.cache:
            # Group by album and artist, handling the time_of_day mode calculation separately
            basic_stats = self.df.groupby(
                [
                    "master_metadata_album_album_name",
                    "master_metadata_album_artist_name",
                ],
                observed=False,
            ).agg(
                {
                    "ms_played": ["sum", "mean", "count"],
                    "master_metadata_track_name": ["count", "nunique"],
                    "ts": ["min", "max"],
                    "is_weekend": "mean",
                }
            )

            # Calculate mode of time_of_day separately
            time_modes = self.df.groupby(
                [
                    "master_metadata_album_album_name",
                    "master_metadata_album_artist_name",
                ],
                observed=False,
            )["time_of_day"].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Various"
            )

            # Combine the stats
            self.cache["album_stats"] = pd.concat([basic_stats, time_modes], axis=1)
            self.cache["album_stats"].sort_values(
                ("ms_played", "sum"), ascending=False, inplace=True
            )

        album_stats = self.cache["album_stats"]

        print("\n💿 Top 10 Most Played Albums:")
        for (album, artist), data in album_stats.head(10).iterrows():
            if pd.isna(album) or pd.isna(artist):
                continue

            print("\n" + "─" * 40)
            hours = data[("ms_played", "sum")] / (1000 * 60 * 60)
            total_plays = data[("master_metadata_track_name", "count")]
            unique_tracks = data[("master_metadata_track_name", "nunique")]
            avg_time = data[("ms_played", "mean")] / (1000 * 60)
            weekend_ratio = data[("is_weekend", "mean")] * 100
            most_common_time = data["time_of_day"]

            print(f"💿 {album}")
            print(f"👤 by {artist}")
            print(f"  • Total Time: {hours:.1f} hours")
            print(f"  • Average Play Duration: {avg_time:.1f} minutes")
            print(f"  • Tracks: {unique_tracks} tracks played {total_plays:,} times")
            print(f"  • Average Plays per Track: {(total_plays/unique_tracks):.1f}")
            print(f"  • Weekend Listening: {weekend_ratio:.1f}%")
            print(f"  • Most Common Time: {most_common_time}")
            print(f"  • First Played: {data[('ts', 'min')].strftime('%Y-%m-%d')}")
            print(f"  • Last Played: {data[('ts', 'max')].strftime('%Y-%m-%d')}")
            print(
                f"  • Days in Rotation: {(data[('ts', 'max')] - data[('ts', 'min')]).days}"
            )

    def _analyze_temporal_patterns(self) -> None:
        """
        Analyze temporal listening patterns.

        Analyzes and displays:
        - Time of day distribution
        - Day of week patterns
        - Monthly listening patterns
        - Seasonal trends
        """
        # Time of day analysis
        time_dist = self.df.groupby("time_of_day", observed=False)[
            "duration_hours"
        ].sum()
        total_hours = time_dist.sum()

        print("\n⏰ Time of Day Distribution:")
        for period, hours in time_dist.items():
            percentage = (hours / total_hours) * 100
            print(f"  • {period}: {hours:.1f} hours ({percentage:.1f}%)")

        # Day of week analysis
        print("\n📅 Day of Week Patterns:")
        dow_stats = (
            self.df.groupby("day_of_week", observed=False)
            .agg(
                {
                    "duration_hours": "sum",
                    "master_metadata_track_name": "count",
                    "master_metadata_album_artist_name": "nunique",
                }
            )
            .reindex(
                [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
            )
        )

        for day, stats in dow_stats.iterrows():
            print(f"\n  • {day}:")
            print(f"    - {stats['duration_hours']:.1f} hours")
            print(f"    - {stats['master_metadata_track_name']:,} tracks played")
            print(
                f"    - {stats['master_metadata_album_artist_name']} different artists"
            )

        # Monthly patterns
        print("\n📊 Monthly Listening Patterns:")
        monthly_stats = self.df.groupby(["year", "month"], observed=False).agg(
            {
                "duration_hours": "sum",
                "master_metadata_track_name": "count",
                "master_metadata_album_artist_name": "nunique",
            }
        )

        for (year, month), stats in monthly_stats.iterrows():
            month_name = pd.Timestamp(year=year, month=month, day=1).strftime("%B")
            print(f"\n  • {month_name} {year}:")
            print(f"    - {stats['duration_hours']:.1f} hours")
            print(f"    - {stats['master_metadata_track_name']:,} tracks played")
            print(
                f"    - {stats['master_metadata_album_artist_name']} different artists"
            )

        # Seasonal patterns
        print("\n🌤️ Seasonal Patterns:")
        season_stats = self.df.groupby("season", observed=False).agg(
            {
                "duration_hours": "sum",
                "master_metadata_track_name": "count",
                "master_metadata_album_artist_name": "nunique",
            }
        )

        for season, stats in season_stats.iterrows():
            print(f"\n  • {season}:")
            print(f"    - {stats['duration_hours']:.1f} hours")
            print(f"    - {stats['master_metadata_track_name']:,} tracks")
            print(f"    - {stats['master_metadata_album_artist_name']} artists")

    def _analyze_advanced_metrics(self) -> None:
        """
        Calculate and display advanced listening metrics.

        Analyzes and displays:
        - Listening consistency scores
        - Daily statistics
        - Listening streaks
        - Time preferences
        - Monthly trends
        """
        # Calculate daily listening time in minutes
        daily_listening = self.df.groupby(self.df["ts"].dt.date)[
            "duration_minutes"
        ].sum()
        active_days = daily_listening[daily_listening > 0]

        consistency_score = (len(active_days) / len(daily_listening)) * 100

        # Calculate unique artists per day
        unique_artists_per_day = self.df.groupby(self.df["ts"].dt.date)[
            "master_metadata_album_artist_name"
        ].nunique()

        # Identify heavy and light listening days using active days only
        mean_listening = active_days.mean()
        std_listening = active_days.std()

        heavy_listening_days = len(
            active_days[active_days > mean_listening + std_listening]
        )
        light_listening_days = len(
            active_days[active_days < mean_listening - std_listening]
        )

        # Find heavily repeated tracks
        track_counts = self.df["master_metadata_track_name"].value_counts()
        heavily_repeated = track_counts[track_counts >= 10].count()

        # Calculate listening streaks
        daily_listening_bool = daily_listening > 0
        dates = sorted(daily_listening_bool.index)

        # Find longest streak
        max_streak = 0
        current_streak = 0
        max_streak_end_date = None

        for i, date in enumerate(dates):
            if daily_listening_bool[date]:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    max_streak_end_date = date
            else:
                current_streak = 0

        # Calculate current streak
        current_streak = 0
        for date in reversed(dates):
            if daily_listening_bool[date]:
                current_streak += 1
            else:
                break

        # Format streak information
        if max_streak_end_date:
            max_streak_start_date = max_streak_end_date - pd.Timedelta(
                days=max_streak - 1
            )
            streak_info = (
                f"{max_streak} days "
                f"({max_streak_start_date:%Y-%m-%d} to {max_streak_end_date:%Y-%m-%d})"
            )
        else:
            streak_info = f"{max_streak} days"

        # Analyze time preferences
        time_preferences = self.df.groupby("time_of_day", observed=False).size()
        primary_time = time_preferences.idxmax()
        primary_time_percentage = (time_preferences.max() / len(self.df)) * 100

        # Print all metrics in a single, organized section
        print("\n📈 Advanced Listening Patterns")
        print("\n📊 Listening Habits and Patterns")

        metrics = {
            "Daily Consistency": f"{consistency_score:.1f}% of days have activity",
            "Average Daily Artists": f"{unique_artists_per_day.mean():.1f}",
            "Heavy Listening Days": f"{heavy_listening_days} ({heavy_listening_days/len(active_days)*100:.1f}% of active days)",
            "Light Listening Days": f"{light_listening_days} ({light_listening_days/len(active_days)*100:.1f}% of active days)",
            "Frequently Repeated Tracks": f"{heavily_repeated:,} tracks played 10+ times",
            "Daily Track Variety": f"{self.df.groupby(self.df['ts'].dt.date)['master_metadata_track_name'].nunique().mean():.1f} unique tracks",
            "Longest Listening Streak": streak_info,
            "Current Streak": f"{current_streak} days",
            "Primary Listening Time": f"{primary_time} ({primary_time_percentage:.1f}% of plays)",
        }

        for metric, value in metrics.items():
            print(f"• {metric}: {value}")

        # Daily statistics using active days only
        print("\n📈 Daily Listening Statistics")
        daily_stats = {
            "Average listening time": f"{active_days.mean():.1f} minutes",
            "Median listening time": f"{active_days.median():.1f} minutes",
            "Most active day": f"{active_days.idxmax():%Y-%m-%d} ({active_days.max():.1f} minutes)",
            "Standard deviation": f"{active_days.std():.1f} minutes",
        }

        for stat, value in daily_stats.items():
            print(f"• {stat}: {value}")

        # Monthly trend analysis
        print("\n📅 Monthly Trend Analysis")
        monthly_listening = self.df.groupby(["year", "month"])["duration_hours"].sum()
        month_stats = {
            "Most active month": f"{monthly_listening.idxmax()[1]}/{monthly_listening.idxmax()[0]} ({monthly_listening.max():.1f} hours)",
            "Average monthly listening": f"{monthly_listening.mean():.1f} hours",
            "Monthly variation": f"{monthly_listening.std():.1f} hours standard deviation",
        }

        for stat, value in month_stats.items():
            print(f"• {stat}: {value}")


def main():
    """
    Main function to run the Spotify listening history analysis.

    Handles:
    - Initial setup and configuration
    - User interaction for file selection
    - Analysis execution
    - Report generation
    """
    print_section("Spotify Listening History Analyzer")
    print("Welcome! This tool will analyze your Spotify listening history in detail.\n")

    # Load environment configuration
    if os.path.exists(".env"):
        load_dotenv(override=True)
        print("✓ Loaded environment configuration")

    # Get directory path from user
    while True:
        dir_path = input(
            "\nEnter the directory path containing your JSON files\n(press Enter for current directory): "
        ).strip()

        if not dir_path:
            dir_path = "."

        try:
            # Convert to absolute path for clarity
            dir_path = os.path.abspath(dir_path)
            if not os.path.isdir(dir_path):
                print("❌ Invalid directory path. Please try again.")
                continue

            file_paths = list(Path(dir_path).glob("*.json"))
            if not file_paths:
                print(f"❌ No JSON files found in {dir_path}")
                if not confirm_continue("Would you like to try a different directory?"):
                    print("\n🛑 Analysis cancelled")
                    input()
                    return
                continue
            break
        except Exception as e:
            print(f"❌ Error accessing directory: {str(e)}")
            if not confirm_continue("Would you like to try again?"):
                print("\n🛑 Analysis cancelled")
                input()
                return

    # Display found files
    print(f"\n📁 Found the following data files in {dir_path}:")
    for i, path in enumerate(file_paths, 1):
        size = os.path.getsize(path)
        print(f"{i}. {path.name} ({format_size(size)})")

    if not confirm_continue("Would you like to proceed with the analysis?"):
        print("\n🛑 Analysis cancelled")
        input()
        return

    # Get API credentials from environment
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not (client_id and client_secret):
        print("\n⚠️ No Spotify API credentials found in .env file")
        print("To include recent plays, create a .env file with:")
        print("SPOTIFY_CLIENT_ID=your_client_id_here")
        print("SPOTIFY_CLIENT_SECRET=your_client_secret_here")

        if not confirm_continue("Continue without recent plays?"):
            print("\n🛑 Analysis cancelled")
            input()
            return

    # Run analysis
    try:
        print("\n🚀 Initializing analysis...")
        analyzer = SpotifyAnalyzer(file_paths, client_id, client_secret)

        analyzer.load_and_process_data()

        if confirm_continue("Would you like to generate the detailed analysis report?"):
            analyzer.generate_report()
            print(
                "\n✨ Analysis complete! Thank you for using the Spotify History Analyzer. Please star the project on GitHub if you found it useful."
            )
            input()
        else:
            print("\n🛑 Analysis cancelled")
            input()

    except KeyboardInterrupt:
        print("\n\n🛑 Analysis cancelled by user")
        input()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check your data files and try again.")
        input()
        sys.exit(0)


if __name__ == "__main__":
    main()
