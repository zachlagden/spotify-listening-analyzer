# Spotify Listening History Analyzer

A comprehensive tool for analyzing your Spotify listening history data. This tool processes your exported Spotify data and optionally connects to the Spotify API to provide detailed insights into your music listening patterns.

## Features

- 📊 Comprehensive listening statistics
- 🎵 Detailed track analysis
- 👥 Artist listening patterns
- 💿 Album play statistics
- ⏰ Temporal listening patterns
- 📈 Advanced metrics and trends
- 🔄 Recent plays integration via Spotify API
- 📤 Export capabilities for processed data

## Getting Started

### Option 1: Using the Executable (Recommended for most users)

1. Naivgate to the latest release from [GitHub Releases](https://github.com/zachlagden/spotify-listening-analyzer/releases/tag/v1.0.0)
2. Download the `spotify_analyzer.exe` file.<br>
   `Please note: Windows may flag the executable as a security risk. You can bypass this by clicking "More Info" and then "Run Anyway". This is because the executable is not signed.`
3. Create a `.env` file in the same folder as the executable with your Spotify API credentials (if using API features):

```plaintext
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

1. Double-click the executable to run the analyzer

### Option 2: Running from Source Code

#### Prerequisites

Required Python Libraries:

```bash
pandas
spotipy
tqdm
python-dotenv
```

#### Installation

1. Clone this repository:

```bash
git clone https://github.com/zachlagden/spotify-listening-analyzer.git
cd spotify-listening-analyzer
```

1. Install required packages:

```bash
pip install -r requirements.txt
```

1. Run the analysis:

```bash
python spotify_analyzer.py
```

### For All Users: Exporting Your Spotify Data

1. Go to your [Spotify Account Privacy Settings](https://www.spotify.com/account/privacy/)
2. Request your data export (Extended streaming history)
3. Wait for the email from Spotify (can take up to 30 days)
4. Download and extract the JSON files

### (Optional) Setting Up Spotify API Access

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/create)
2. Create a new application
3. Get your Client ID and Client Secret
4. Create a `.env` file as described above

## Analysis Features

### Overall Statistics

- Total listening time
- Daily averages
- Unique tracks/artists/albums
- Active listening days
- Weekend vs. weekday patterns

### Artist Analysis

- Top artists by listening time
- Artist-specific statistics
- Listening patterns per artist
- First/last played dates

### Track Analysis

- Most played tracks
- Play counts and durations
- Listening patterns
- Track popularity over time

### Album Analysis

- Most played albums
- Album completion rates
- Listening patterns
- Album lifecycle information

### Temporal Patterns

- Time of day distribution
- Day of week patterns
- Monthly listening trends
- Seasonal preferences

### Advanced Metrics

- Listening consistency scores
- Listening streaks
- Daily statistics
- Activity patterns

## Output Examples

```plaintext
📊 Overall Statistics:
• Total listening time: 1,234.5 hours (51.4 days)
• Daily average: 127.3 minutes
• Unique tracks: 3,456
• Active listening days: 280 (76.7% of period)

🎸 Top Artist Example:
• Artist Name
  - Total Time: 45.6 hours
  - Tracks: 123 unique across 8 albums
  - Weekend Listening: 65.4% of plays
```

## Privacy & Security

- All data is processed locally
- No data is sent to external servers
- API credentials are stored locally in `.env`
- Original data files are never modified

## Troubleshooting

### Common Issues

1. **Missing JSON Files**
   - Ensure you've extracted all files from the Spotify data export
   - Check file permissions

2. **API Connection Issues**
   - Verify your `.env` file configuration
   - Check your internet connection
   - Ensure API credentials are correct

3. **Processing Errors**
   - Check JSON file formatting
   - Ensure sufficient disk space
   - For executable users: Make sure the `.env` file is in the same directory as the exe

### Getting Help

- Create an issue in the repository
- Check existing issues for solutions
- Include error messages and system details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to Spotify for providing user data export
- Built with Python and various open-source libraries
- Inspired by the music analysis community

---
Created by Zachariah Michael Lagden
© 2024 - MIT License
