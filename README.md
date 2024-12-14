# Spotify Listening History Analyzer

A comprehensive Python tool for analyzing your Spotify listening history data. This script processes your exported Spotify data and optionally connects to the Spotify API to provide detailed insights into your music listening patterns.

## Features

- 📊 Comprehensive listening statistics
- 🎵 Detailed track analysis
- 👥 Artist listening patterns
- 💿 Album play statistics
- ⏰ Temporal listening patterns
- 📈 Advanced metrics and trends
- 🔄 Recent plays integration via Spotify API
- 📤 Export capabilities for processed data

## Prerequisites

### Required Python Libraries

```bash
pandas
spotipy
tqdm
python-dotenv
```

### Installation

1. Clone this repository:

```bash
git clone https://github.com/zachlagden/spotify-listening-analyzer.git
cd spotify-listening-analyzer
```

1. Install required packages:

```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Export Your Spotify Data

1. Go to your [Spotify Account Privacy Settings](https://www.spotify.com/account/privacy/)
2. Request your data export (Extended streaming history)
3. Wait for the email from Spotify (can take up to 30 days)
4. Download and extract the JSON files

### 2. (Optional) Set Up Spotify API Access

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/create)
2. Create a new application
3. Get your Client ID and Client Secret
4. Create a `.env` file in the project directory:

```plaintext
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

### 3. Run the Analysis

```bash
python spotify_analyzer.py
```

Follow the prompts to:

1. Select your data directory
2. Choose gap-filling options for recent plays
3. Generate comprehensive analysis

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

## Data Processing

The analyzer handles:

- Multiple JSON file processing
- Duplicate removal
- Timestamp processing
- Data validation
- Recent plays integration
- Data export capabilities

## Privacy & Security

- All data is processed locally
- No data is sent to external servers
- API credentials are stored locally in `.env`
- Original data files are never modified

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
   - Verify Python environment setup

### Getting Help

- Create an issue in the repository
- Check existing issues for solutions
- Include error messages and system details

---
Created by Zachariah Michael Lagden
© 2024 - MIT License
