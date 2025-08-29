# CSRR Virtual Event Survey Analyzer

This program analyzes survey responses from CSRR (Center for Social Justice Research) virtual events to create idea maps, word frequency analysis, and sentiment analysis.

## Features

- **Word Frequency Analysis**: Identifies the most common words in survey responses
- **Sentiment Analysis**: Analyzes the emotional tone of text responses using VADER sentiment analysis
- **Word Clouds**: Creates visual representations of word frequencies
- **Rating Distributions**: Shows how participants rated different aspects of events
- **Comprehensive Reporting**: Generates detailed summary reports with key insights

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your CSV survey file in the same directory as the script
2. Run the analysis:

```bash
python survey_analyzer.py
```

## What the Program Does

### 1. Data Loading and Cleaning
- Loads the CSV survey data
- Cleans and preprocesses text responses
- Removes stop words and irrelevant terms

### 2. Word Frequency Analysis
- Extracts meaningful words from text responses
- Counts word frequencies
- Identifies the most common themes and topics

### 3. Sentiment Analysis
- Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis
- Calculates positive, negative, neutral, and compound sentiment scores
- Provides insights into the emotional tone of responses

### 4. Visualizations
- **Word Clouds**: Visual representation of word frequencies
- **Sentiment Charts**: Distribution of sentiment scores and components
- **Rating Distributions**: How participants rated different aspects

### 5. Summary Report
- Basic statistics (total responses, date range)
- Event analysis (which events were attended)
- Attendee demographics (first-time vs returning)
- Average ratings for different aspects
- Event length feedback analysis

## Output Files

The program generates several visualization files:

- `wordcloud_*.png` - Word frequency visualizations
- `sentiment_analysis_*.png` - Sentiment analysis charts  
- `rating_distributions.png` - Rating distribution plots

## Survey Data Structure

The program expects a CSV file with the following columns:
- Timestamp
- Which event did you attend?
- Was this your first CSRR event?
- Overall, how would you rate the event?
- Please rate the speaker(s):
- Please rate the event length:
- Please rate the organization of the event:
- Please rate the date and time of the event:
- Based on your experience, how likely are you to attend future CSRR events?
- How likely are you to recommend our events to a friend/colleague?
- What did you like most/least about the event?
- Do you have suggestions or comments to help CSRR improve future events?

## Key Insights You'll Get

1. **Most Common Themes**: What topics and themes appear most frequently in responses
2. **Sentiment Trends**: Whether responses are generally positive, negative, or neutral
3. **Rating Patterns**: How participants rate different aspects of events
4. **Event Preferences**: Which events are most popular and well-received
5. **Improvement Areas**: Common suggestions and feedback for future events

## Example Output

The program will provide console output like:

```
🚀 Starting CSRR Survey Analysis...

📊 BASIC STATISTICS:
   Total responses: 25
   Date range: 2023/04/24 1:09:36 PM AST to 2025/05/07 1:46:45 PM AST

🎯 EVENT ANALYSIS:
   Events attended:
     • Coming to Understand Latino Anti-Black Bias: 5 responses
     • Teach-In on Gaza: 12 responses
     • Hostile Homelands: 3 responses
     • Democracy and Ethnonationalism: 1 response
     • Know Your Rights for Immigrants: 3 responses

⭐ RATING ANALYSIS:
   Overall, how would you rate the event?: 9.12/10
   Please rate the speaker(s):: 9.48/10
   Please rate the organization of the event:: 9.24/10
```

## Customization

You can modify the `SurveyAnalyzer` class to:
- Add more stop words for your specific context
- Change visualization styles and colors
- Add additional analysis methods
- Customize the summary report format

## Troubleshooting

- **File not found**: Make sure the CSV file is in the same directory as the script
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **NLTK data**: The program will automatically download required NLTK data on first run
