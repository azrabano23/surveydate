#!/usr/bin/env python3
"""
CSRR Virtual Event Survey Analyzer
Analyzes survey responses to create idea maps and sentiment analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SurveyAnalyzer:
    def __init__(self, csv_file):
        """Initialize the analyzer with the CSV file"""
        self.csv_file = csv_file
        self.df = None
        self.stop_words = set(stopwords.words('english'))
        # Add custom stop words for this survey context
        self.stop_words.update(['event', 'events', 'csrr', 'would', 'could', 'should', 'like', 'liked', 'good', 'great', 'very', 'really', 'much', 'more', 'time', 'topic', 'speaker', 'presentation'])
        
    def load_data(self):
        """Load and clean the survey data"""
        print("Loading survey data...")
        self.df = pd.read_csv(self.csv_file)
        print(f"Loaded {len(self.df)} survey responses")
        return self.df
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_words(self, text):
        """Extract meaningful words from text"""
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []
        
        # Tokenize
        words = word_tokenize(cleaned_text)
        
        # Remove stop words and short words
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return words
    
    def analyze_word_frequency(self, column_name):
        """Analyze word frequency in a specific column"""
        print(f"\nAnalyzing word frequency in '{column_name}'...")
        
        all_words = []
        for text in self.df[column_name]:
            words = self.extract_words(text)
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Get top words
        top_words = word_counts.most_common(20)
        
        print(f"Top 20 most frequent words in '{column_name}':")
        for word, count in top_words:
            print(f"  {word}: {count}")
        
        return word_counts, top_words
    
    def analyze_sentiment(self, column_name):
        """Analyze sentiment in text responses"""
        print(f"\nAnalyzing sentiment in '{column_name}'...")
        
        sia = SentimentIntensityAnalyzer()
        sentiments = []
        
        for text in self.df[column_name]:
            if pd.notna(text) and str(text).strip():
                sentiment_scores = sia.polarity_scores(str(text))
                sentiments.append(sentiment_scores)
            else:
                sentiments.append({'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0})
        
        # Add sentiment columns to dataframe
        self.df[f'{column_name}_compound'] = [s['compound'] for s in sentiments]
        self.df[f'{column_name}_positive'] = [s['pos'] for s in sentiments]
        self.df[f'{column_name}_negative'] = [s['neg'] for s in sentiments]
        self.df[f'{column_name}_neutral'] = [s['neu'] for s in sentiments]
        
        # Calculate average sentiments
        avg_compound = np.mean([s['compound'] for s in sentiments])
        avg_positive = np.mean([s['pos'] for s in sentiments])
        avg_negative = np.mean([s['neg'] for s in sentiments])
        avg_neutral = np.mean([s['neu'] for s in sentiments])
        
        print(f"Average sentiment scores for '{column_name}':")
        print(f"  Compound: {avg_compound:.3f}")
        print(f"  Positive: {avg_positive:.3f}")
        print(f"  Negative: {avg_negative:.3f}")
        print(f"  Neutral: {avg_neutral:.3f}")
        
        return sentiments
    
    def create_wordcloud(self, word_counts, title, filename):
        """Create a word cloud visualization"""
        if not word_counts:
            print(f"No words found for {title}")
            return
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate_from_frequencies(word_counts)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Word cloud saved as {filename}")
    
    def plot_rating_distributions(self):
        """Plot distributions of numerical ratings"""
        print("\nCreating rating distribution plots...")
        
        # Identify rating columns (columns with numeric ratings)
        rating_columns = []
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64'] and col != 'Timestamp':
                # Check if values are mostly between 1-10 (typical rating scale)
                unique_vals = self.df[col].dropna().unique()
                if len(unique_vals) <= 10 and all(1 <= val <= 10 for val in unique_vals if pd.notna(val)):
                    rating_columns.append(col)
        
        if not rating_columns:
            print("No rating columns found")
            return
        
        # Create subplots for each rating
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(rating_columns[:6]):  # Limit to 6 plots
            if i < len(axes):
                ax = axes[i]
                self.df[col].value_counts().sort_index().plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title(f'{col} Distribution', fontweight='bold')
                ax.set_xlabel('Rating')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(rating_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('rating_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Rating distributions saved as rating_distributions.png")
    
    def plot_sentiment_analysis(self, column_name):
        """Plot sentiment analysis results"""
        print(f"\nCreating sentiment analysis plot for '{column_name}'...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sentiment distribution
        compound_scores = self.df[f'{column_name}_compound']
        ax1.hist(compound_scores, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
        ax1.set_title(f'Sentiment Distribution - {column_name}', fontweight='bold')
        ax1.set_xlabel('Compound Sentiment Score')
        ax1.set_ylabel('Frequency')
        ax1.axvline(compound_scores.mean(), color='red', linestyle='--', label=f'Mean: {compound_scores.mean():.3f}')
        ax1.legend()
        
        # Sentiment components
        sentiment_components = ['positive', 'negative', 'neutral']
        avg_sentiments = [
            self.df[f'{column_name}_positive'].mean(),
            self.df[f'{column_name}_negative'].mean(),
            self.df[f'{column_name}_neutral'].mean()
        ]
        
        colors = ['green', 'red', 'gray']
        bars = ax2.bar(sentiment_components, avg_sentiments, color=colors, alpha=0.7)
        ax2.set_title(f'Average Sentiment Components - {column_name}', fontweight='bold')
        ax2.set_ylabel('Average Score')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_sentiments):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'sentiment_analysis_{column_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Sentiment analysis saved as sentiment_analysis_{column_name.replace(' ', '_')}.png")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("CSRR VIRTUAL EVENT SURVEY ANALYSIS REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   Total responses: {len(self.df)}")
        print(f"   Date range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}")
        
        # Event analysis
        print(f"\n🎯 EVENT ANALYSIS:")
        event_counts = self.df['Which event did you attend?'].value_counts()
        print("   Events attended:")
        for event, count in event_counts.items():
            print(f"     • {event}: {count} responses")
        
        # First-time attendees
        first_time = self.df['Was this your first CSRR event?'].value_counts()
        print(f"\n   First-time attendees: {first_time.get('Yes', 0)} ({first_time.get('Yes', 0)/len(self.df)*100:.1f}%)")
        print(f"   Returning attendees: {first_time.get('No', 0)} ({first_time.get('No', 0)/len(self.df)*100:.1f}%)")
        
        # Overall ratings
        print(f"\n⭐ RATING ANALYSIS:")
        rating_cols = ['Overall, how would you rate the event?', 'Please rate the speaker(s):', 
                      'Please rate the organization of the event:', 'Please rate the date and time of the event:',
                      'Based on your experience, how likely are you to attend future CSRR events?',
                      'How likely are you to recommend our events to a friend/colleague?']
        
        for col in rating_cols:
            if col in self.df.columns:
                avg_rating = self.df[col].mean()
                print(f"   {col}: {avg_rating:.2f}/10")
        
        # Event length feedback
        print(f"\n⏰ EVENT LENGTH FEEDBACK:")
        length_feedback = self.df['Please rate the event length:'].value_counts()
        for feedback, count in length_feedback.items():
            print(f"   • {feedback}: {count} responses")
        
        print("\n" + "="*60)
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting CSRR Survey Analysis...")
        
        # Load data
        self.load_data()
        
        # Generate summary report
        self.generate_summary_report()
        
        # Analyze text responses
        text_columns = [
            'What did you like most/least about the event?',
            'Do you have suggestions or comments to help CSRR improve future events?'
        ]
        
        for col in text_columns:
            if col in self.df.columns:
                # Word frequency analysis
                word_counts, top_words = self.analyze_word_frequency(col)
                
                # Create word cloud
                self.create_wordcloud(word_counts, f'Word Cloud - {col}', f'wordcloud_{col.replace(" ", "_").replace("?", "").replace("/", "_")}.png')
                
                # Sentiment analysis
                sentiments = self.analyze_sentiment(col)
                self.plot_sentiment_analysis(col)
        
        # Plot rating distributions
        self.plot_rating_distributions()
        
        print("\n✅ Analysis complete! Check the generated visualizations and console output for insights.")
        print("\n📁 Generated files:")
        print("   • wordcloud_*.png - Word frequency visualizations")
        print("   • sentiment_analysis_*.png - Sentiment analysis charts")
        print("   • rating_distributions.png - Rating distribution plots")

def main():
    """Main function to run the analysis"""
    csv_file = "Master CSRR Virtual Event Survey .csv"
    
    try:
        analyzer = SurveyAnalyzer(csv_file)
        analyzer.run_complete_analysis()
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file '{csv_file}'")
        print("Please make sure the CSV file is in the current directory.")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
