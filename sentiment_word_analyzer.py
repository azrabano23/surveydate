#!/usr/bin/env python3
"""
Sentiment-Focused Word Analyzer for CSRR Survey
Focuses only on words that express emotions, opinions, and sentiments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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

class SentimentWordAnalyzer:
    def __init__(self, csv_file):
        """Initialize the analyzer with the CSV file"""
        self.csv_file = csv_file
        self.df = None
        
        # Comprehensive list of sentiment words (positive and negative)
        self.sentiment_words = {
            # Positive emotions and opinions
            'excellent', 'amazing', 'wonderful', 'fantastic', 'outstanding', 'brilliant', 'superb',
            'great', 'good', 'nice', 'pleasant', 'enjoyable', 'satisfying', 'fulfilling',
            'inspiring', 'motivating', 'encouraging', 'uplifting', 'energizing',
            'interesting', 'fascinating', 'engaging', 'compelling', 'captivating',
            'informative', 'educational', 'enlightening', 'insightful', 'valuable',
            'helpful', 'useful', 'beneficial', 'productive', 'constructive',
            'organized', 'well-structured', 'professional', 'polished', 'smooth',
            'dynamic', 'energetic', 'passionate', 'enthusiastic', 'excited',
            'knowledgeable', 'expert', 'skilled', 'competent', 'qualified',
            'personable', 'friendly', 'approachable', 'welcoming', 'warm',
            'relevant', 'timely', 'current', 'important', 'significant',
            'thorough', 'comprehensive', 'detailed', 'in-depth', 'complete',
            'clear', 'understandable', 'accessible', 'straightforward',
            'innovative', 'creative', 'original', 'unique', 'fresh',
            'powerful', 'impactful', 'influential', 'meaningful', 'profound',
            
            # Negative emotions and opinions
            'terrible', 'awful', 'horrible', 'dreadful', 'abysmal', 'atrocious',
            'bad', 'poor', 'mediocre', 'subpar', 'inferior', 'disappointing',
            'frustrating', 'annoying', 'irritating', 'bothersome', 'troublesome',
            'confusing', 'unclear', 'vague', 'ambiguous', 'misleading',
            'boring', 'dull', 'tedious', 'monotonous', 'repetitive',
            'difficult', 'challenging', 'complex', 'complicated', 'overwhelming',
            'rushed', 'hurried', 'pressed', 'cramped', 'squeezed',
            'short', 'brief', 'limited', 'restricted', 'constrained',
            'long', 'extended', 'drawn-out', 'lengthy', 'protracted',
            'biased', 'one-sided', 'partial', 'prejudiced', 'unfair',
            'superficial', 'shallow', 'surface-level', 'basic', 'elementary',
            'irrelevant', 'unrelated', 'tangential', 'off-topic', 'distracting',
            'unprofessional', 'amateurish', 'sloppy', 'careless', 'negligent',
            'disorganized', 'chaotic', 'messy', 'confused', 'scattered',
            
            # Intensity and emphasis words
            'very', 'really', 'extremely', 'incredibly', 'absolutely', 'completely',
            'totally', 'entirely', 'thoroughly', 'deeply', 'profoundly',
            'slightly', 'somewhat', 'moderately', 'fairly', 'reasonably',
            'barely', 'hardly', 'scarcely', 'minimally', 'marginally',
            
            # Opinion and preference words
            'like', 'love', 'enjoy', 'appreciate', 'value', 'prefer',
            'dislike', 'hate', 'loathe', 'despise', 'abhor', 'detest',
            'want', 'need', 'desire', 'wish', 'hope', 'expect',
            'think', 'believe', 'feel', 'consider', 'regard', 'view',
            'should', 'could', 'would', 'might', 'may', 'can',
            
            # Quality assessment words
            'better', 'worse', 'best', 'worst', 'improved', 'declined',
            'adequate', 'sufficient', 'enough', 'satisfactory', 'acceptable',
            'inadequate', 'insufficient', 'lacking', 'deficient', 'unsatisfactory',
            
            # Emotional state words
            'happy', 'pleased', 'satisfied', 'content', 'grateful', 'thankful',
            'sad', 'disappointed', 'frustrated', 'angry', 'upset', 'concerned',
            'excited', 'enthusiastic', 'motivated', 'inspired', 'energized',
            'tired', 'exhausted', 'overwhelmed', 'stressed', 'anxious',
            'calm', 'relaxed', 'peaceful', 'comfortable', 'at ease',
            'nervous', 'worried', 'anxious', 'tense', 'uncomfortable',
            
            # Experience words
            'enjoyed', 'appreciated', 'valued', 'benefited', 'gained',
            'suffered', 'struggled', 'endured', 'tolerated', 'survived',
            'learned', 'discovered', 'realized', 'understood', 'grasped',
            'missed', 'overlooked', 'ignored', 'neglected', 'forgot',
            
            # Comparative words
            'more', 'less', 'better', 'worse', 'improved', 'declined',
            'similar', 'different', 'same', 'unique', 'special',
            'usual', 'unusual', 'normal', 'abnormal', 'typical',
            
            # Temporal sentiment words
            'always', 'never', 'sometimes', 'often', 'rarely', 'frequently',
            'recently', 'previously', 'currently', 'eventually', 'finally',
            
            # Specific to this survey context
            'rushed', 'hurried', 'pressed', 'cramped', 'squeezed',
            'short', 'brief', 'limited', 'restricted', 'constrained',
            'long', 'extended', 'drawn-out', 'lengthy', 'protracted',
            'adequate', 'sufficient', 'enough', 'satisfactory', 'acceptable',
            'inadequate', 'insufficient', 'lacking', 'deficient', 'unsatisfactory',
            'informative', 'educational', 'enlightening', 'insightful', 'valuable',
            'helpful', 'useful', 'beneficial', 'productive', 'constructive',
            'organized', 'well-structured', 'professional', 'polished', 'smooth',
            'dynamic', 'energetic', 'passionate', 'enthusiastic', 'excited',
            'knowledgeable', 'expert', 'skilled', 'competent', 'qualified',
            'personable', 'friendly', 'approachable', 'welcoming', 'warm',
            'relevant', 'timely', 'current', 'important', 'significant',
            'thorough', 'comprehensive', 'detailed', 'in-depth', 'complete',
            'clear', 'understandable', 'accessible', 'straightforward',
            'innovative', 'creative', 'original', 'unique', 'fresh',
            'powerful', 'impactful', 'influential', 'meaningful', 'profound'
        }
        
        # Stop words to exclude
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['event', 'events', 'csrr', 'topic', 'speaker', 'presentation', 'time', 'information', 'questions', 'nations', 'arab', 'gaza', 'latino', 'black', 'bias', 'rights', 'immigrants', 'students', 'homelands', 'democracy', 'ethnonationalism', 'professor', 'butler', 'cle', 'kyr'])
        
    def load_data(self):
        """Load the survey data"""
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
    
    def extract_sentiment_words(self, text):
        """Extract only sentiment words from text"""
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []
        
        # Tokenize
        words = word_tokenize(cleaned_text)
        
        # Keep only sentiment words and remove stop words
        sentiment_words = [word for word in words if word in self.sentiment_words and word not in self.stop_words]
        
        return sentiment_words
    
    def analyze_sentiment_words(self, column_name):
        """Analyze sentiment words in a specific column"""
        print(f"\nAnalyzing sentiment words in '{column_name}'...")
        
        all_sentiment_words = []
        for text in self.df[column_name]:
            sentiment_words = self.extract_sentiment_words(text)
            all_sentiment_words.extend(sentiment_words)
        
        # Count sentiment word frequencies
        word_counts = Counter(all_sentiment_words)
        
        # Get top sentiment words
        top_words = word_counts.most_common(20)
        
        print(f"Top 20 most frequent sentiment words in '{column_name}':")
        for word, count in top_words:
            print(f"  {word}: {count}")
        
        return word_counts, top_words
    
    def create_sentiment_wordcloud(self, word_counts, title, filename):
        """Create a word cloud visualization for sentiment words only"""
        if not word_counts:
            print(f"No sentiment words found for {title}")
            return
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=50,
            colormap='RdYlBu',  # Red-Yellow-Blue for sentiment
            relative_scaling=0.5
        ).generate_from_frequencies(word_counts)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Sentiment word cloud saved as {filename}")
    
    def categorize_sentiment_words(self, word_counts):
        """Categorize sentiment words into positive, negative, and neutral"""
        positive_words = []
        negative_words = []
        neutral_words = []
        
        # Define positive and negative word lists
        positive_list = {
            'excellent', 'amazing', 'wonderful', 'fantastic', 'outstanding', 'brilliant', 'superb',
            'great', 'good', 'nice', 'pleasant', 'enjoyable', 'satisfying', 'fulfilling',
            'inspiring', 'motivating', 'encouraging', 'uplifting', 'energizing',
            'interesting', 'fascinating', 'engaging', 'compelling', 'captivating',
            'informative', 'educational', 'enlightening', 'insightful', 'valuable',
            'helpful', 'useful', 'beneficial', 'productive', 'constructive',
            'organized', 'well-structured', 'professional', 'polished', 'smooth',
            'dynamic', 'energetic', 'passionate', 'enthusiastic', 'excited',
            'knowledgeable', 'expert', 'skilled', 'competent', 'qualified',
            'personable', 'friendly', 'approachable', 'welcoming', 'warm',
            'relevant', 'timely', 'current', 'important', 'significant',
            'thorough', 'comprehensive', 'detailed', 'in-depth', 'complete',
            'clear', 'understandable', 'accessible', 'straightforward',
            'innovative', 'creative', 'original', 'unique', 'fresh',
            'powerful', 'impactful', 'influential', 'meaningful', 'profound',
            'like', 'love', 'enjoy', 'appreciate', 'value', 'prefer',
            'want', 'need', 'desire', 'wish', 'hope', 'expect',
            'better', 'best', 'improved', 'adequate', 'sufficient', 'enough', 'satisfactory', 'acceptable',
            'happy', 'pleased', 'satisfied', 'content', 'grateful', 'thankful',
            'excited', 'enthusiastic', 'motivated', 'inspired', 'energized',
            'calm', 'relaxed', 'peaceful', 'comfortable', 'at ease',
            'enjoyed', 'appreciated', 'valued', 'benefited', 'gained',
            'learned', 'discovered', 'realized', 'understood', 'grasped',
            'more', 'similar', 'same', 'special', 'usual', 'normal', 'typical',
            'always', 'often', 'frequently', 'currently', 'finally',
            'adequate', 'sufficient', 'enough', 'satisfactory', 'acceptable',
            'informative', 'educational', 'enlightening', 'insightful', 'valuable',
            'helpful', 'useful', 'beneficial', 'productive', 'constructive',
            'organized', 'well-structured', 'professional', 'polished', 'smooth',
            'dynamic', 'energetic', 'passionate', 'enthusiastic', 'excited',
            'knowledgeable', 'expert', 'skilled', 'competent', 'qualified',
            'personable', 'friendly', 'approachable', 'welcoming', 'warm',
            'relevant', 'timely', 'current', 'important', 'significant',
            'thorough', 'comprehensive', 'detailed', 'in-depth', 'complete',
            'clear', 'understandable', 'accessible', 'straightforward',
            'innovative', 'creative', 'original', 'unique', 'fresh',
            'powerful', 'impactful', 'influential', 'meaningful', 'profound'
        }
        
        negative_list = {
            'terrible', 'awful', 'horrible', 'dreadful', 'abysmal', 'atrocious',
            'bad', 'poor', 'mediocre', 'subpar', 'inferior', 'disappointing',
            'frustrating', 'annoying', 'irritating', 'bothersome', 'troublesome',
            'confusing', 'unclear', 'vague', 'ambiguous', 'misleading',
            'boring', 'dull', 'tedious', 'monotonous', 'repetitive',
            'difficult', 'challenging', 'complex', 'complicated', 'overwhelming',
            'rushed', 'hurried', 'pressed', 'cramped', 'squeezed',
            'short', 'brief', 'limited', 'restricted', 'constrained',
            'long', 'extended', 'drawn-out', 'lengthy', 'protracted',
            'biased', 'one-sided', 'partial', 'prejudiced', 'unfair',
            'superficial', 'shallow', 'surface-level', 'basic', 'elementary',
            'irrelevant', 'unrelated', 'tangential', 'off-topic', 'distracting',
            'unprofessional', 'amateurish', 'sloppy', 'careless', 'negligent',
            'disorganized', 'chaotic', 'messy', 'confused', 'scattered',
            'dislike', 'hate', 'loathe', 'despise', 'abhor', 'detest',
            'worse', 'worst', 'declined', 'inadequate', 'insufficient', 'lacking', 'deficient', 'unsatisfactory',
            'sad', 'disappointed', 'frustrated', 'angry', 'upset', 'concerned',
            'tired', 'exhausted', 'overwhelmed', 'stressed', 'anxious',
            'nervous', 'worried', 'anxious', 'tense', 'uncomfortable',
            'suffered', 'struggled', 'endured', 'tolerated', 'survived',
            'missed', 'overlooked', 'ignored', 'neglected', 'forgot',
            'less', 'worse', 'declined', 'different', 'unique', 'unusual', 'abnormal',
            'never', 'rarely', 'previously', 'eventually',
            'rushed', 'hurried', 'pressed', 'cramped', 'squeezed',
            'short', 'brief', 'limited', 'restricted', 'constrained',
            'long', 'extended', 'drawn-out', 'lengthy', 'protracted',
            'inadequate', 'insufficient', 'lacking', 'deficient', 'unsatisfactory'
        }
        
        for word, count in word_counts.items():
            if word in positive_list:
                positive_words.append((word, count))
            elif word in negative_list:
                negative_words.append((word, count))
            else:
                neutral_words.append((word, count))
        
        return positive_words, negative_words, neutral_words
    
    def plot_sentiment_categories(self, positive_words, negative_words, neutral_words, column_name):
        """Plot sentiment words by category"""
        print(f"\nCreating sentiment category plot for '{column_name}'...")
        
        # Prepare data for plotting
        categories = ['Positive', 'Negative', 'Neutral']
        counts = [len(positive_words), len(negative_words), len(neutral_words)]
        total_words = sum(counts)
        
        # Calculate percentages
        percentages = [count/total_words*100 if total_words > 0 else 0 for count in counts]
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        colors = ['lightgreen', 'lightcoral', 'lightblue']
        wedges, texts, autotexts = ax1.pie(counts, labels=categories, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title(f'Sentiment Word Distribution - {column_name}', fontweight='bold')
        
        # Bar chart of top words in each category
        ax2.bar(categories, counts, color=colors, alpha=0.7)
        ax2.set_title(f'Sentiment Word Counts - {column_name}', fontweight='bold')
        ax2.set_ylabel('Number of Unique Words')
        
        # Add value labels on bars
        for i, count in enumerate(counts):
            ax2.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'sentiment_categories_{column_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Sentiment categories saved as sentiment_categories_{column_name.replace(' ', '_')}.png")
        
        # Print detailed breakdown
        print(f"\nSentiment Word Breakdown for '{column_name}':")
        print(f"  Positive words: {len(positive_words)} ({percentages[0]:.1f}%)")
        print(f"  Negative words: {len(negative_words)} ({percentages[1]:.1f}%)")
        print(f"  Neutral words: {len(neutral_words)} ({percentages[2]:.1f}%)")
        
        if positive_words:
            print(f"\n  Top positive words: {', '.join([word for word, _ in positive_words[:5]])}")
        if negative_words:
            print(f"  Top negative words: {', '.join([word for word, _ in negative_words[:5]])}")
    
    def run_sentiment_analysis(self):
        """Run the complete sentiment word analysis"""
        print("🎭 Starting Sentiment Word Analysis...")
        
        # Load data
        self.load_data()
        
        # Analyze text responses
        text_columns = [
            'What did you like most/least about the event?',
            'Do you have suggestions or comments to help CSRR improve future events?'
        ]
        
        for col in text_columns:
            if col in self.df.columns:
                # Analyze sentiment words
                word_counts, top_words = self.analyze_sentiment_words(col)
                
                if word_counts:
                    # Create sentiment word cloud
                    self.create_sentiment_wordcloud(word_counts, f'Sentiment Words - {col}', f'sentiment_wordcloud_{col.replace(" ", "_").replace("?", "").replace("/", "_")}.png')
                    
                    # Categorize and plot sentiment words
                    positive_words, negative_words, neutral_words = self.categorize_sentiment_words(word_counts)
                    self.plot_sentiment_categories(positive_words, negative_words, neutral_words, col)
                else:
                    print(f"No sentiment words found in '{col}'")
        
        print("\n✅ Sentiment word analysis complete!")
        print("\n📁 Generated files:")
        print("   • sentiment_wordcloud_*.png - Sentiment word clouds")
        print("   • sentiment_categories_*.png - Sentiment category breakdowns")

def main():
    """Main function to run the sentiment analysis"""
    csv_file = "Master CSRR Virtual Event Survey .csv"
    
    try:
        analyzer = SentimentWordAnalyzer(csv_file)
        analyzer.run_sentiment_analysis()
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file '{csv_file}'")
        print("Please make sure the CSV file is in the current directory.")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
