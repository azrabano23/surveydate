#!/usr/bin/env python3
"""
Comparison script to show the difference between general word frequency 
and sentiment-focused word analysis
"""

import pandas as pd
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def analyze_comparison():
    """Compare general word frequency vs sentiment word frequency"""
    
    # Load data
    df = pd.read_csv("Master CSRR Virtual Event Survey .csv")
    column_name = 'What did you like most/least about the event?'
    
    # General stop words
    general_stop_words = set(stopwords.words('english'))
    general_stop_words.update(['event', 'events', 'csrr', 'would', 'could', 'should', 'like', 'liked', 'good', 'great', 'very', 'really', 'much', 'more', 'time', 'topic', 'speaker', 'presentation'])
    
    # Sentiment words (same as in sentiment analyzer)
    sentiment_words = {
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
        'very', 'really', 'extremely', 'incredibly', 'absolutely', 'completely',
        'totally', 'entirely', 'thoroughly', 'deeply', 'profoundly',
        'slightly', 'somewhat', 'moderately', 'fairly', 'reasonably',
        'barely', 'hardly', 'scarcely', 'minimally', 'marginally',
        'like', 'love', 'enjoy', 'appreciate', 'value', 'prefer',
        'dislike', 'hate', 'loathe', 'despise', 'abhor', 'detest',
        'want', 'need', 'desire', 'wish', 'hope', 'expect',
        'think', 'believe', 'feel', 'consider', 'regard', 'view',
        'should', 'could', 'would', 'might', 'may', 'can',
        'better', 'worse', 'best', 'worst', 'improved', 'declined',
        'adequate', 'sufficient', 'enough', 'satisfactory', 'acceptable',
        'inadequate', 'insufficient', 'lacking', 'deficient', 'unsatisfactory',
        'happy', 'pleased', 'satisfied', 'content', 'grateful', 'thankful',
        'sad', 'disappointed', 'frustrated', 'angry', 'upset', 'concerned',
        'excited', 'enthusiastic', 'motivated', 'inspired', 'energized',
        'tired', 'exhausted', 'overwhelmed', 'stressed', 'anxious',
        'calm', 'relaxed', 'peaceful', 'comfortable', 'at ease',
        'nervous', 'worried', 'anxious', 'tense', 'uncomfortable',
        'enjoyed', 'appreciated', 'valued', 'benefited', 'gained',
        'suffered', 'struggled', 'endured', 'tolerated', 'survived',
        'learned', 'discovered', 'realized', 'understood', 'grasped',
        'missed', 'overlooked', 'ignored', 'neglected', 'forgot',
        'more', 'less', 'better', 'worse', 'improved', 'declined',
        'similar', 'different', 'same', 'unique', 'special',
        'usual', 'unusual', 'normal', 'abnormal', 'typical',
        'always', 'never', 'sometimes', 'often', 'rarely', 'frequently',
        'recently', 'previously', 'currently', 'eventually', 'finally'
    }
    
    # Sentiment stop words (exclude factual terms)
    sentiment_stop_words = set(stopwords.words('english'))
    sentiment_stop_words.update(['event', 'events', 'csrr', 'topic', 'speaker', 'presentation', 'time', 'information', 'questions', 'nations', 'arab', 'gaza', 'latino', 'black', 'bias', 'rights', 'immigrants', 'students', 'homelands', 'democracy', 'ethnonationalism', 'professor', 'butler', 'cle', 'kyr'])
    
    def clean_text(text):
        if pd.isna(text) or text == '':
            return ''
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_general_words(text):
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return []
        words = word_tokenize(cleaned_text)
        words = [word for word in words if word not in general_stop_words and len(word) > 2]
        return words
    
    def extract_sentiment_words(text):
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return []
        words = word_tokenize(cleaned_text)
        sentiment_words_only = [word for word in words if word in sentiment_words and word not in sentiment_stop_words]
        return sentiment_words_only
    
    # Analyze both approaches
    all_general_words = []
    all_sentiment_words = []
    
    for text in df[column_name]:
        general_words = extract_general_words(text)
        sentiment_words_only = extract_sentiment_words(text)
        all_general_words.extend(general_words)
        all_sentiment_words.extend(sentiment_words_only)
    
    general_counts = Counter(all_general_words)
    sentiment_counts = Counter(all_sentiment_words)
    
    print("="*80)
    print("COMPARISON: GENERAL WORD FREQUENCY vs SENTIMENT WORD FREQUENCY")
    print("="*80)
    
    print(f"\n📊 ANALYSIS OF: '{column_name}'")
    print(f"   Total responses analyzed: {len(df)}")
    
    print(f"\n🔤 GENERAL WORD FREQUENCY (Top 15):")
    print("   (Includes factual terms, topics, and descriptive words)")
    for word, count in general_counts.most_common(15):
        print(f"     {word}: {count}")
    
    print(f"\n🎭 SENTIMENT WORD FREQUENCY (Top 15):")
    print("   (Only emotional, opinion, and sentiment words)")
    for word, count in sentiment_counts.most_common(15):
        print(f"     {word}: {count}")
    
    print(f"\n📈 COMPARISON INSIGHTS:")
    print(f"   • General words found: {len(general_counts)} unique words")
    print(f"   • Sentiment words found: {len(sentiment_counts)} unique words")
    print(f"   • Sentiment words represent {len(sentiment_counts)/len(general_counts)*100:.1f}% of all words")
    
    # Show examples of what was filtered out
    filtered_out = [word for word in general_counts.keys() if word not in sentiment_counts.keys()]
    print(f"\n🚫 EXAMPLES OF WORDS FILTERED OUT (factual/descriptive terms):")
    print(f"   {', '.join(filtered_out[:10])}...")
    
    print(f"\n✅ SENTIMENT WORDS FOCUS ON:")
    sentiment_examples = list(sentiment_counts.keys())[:10]
    print(f"   {', '.join(sentiment_examples)}...")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_comparison()
