import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Download necessary NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
except:
    print("Note: Make sure you have an internet connection for downloading NLTK resources")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def correct_spelling(tokens):
    corrected = []
    for word in tokens:
        blob = TextBlob(word)
        corrected.append(str(blob.correct()))
    return corrected

def main():
    # Sample text if file not found
    sample_text = "This is a sample text for tokenization. It contains multiple sentences with various words to process."
    
    try:
        with open('tech1.txt', 'r') as file:
            text = file.read()
    except FileNotFoundError:
        print("Warning: 'tech.txt' not found. Using sample text instead.")
        text = sample_text

    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    corrected_tokens = correct_spelling(tokens)

    print("Final Tokens after processing:", corrected_tokens)

if __name__ == "__main__":
    main()
