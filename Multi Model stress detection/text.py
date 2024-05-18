import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

def detect_stress(text):
    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    # Analyze sentiment of the text
    scores = sid.polarity_scores(text)
    # Check if sentiment indicates stress
    if scores['neg'] > scores['pos']:
        return 1
    else:
        return 0
# Example text to analyze

data = pd.read_csv(r"chat.csv")
text_examples = data['text'].tolist()


stress_levels = []  # List to store stress levels

# Analyze each example text
for text in text_examples:
    stress_levels.append(detect_stress(text))

# Calculate percentage of stress level
percentage_stress_level = (sum(stress_levels) / len(stress_levels)) * 100

# Visualize stress levels
plt.bar(range(len(text_examples)), stress_levels, color='skyblue')
plt.xlabel('Text Example')
plt.ylabel('Stress Level')
plt.title('Stress Levels in Text Examples')
plt.xticks(range(len(text_examples)), range(1, len(text_examples) + 1))
plt.axhline(y=sum(stress_levels)/len(stress_levels), color='red', linestyle='--', label=f'Average Stress Level: {percentage_stress_level:.2f}%')
plt.legend()
plt.show()

print("Percentage of Stress Level:", percentage_stress_level,"%")