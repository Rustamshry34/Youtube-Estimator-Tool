import streamlit as st
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import joblib
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')
import nltk
nltk.download('wordnet')


api_key = 'AIzaSyBmM-Z_PfxgXwOlnGNff4OzWCASjMIrpnw'
youtube = build('youtube', 'v3', developerKey=api_key)

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


lemmatizer = WordNetLemmatizer()

def get_channel_videos(channel_id):
    """
    Fetches videos from a YouTube channel using the YouTube Data API.
    """
    videos = []
    request = youtube.search().list(
        part='snippet',
        channelId=channel_id,
        maxResults=50,  # Adjust based on your need and quota usage
        type='video',  # Ensures we are only fetching video results
        order='date'  # Fetch recent videos first
    ).execute()

    # Iterate through items and extract video IDs and titles
    for item in request['items']:
        # Check if 'videoId' exists in 'id'
        if 'videoId' in item['id']:
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
            videos.append({'video_id': video_id, 'video_title': video_title})
        else:
            # Skip items that don't have a video ID (e.g., channels, playlists)
            continue

    return videos  # Return the list of videos

def get_video_details(video_id):
    request = youtube.videos().list(part='statistics', id=video_id).execute()
    details = request['items'][0]['statistics']
    return details

def get_comments(video_id):
    request = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=100).execute()
    comments = []
    for item in request['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)
    return comments


def extract_channel_id(youtube_url):
    """
    Extracts channel ID from a YouTube URL, handling 'channel', 'user', and 'custom' URLs.
    """
    youtube_url = youtube_url.strip()  # Strip any extra spaces or newlines

    # Case 1: Channel URL (e.g., https://www.youtube.com/channel/CHANNEL_ID)
    match = re.match(r'(https?://)?(www\.)?youtube\.com/channel/([a-zA-Z0-9_-]+)', youtube_url)
    if match:
        return match.group(3)  # The channel ID is in the third group

    # Case 2: User URL (e.g., https://www.youtube.com/user/USERNAME)
    match = re.match(r'(https?://)?(www\.)?youtube\.com/user/([a-zA-Z0-9_-]+)', youtube_url)
    if match:
        username = match.group(3)  # The username is in the third group
        return get_channel_id_by_username(username)

    # Case 3: Custom URL (e.g., https://www.youtube.com/c/CUSTOM_NAME)
    match = re.match(r'(https?://)?(www\.)?youtube\.com/c/([a-zA-Z0-9_-]+)', youtube_url)
    if match:
        custom_name = match.group(3)  # The custom name is in the third group
        return get_channel_id_by_custom_name(custom_name)

    # Case 4: Channel homepage URL (e.g., https://www.youtube.com/USERNAME)
    match = re.match(r'(https?://)?(www\.)?youtube\.com/([a-zA-Z0-9_-]+)', youtube_url)
    if match:
        homepage_name = match.group(3)
        return get_channel_id_by_custom_name(homepage_name)

    # Fallback: Invalid URL format
    raise ValueError("Invalid YouTube URL format. Please enter a valid channel URL.")

def get_channel_id_by_username(username):
    """
    Uses the YouTube API to resolve the channel ID from a YouTube 'user' URL.
    """
    request = youtube.channels().list(part='id', forUsername=username).execute()

    if request['items']:
        return request['items'][0]['id']
    else:
        raise ValueError(f"Could not find a channel for username: {username}")

def get_channel_id_by_custom_name(custom_name):
    """
    Uses the YouTube API to resolve the channel ID from a YouTube 'c' (custom name) URL.
    """
    request = youtube.search().list(part='snippet', q=custom_name, type='channel', maxResults=1).execute()

    if request['items']:
        return request['items'][0]['snippet']['channelId']
    else:
        raise ValueError(f"Could not find a channel for custom name: {custom_name}")


def analyze_sentiment(comments):
    """
    Analyzes the sentiment of a list of comments using the pre-trained sentiment analysis model.
    The text must be transformed by the vectorizer before passing to the model.
    """
    if isinstance(comments, list):
        # Vectorize the list of comments
        comment_array = vectorizer.transform(comments)
    else:
        # If a single comment is passed, wrap it in a list and then vectorize
        comment_array = vectorizer.transform([comments])
    
    # Use the model to predict the sentiment
    sentiment_scores = model.predict(comment_array)

    sentiment_mapping = {
        'positive': 1,
        'negative': 0,
        'neutral': 0.5 
    }
    
    # Convert string labels to numeric values using the mapping
    numeric_sentiment_scores = np.array([sentiment_mapping.get(score, 0) for score in sentiment_scores], dtype=np.float64)
    
    # Calculate the average sentiment score
    return np.mean(numeric_sentiment_scores) if len(numeric_sentiment_scores) > 0 else 0


def is_video_sponsored(video_title, video_description):
    """
    Determines if a video is sponsored by checking for specific keywords in the title or description.
    """
    sponsored_keywords = ['sponsored', 'ad', 'advertisement', 'paid promotion', 'partnered with', 'includes paid promotion', 'brand deal', 'paid partnership']
    
    # Combine title and description into a single string for easier keyword searching
    combined_text = f"{video_title} {video_description}".lower()
    
    # Check for keywords in title or description
    return any(keyword in combined_text for keyword in sponsored_keywords)

def get_video_description(video_id):
    """
    Fetches the description of a YouTube video using the video_id.
    """
    # Make the API request to fetch video details
    request = youtube.videos().list(
        part='snippet',  # We want details from the snippet part, which contains description
        id=video_id
    ).execute()

    # Extract the description if available
    if request['items']:
        return request['items'][0]['snippet']['description']
    else:
        return None
    


def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stopwords.words('english')]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


def evaluate_channel(channel_id):
    """
    Evaluates the YouTube channel by comparing sentiment scores for sponsored and unsponsored videos.
    """
    videos = get_channel_videos(channel_id)  # This returns a list of videos
    sponsored_sentiments = []
    unsponsored_sentiments = []

    for video in videos:
        video_id = video['video_id']
        video_title = video['video_title']
        video_description = get_video_description(video_id)
        

        # Get video details and comments
        details = get_video_details(video_id)
        comments = get_comments(video_id)
        
        #comments = comments.apply(clean_text)
        comments = [clean_text(comment) for comment in comments]

        # Perform sentiment analysis on the comments
        sentiment_score = analyze_sentiment(comments)
        
        # Naive check: if "sponsored" or "ad" appears in title, assume it's sponsored

        #if 'sponsored' in video_title.lower() or 'ad' in video_title.lower():
            #sponsored_sentiments.append(sentiment_score)
        #else:
            #unsponsored_sentiments.append(sentiment_score)

        if is_video_sponsored(video_title, video_description,):
            sponsored_sentiments.append(analyze_sentiment(get_comments(video_id)))
        else:
            unsponsored_sentiments.append(analyze_sentiment(get_comments(video_id)))

          
    # Calculate average sentiment scores
    #avg_sponsored_sentiment = sum(sponsored_sentiments) / len(sponsored_sentiments) if sponsored_sentiments else 0
    #avg_unsponsored_sentiment = sum(unsponsored_sentiments) / len(unsponsored_sentiments) if unsponsored_sentiments else 0

    avg_sponsored_sentiment = np.mean(sponsored_sentiments) if sponsored_sentiments else 0
    avg_unsponsored_sentiment = np.mean(unsponsored_sentiments) if unsponsored_sentiments else 0

    return avg_sponsored_sentiment, avg_unsponsored_sentiment


st.title("YouTube Partner Estimator Tool")

channel_url = st.text_input("Enter YouTube Channel URL")
if st.button("Evaluate"):
    channel_id = extract_channel_id(channel_url)  # Define function to extract channel ID from URL
    avg_sponsored_sentiment, avg_unsponsored_sentiment = evaluate_channel(channel_id)
    
    st.write(f"Average Sponsored Sentiment Score: {avg_sponsored_sentiment}")
    st.write(f"Average Unsponsored Sentiment Score: {avg_unsponsored_sentiment}")
    
    # Example estimation logic: if sponsored sentiment is better, they might be a good partner
    if avg_sponsored_sentiment > avg_unsponsored_sentiment:
        st.success("This YouTuber is a potential good partner for sponsored videos!")
    else:
        st.warning("This YouTuber may not be a good fit for sponsored videos.")

