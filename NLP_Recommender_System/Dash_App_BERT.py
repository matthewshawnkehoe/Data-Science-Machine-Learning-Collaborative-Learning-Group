# Import required libraries
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from dash import dcc, html, Dash
from dash.dependencies import Input, Output
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords from nltk
nltk.download(['stopwords', 'wordnet'])
nltk.download('punkt')

# Read the travel data
travel_df = pd.read_csv('~/dev/Data_Analysis/Data/all_things_to_do.csv')

def clean_text_column(df, column_name):
    """
    Clean the specified text column in the DataFrame using NLTK for tokenization,
    stopword removal, lemmatization, and punctuation removal.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the text column.
    - column_name (str): Name of the text column to be cleaned.

    Returns:
    None
    """
    # Ensure the specified column exists in the DataFrame
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return

    # Define NLTK objects for stop words and lemmatization
    stop_words_ = set(stopwords.words('english'))
    wn = WordNetLemmatizer()

    def black_txt(token):
        # Check if the token is not a stop word, not a punctuation, and has a length greater than 2
        return token not in stop_words_ and token not in list(string.punctuation) and len(token) > 2

    def clean_txt(text):
        # Remove apostrophes, digits, non-word characters, and replace 'nbsp'
        text = re.sub("'", "", text)
        text = re.sub("(\\d|\\W)+", " ", text)
        text = text.replace("nbsp", "")

        # Tokenize, lemmatize, and filter based on defined conditions
        clean_text = [wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]

        return " ".join(clean_text)

    # Apply the cleaning function to the specified column
    df[column_name] = df[column_name].apply(clean_txt)

# Apply the clean text function to the "Text"
clean_text_column(travel_df, 'Text')

# Load the preprocessor and encoder for BERT
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1", trainable=True)

def get_bert_embeddings(text, preprocessor, encoder):
    """
    Get BERT embeddings for a given text using preprocessor and encoder.

    Parameters:
    - text (str): The input text for which embeddings are needed.
    - preprocessor: The preprocessor layer for BERT.
    - encoder: The encoder layer for BERT.

    Returns:
    - np.array: BERT embeddings for the input text.
    """
    text_input = tf.constant([text])
    encoder_inputs = preprocessor(text_input)
    outputs = encoder(encoder_inputs)
    return outputs['pooled_output']

# Compute BERT embeddings for all texts in the 'Text' column
travel_df['encodings'] = travel_df['Text'].apply(lambda x: get_bert_embeddings(x, preprocessor, encoder))

# Create a function to compute similarity scores
def compute_similarity_score(query_encoding, encodings):
    return cosine_similarity([np.squeeze(encodings)], [query_encoding])[0][0]
    
# Preprocess text
def preprocess_text(text):
    """
    Preprocesses the input text by converting to lowercase and removing non-alphanumeric characters.

    Parameters:
    - text (str): The input text to be preprocessed.

    Returns:
    - str: Preprocessed text.
    """
    text = text.lower()
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    return text

# Define the image sources for each location
location_images = {
    'Moscow, Russia': '/assets/moscow.jpg',
    'Tokyo, Japan': '/assets/tokyo.jpg',
    'South Korea, East Asia': '/assets/south_korea.jpg',
    'Norway, Europe': '/assets/norway.jpg',
    'New Zealand, Pacific Ocean': '/assets/new_zealand.jpg',
    'Bali, Indonesia': '/assets/bali.jpg',
    'Rome, Italy': '/assets/rome.jpg',
    'Hokkaido, Japan': '/assets/hokkaido.jpg',
    'New York City, New York': '/assets/new_york_city.jpg',
    'Paris, France': '/assets/paris.jpg',
    'Bangkok, Tahiland': '/assets/bangkok.jpg',
    'Prague, Czech Republic': '/assets/prague.jpg',
    'Rajasthan, India': '/assets/rajasthan.jpg',
    'Singapore, Asia': '/assets/singapore.jpg',
    'Melbourne, Austrailia': '/assets/melbourne.jpg',
    'Berlin, Germany': '/assets/berlin.jpg',
    'Mexico City, Mexico': '/assets/mexico_city.jpg',
    'Las Vegas, Nevada': '/assets/las_vegas.jpg',
    'Stockholm, Sweden': '/assets/stockholm.jpg',
    'Bahamas, Atlantic Ocean': '/assets/bahamas.jpg',
    'Bahai, Brazil': '/assets/bahai.jpg',
    'Amsterdam, Netherlands': '/assets/amsterdam.jpg',
    'Hong Kong, China': '/assets/hong_kong.jpg',
    'Honolulu, Hawaii': '/assets/honolulu.jpg',
    'San Franscisco, California': '/assets/san_francisco.jpg',
    'Sydney, Austrailia': '/assets/sydney.jpg',
    'Toronto, Canada': '/assets/toronto.jpg',
    'Swiss Alps, Switzerland': '/assets/swiss_alps.jpg',
    'Rio de Janeiro, Brazil': '/assets/rio_de_janeiro.jpg',
    'Atacama Desert, Chile': '/assets/atacama_desert.jpg'
}

# Create the Dash app layout
def create_app_layout():
    """
    Create the layout for the Dash app.
    """
    return html.Div(children=[
        html.H1(children='Wanderlust Wizard - Your Travel Advisor Companion', style={'textAlign': 'center', 'margin-top': '5%', 'color': '#3366cc'}),
        html.H4(children='Craft your dream journey by sharing your ideal vacation activities.', style={'textAlign': 'center', 'color': '#3366cc'}),
        html.Br(),
        html.Div([
            html.Span("What's your dream vacation activity? Tell us, and let's create your perfect adventure:  ", style={'color': '#3366cc'}),
            dcc.Input(id='my-input', value='', type='text',
                      placeholder='e.g., Sailing under the Northern Lights', style={'width': '65%', 'color': '#3366cc'}),
        ], style={'margin-left': '10%', 'margin-right': '10%'}),
        html.Hr(),
        html.H5(children='Your Personalized Travel Recommendation Awaits:', style={'textAlign': 'center', 'color': '#3366cc'}),
        html.H4(id='my-output', style={'textAlign': 'center', 'color': '#3366cc'}),
        html.Br(),
        html.Img(id='image', style={'width': '50%', 'margin': '0 auto', 'display': 'block'}),
        html.Hr(),
        html.H5(children='Embark on a Journey of Discovery', style={'margin-left': '10%', 'margin-right': '10%', 'color': '#3366cc'}),
        html.Div(children="Join us on a journey fueled by data from over 50,000 attractions across 30 locations on Trip Advisor. Our state-of-the-art BERT-based similarity scoring ensures personalized recommendations for your dream adventure, making every exploration unforgettable. Our advanced algorithm is dedicated to crafting personalized recommendations, ensuring that every adventure is uniquely yours, leaving a lasting impression with each step you take.",
                 style={'margin-left': '10%', 'margin-right': '10%', 'color': '#3366cc'}),
    ], style={'background-color': '#f2f2f2'})  # Set the default background color

# Run the Dash app
def run_travel_app():
    """
    Run the Dash app for the travel recommendation system.
    """
    # Initialize the Dash app
    app = Dash(__name__)

    # Set external stylesheets if needed
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = Dash(__name__, external_stylesheets=external_stylesheets)

    # Set the app layout
    app.layout = create_app_layout()

    # Set the app callbacks
    @app.callback(
        Output(component_id='my-output', component_property='children'),
        Input(component_id='my-input', component_property='value')
    )
    def update_output_div(input_value):
        if not input_value:  # Check if input is empty
            return "Please enter some text to get a recommendation"
       
        # Compute BERT embeddings for the input text
        input_value = preprocess_text(input_value)
        query_encoding = get_bert_embeddings(input_value, preprocessor, encoder)
        query_encoding = tf.squeeze(query_encoding, axis=0)  # Squeeze the batch dimension
       
        # Convert BERT embeddings in DataFrame to 2D array
        encodings = np.stack(travel_df['encodings'].apply(lambda x: np.array(x)).values)
        encodings = np.squeeze(encodings, axis=1)  # Squeeze extra dimension
       
        # Compute similarity scores between input text and all locations
        similarity_scores = cosine_similarity([query_encoding.numpy()], encodings)[0]  # Compute cosine similarity
        top_location_index = np.argmax(similarity_scores)
        top_location = travel_df.loc[top_location_index, 'Location']
       
        return f"Explore {top_location}!"
    
    @app.callback(
        Output(component_id='image', component_property='src'),
        Input('my-output', 'children'))
    def update_image(input_value):
        if not input_value:  # Check if input is empty
            return "Please enter some text to get a recommendation", ''

        # Get the predicted location
        input_value = preprocess_text(input_value)
        query_encoding = get_bert_embeddings(input_value, preprocessor, encoder)
        query_encoding = tf.squeeze(query_encoding, axis=0)  # Squeeze the batch dimension
       
        # Convert BERT embeddings in DataFrame to 2D array
        encodings = np.stack(travel_df['encodings'].apply(lambda x: np.array(x)).values)
        encodings = np.squeeze(encodings, axis=1)  # Squeeze extra dimension
       
        # Compute similarity scores between input text and all locations
        similarity_scores = cosine_similarity([query_encoding.numpy()], encodings)[0]  # Compute cosine similarity
        top_location_index = np.argmax(similarity_scores)
        top_location = travel_df.loc[top_location_index, 'Location']
       
        # Set the image source based on the predicted location
        image_src = location_images.get(top_location, '/assets/atacama_desert.jpg')
        return image_src

    # Run the app
    if __name__ == '__main__':
        app.run_server(debug=True, host='localhost', port=8067)

# Run the Dash app
run_travel_app()
