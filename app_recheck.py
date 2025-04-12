import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
from wordcloud import WordCloud
plt.style.use('ggplot')
#from streamlit.errors import StreamlitInvalidColumnSpecError
import warnings

warnings.filterwarnings("ignore")

# Load the tokenizer and model
@st.cache_resource  # Cache the model and tokenizer for better performance
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Function to calculate sentiment scores
def polarity_score_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt', truncation=True)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

# App Interface
st.title("Sentiment Analysis with RoBERTa")
st.write("This app uses the RoBERTa model to analyze sentiment from text reviews.")

# File uploader for the dataset
df=pd.DataFrame()
x=int()
uploaded_file = st.file_uploader("Upload a CSV file with reviews", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    num_input=st.number_input("enter the number of reviews")
    num_samples=int(num_input)
    df=df[:num_samples]
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())
    x=list(df.columns)

#cols=st.columns(len(x))

st.write('select the column with text reviews')
selected=str()
for j in range(len(x)):
    if st.button(str(x[j])):
        selected=x[j]
    #with cols[j]:
        #if st.button(str(x[j])):
            #selected=x[j]
        
st.write(selected)

# Process the dataset
if st.button("Analyze Sentiments"):
    st.write("Analyzing sentiments... Please wait.")

    # Analyze sentiments for each review
    res = {}
    st.write(':red[remember, RoBERTa can\'t handle very large sentences.]')
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            #text = row['Text'] #---------------------------------------------------------------
            text=selected
            myid = row['Id']   #---------------------------------------------------------------
            
            roberta = polarity_score_roberta(text)
            res[myid] = roberta
        except RuntimeError:
            st.warning(f"Broke for ID {myid}")

    # Convert results to DataFrame
    res_df = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
    res_df = res_df.merge(df, how='left')

    st.write("Sentiment Analysis Results:")
    st.dataframe(res_df.head())

    # Create plots
    st.subheader("Visualizations")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Positive Sentiment")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=res_df, x='Score', y='roberta_pos', ax=ax)
        ax.set_title("Positive Scores by Review Score")
        st.pyplot(fig)

    with col2:
        st.write("Negative Sentiment")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=res_df, x='Score', y='roberta_neg', ax=ax)
        ax.set_title("Negative Scores by Review Score")
        st.pyplot(fig)

    with col3:
        st.write("Neutral Sentiment")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=res_df, x='Score', y='roberta_neu', ax=ax)
        ax.set_title("Neutral Scores by Review Score")
        st.pyplot(fig)

st.divider()
if st.button('make wordcloud'):
    st.subheader("WordCloud")
    text = " ".join(df['Text'].astype(str))  # Combine all text for the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


st.divider()
# Input for custom text
st.subheader("Test Sentiment on 'Custom Text")
custom_text = st.text_input("Enter your text here:")

result=float()
if st.button("Analyze"):
    blob = TextBlob(custom_text)
    result = blob.sentiment.polarity
if result > 0.0:
    custom_emoji = ':blush:'
    st.success('Happy : {}'.format(custom_emoji))
elif result < 0.0:
    custom_emoji = ':disappointed:'
    st.warning('Sad : {}'.format(custom_emoji))
else:
    custom_emoji = ':confused:'
    st.info('Confused : {}'.format(custom_emoji))
    st.success("Polarity Score is: {}".format(result))

