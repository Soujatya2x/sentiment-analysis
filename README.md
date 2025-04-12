This streamlit based web application takes csv dataset from user and produces sentiment analysis on the text reviews. 
this application employs **Roberta** model for measuring sentiments and intensity (positive/negative/neutral). 
User can enter number of samples to check the sentiments for. then the app categorizes the text data into 3 polarities. 
after that it generates barpplots for each of the intensity polarities, creates **wordcloud** to check the most encountering words in the data
and finally it can also predict sentiment intensity of a sentence given by user instantaneously.

improvements will be made on the application like- 
1. making it more generalized for any dataset
2. allowing user to review the features and select the output columns interactively.
