
from cfg import plt, sns

from cfg import re
from cfg import string
from cfg import BeautifulSoup
from cfg import stopwords
from cfg import WordNetLemmatizer
from cfg import contractions
from cfg import np

# Load stopwords
stop = set(stopwords.words('english'))

def expand_contractions(text):
    return contractions.fix(text)

# Function to clean data
def preprocess_text(text):
    wl = WordNetLemmatizer()
    
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    
    # Expand contractions
    text = expand_contractions(text)
    
    # Remove emojis
    emoji_clean = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)
    text = emoji_clean.sub(r'', text)
    
    # Add space after full stop
    text = re.sub(r"\.(?=\S)", '. ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove punctuation and lowercase the text
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    
    # Lemmatize and remove stopwords
    text = " ".join([wl.lemmatize(word) for word in text.split() if word not in stop and word.isalpha()])
    
    return text
    
def visual(df):
    words_len = df['review'].str.split().map(lambda x: len(x) )
    df_temp = df. copy ()
    df_temp ['words length'] = words_len

    hist_positive = sns. displot (
        data = df_temp [df_temp['sentiment'] == 'positive'] ,
        x="words length", hue="sentiment", kde=True , height =7 , aspect =1.1 , legend = False
        ) .set ( title ='Words in positive reviews')
    plt.show (hist_positive)

    hist_negative = sns. displot (
        data = df_temp[df_temp ['sentiment'] == 'negative'] ,
        x = "words length", hue="sentiment", kde=True , height =7 , aspect =1.1 , legend = False , palette =['red']
    ) .set ( title ='Words in negative reviews')
    plt.show( hist_negative )
    plt.figure( figsize =(7 ,7.1) )

    kernel_distibution_number_words_plot = sns. kdeplot (
        data = df_temp , x="words length", hue="sentiment", fill =True , palette =[sns.
        color_palette()[0] ,'red']
    ).set(title ='Words in reviews')

    plt.legend(title ='sentiment', labels =['negative','positive'])
    plt.show(kernel_distibution_number_words_plot)
  
def pie_chart(df):  
    def func (pct , allvalues ) :
        absolute = int( pct / 100.* np. sum( allvalues ) )
        return "{:.1f}%\n({:d})". format (pct , absolute )

    freq_pos = len(df[df['sentiment'] == 'positive'])
    freq_neg = len(df[df['sentiment'] == 'negative'])
    data = [ freq_pos , freq_neg ]
    labels = ['positive', 'negative']

    # Create pie chart
    pie , ax = plt . subplots ( figsize =[11 ,7])
    plt.pie(x=data , autopct = lambda pct: func (pct , data ) , explode =[0.0025]*2 ,
        pctdistance =0.5 , colors =[sns. color_palette () [0] ,'tab:red'] , textprops ={'fontsize':16})
    # plt . title ( ’ Frequencies of sentiment labels ’, fontsize =14 , fontweight = ’ bold ’)
    labels = [r'Positive', r'Negative']
    plt.legend (labels , loc ="best", prop ={'size': 14})
    pie.savefig("PieChart.png")
    plt.show ()