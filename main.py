from cfg import *
from utilities import visual, pie_chart
from utilities import preprocess_text
def config_graphviz()->None:
    dot_path = '/opt/homebrew/bin/dot'
    print(f'os path: {os.pathsep + os.path.dirname(dot_path)}')
    os.environ["PATH"] += os.pathsep + os.path.dirname(dot_path)
    
def run():
    """ ----------------------------------Data Loading and Drop Duplicate"""
    df = pd.read_csv("IMDB-Dataset.csv")
    df = df.drop_duplicates()

    """ ---------------------------------- Text Preprocessing """
    df['review'] = df['review'].apply(preprocess_text)
    
    """ ----------------------------------Visualization and Statistics """
    pie_chart(df)
    #visual(df)
    
    
    """ ----------------------------------Data Encoded """
    x_data = df['review']
    y = df.drop('review', axis=1)
    
    """ ----------------------------------Train Testing Spliting """

    label_encode = LabelEncoder ()

    y_data = label_encode . fit_transform (df['sentiment'])
    x_train , x_test , y_train , y_test = train_test_split (
        x_data , y_data , test_size =0.2 , random_state =42
    )
    
    """ ----------------------------------Vectorization by apply TF-IDF """
    tfidf_vectorizer = TfidfVectorizer ( max_features =10000)
    tfidf_vectorizer .fit ( x_train , y_train )

    x_train_encoded = tfidf_vectorizer . transform ( x_train )
    x_test_encoded = tfidf_vectorizer . transform ( x_test )
    
    """ ---------------------------------- Decision Tree """
    dt_classifier = DecisionTreeClassifier (
        criterion ='entropy',
        random_state =42
        )
    dt_classifier.fit( x_train_encoded , y_train)
    y_pred = dt_classifier.predict(x_test_encoded)
    dt_accuracy = accuracy_score (y_pred , y_test )
    print(f'Decision Tree Accuracy: {dt_accuracy}')
    
    """ ---------------------------------- Random Forest """
    rf_classifier = RandomForestClassifier (random_state =42)
    rf_classifier .fit( x_train_encoded , y_train )
    y_pred = rf_classifier . predict ( x_test_encoded )
    rf_accuracy = accuracy_score (y_pred , y_test )
    print(f'Random Forest Accuracy: {rf_accuracy}')
    
    """ ---------------------------------- AdaBoosting """
    ada_classifier = AdaBoostClassifier (n_estimators=30,random_state =42)
    ada_classifier . fit( x_train_encoded , y_train )
    y_pred = ada_classifier . predict ( x_test_encoded )
    ada_accuracy = accuracy_score (y_pred , y_test )
    print(f'AdaBoosting Accuracy: {ada_accuracy}')
    
    """ ---------------------------------- Gradient Boosting """
    grd_classifier = GradientBoostingClassifier (n_estimators=30,random_state =42)
    grd_classifier . fit( x_train_encoded , y_train )
    y_pred = grd_classifier . predict ( x_test_encoded )
    grd_accuracy = accuracy_score (y_pred , y_test )
    print(f'Gradient Boosting Accuracy: {grd_accuracy}')
    
    """ ---------------------------------- XGBoosting """
    xgb_classifier = xgb.XGBClassifier(n_estimators=20,learning_rate = 0.8,max_depth = 4,
                                random_state =42
    )
    xgb_classifier.fit(x_train_encoded , y_train )
    y_pred = xgb_classifier.predict(x_test_encoded )
    xgb_accuracy = accuracy_score (y_pred,y_test )
    print(f'XGBoost Accuracy: {xgb_accuracy}')
    
if __name__ == "__main__":
    run()