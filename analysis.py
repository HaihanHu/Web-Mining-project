import pandas as pd
import numpy as np
import nltk
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import scipy.sparse as sp
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
stemmer = SnowballStemmer("english")
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.cluster import KMeans

def showCON(name, cla, x, y):
    
    predict = cla.predict(x)
    probs = cla.predict_proba(x)
    probs = probs[:,1]
    draw_confusion_matrices([(name, confusion_matrix(y, predict))])

    fpr_rf, tpr_rf, thresholds = roc_curve(y, probs)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label=name)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def cal_evaluation(classifier, cm):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 0.0)
    precision = tp / (tp + fp + 0.0)
    recall = tp / (tp + fn + 0.0)
    print (classifier)
    print ("Accuracy is: %0.3f" % accuracy)
    print ("precision is: %0.3f" % precision)
    print ("recall is: %0.3f" % recall)

def draw_confusion_matrices(confusion_matricies):
    class_names = ['Not','Churn']
    for cm in confusion_matricies:
        classifier, cm = cm[0], cm[1]
        cal_evaluation(classifier, cm)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm, interpolation='nearest',cmap=plt.get_cmap('Reds'))
        plt.title('Confusion matrix for %s' % classifier)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

def tokenization_and_stemming(contents):
    
    stopwords = nltk.corpus.stopwords.words('english')
    stems = []
    for sent in contents:
        nsent = []
        for word in nltk.word_tokenize(sent):
            if word not in stopwords:
                nsent.append(stemmer.stem(word))
        stems.append(" ".join(nsent))
    return stems
        

if __name__ == '__main__':
    
    data = pd.read_csv('dataset.csv').reset_index(drop=True)
#   data.dropna(how='any', inplace=True, subset=['loc_country'])
    
#    print(data.isnull().sum())
    
    data.videoCount = data.videoCount.fillna(0)
    data.imageCount = data.imageCount.fillna(0)
    
    features = ['goal', 'backers_count', 
                'loc_country', 'cate_name', 
                'videoCount', 'imageCount']
    
    X = data[features]
    y = data.success
    
    le = preprocessing.LabelEncoder()

    le.fit(X.loc_country)
    X.loc[:,'loc_country'] = le.transform(X['loc_country'])

    le.fit(X.cate_name)
    X.loc[:,'cate_name'] = le.transform(X['cate_name'])
    
    cates = X['cate_slug']
    X.loc[:,'category'] = [item.split('/')[0] for item in cates]
    X.drop('cate_slug', axis=1, inplace=True)
    
    le.fit(X.cate_name)
    X.loc[:,'category'] = le.transform(X['cate_name'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    train_contents = X_train.content
    test_contents = X_test.content
    
    X_train.drop(['content', 'blurb', 'pledged'], axis=1, inplace=True)
    X_test.drop(['content', 'blurb', 'pledged'], axis=1, inplace=True)
    
    train_contents = tokenization_and_stemming(train_contents)
    test_contents = tokenization_and_stemming(test_contents)

    vectorizer = TfidfVectorizer(max_df=0.8,
                             min_df=0.2, stop_words='english',
                             use_idf=True, ngram_range=(1,1))
    vectorizer.fit(train_contents)
    counts_train = vectorizer.transform(train_contents)
    counts_test = vectorizer.transform(test_contents)
    
    counter = CountVectorizer()
    counter.fit(train_contents)
    counts_train = counter.transform(train_contents)
    counts_test = counter.transform(test_contents)
    
    X_train = sp.csr_matrix(X_train)
    X_test = sp.csr_matrix(X_test)
    
    X_train = sp.hstack((counts_train, X_train), format='csr')
    X_test = sp.hstack((counts_test, X_test), format='csr')
    
    clf1 = MultinomialNB(alpha=0.1)
    clf1.fit(X_train,y_train)
    pred = clf1.predict(X_test)
    print (accuracy_score(pred,y_test))
    showCON('DTC', clf1, X_test, y_test)
    
    clf2 = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train,y_train)
    pred = clf2.predict(X_test)
    print (accuracy_score(pred,y_test))
    showCON('DTC', clf2, X_test, y_test)
    
    clf3 = LinearDiscriminantAnalysis()
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    clf3.fit(X_train,y_train)
    clf3_score = clf3.score(X_test, y_test)
    print (clf3_score)
    showCON('DTC', clf3, X_test, y_test)
      
    clf4 = KNeighborsClassifier(n_neighbors=4)
    clf4.fit(X_train,y_train)
    clf4_score = clf4.score(X_test, y_test)
    print (clf4_score)
    showCON('KNN(K=4)', clf4, X_test, y_test)


    clf5 = tree.DecisionTreeClassifier()
    clf5 = clf5.fit(X_train,y_train)
    pred = clf5.predict(X_test)
    print (accuracy_score(pred,y_test))
    showCON('DTC', clf5, X_test, y_test)
   

    clf6 = KMeans(n_clusters=4)
    clf6 = KMeans.fit(X_train,y_train)
    pred = clf6.predict(X_test)
    print (accuracy_score(pred,y_test))
    showCON('KMeans', clf6, X_test, y_test)



























