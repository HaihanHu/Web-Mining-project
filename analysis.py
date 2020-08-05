'''
Team 4
2018-11-14
1542178800
'''

#import pandas as pd
#import numpy as np
#import nltk
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import roc_curve
#import scipy.sparse as sp
#from nltk.stem.snowball import SnowballStemmer
#from sklearn.linear_model import LogisticRegression
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.neighbors import KNeighborsClassifier
#stemmer = SnowballStemmer("english")
#from sklearn import preprocessing
#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
#import matplotlib.pyplot as plt
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import accuracy_score
#import seaborn as sns

#def showCON(name, cla, x, y):
#    
#    predict = cla.predict(x)
#    draw_confusion_matrices([(name, confusion_matrix(y, predict))])
#
#    fpr_rf, tpr_rf, _ = roc_curve(y, predict)
#    plt.figure(1)
#    plt.plot([0, 1], [0, 1], 'k--')
#    plt.plot(fpr_rf, tpr_rf, label=name)
#    plt.xlabel('False positive rate')
#    plt.ylabel('True positive rate')
#    plt.title('ROC curve - RF model')
#    plt.legend(loc='best')
#    plt.show()
#
#def cal_evaluation(classifier, cm):
#    tn = cm[0][0]
#    fp = cm[0][1]
#    fn = cm[1][0]
#    tp = cm[1][1]
#    accuracy  = (tp + tn) / (tp + fp + fn + tn + 0.0)
#    precision = tp / (tp + fp + 0.0)
#    recall = tp / (tp + fn + 0.0)
#    print (classifier)
#    print ("Accuracy is: %0.3f" % accuracy)
#    print ("precision is: %0.3f" % precision)
#    print ("recall is: %0.3f" % recall)

#def draw_confusion_matrices(confusion_matricies):
#    class_names = ['Not','Churn']
#    for cm in confusion_matricies:
#        classifier, cm = cm[0], cm[1]
#        cal_evaluation(classifier, cm)
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        cax = ax.matshow(cm, interpolation='nearest',cmap=plt.get_cmap('Reds'))
#        plt.title('Confusion matrix for %s' % classifier)
#        fig.colorbar(cax)
#        ax.set_xticklabels([''] + class_names)
#        ax.set_yticklabels([''] + class_names)
#        plt.xlabel('Predicted')
#        plt.ylabel('True')
#        plt.show()
#
#def tokenization_and_stemming(contents):
#    
#    stopwords = nltk.corpus.stopwords.words('english')
#    stems = []
#    for sent in contents:
#        nsent = []
#        for word in nltk.word_tokenize(sent):
#            if word not in stopwords:
#                nsent.append(stemmer.stem(word))
#        stems.append(" ".join(nsent))
#    return stems
#        
#
#if __name__ == '__main__':
    
#    data = pd.read_csv('/Users/shirleyhu/Desktop/data2.csv').reset_index(drop=True)
#    data.dropna(how='any', inplace=True, subset=['loc_state'])
    
#    print(data.isnull().sum())
    
#    data.videoCount = data.videoCount.fillna(0)
#    data.imageCount = data.imageCount.fillna(0)
#    data.content = data.content.fillna('nan')
    

    
#    X = data[features]
#    y = data.success
    
#    X.loc[:,'len_blurb'] = [len(item) for item in X['blurb']]

#    le = preprocessing.LabelEncoder()
#    
#    le.fit(X.loc_state)
#    X.loc[:,'loc_state'] = le.transform(X['loc_state'])
#    
#    le.fit(X.loc_country)
#    X.loc[:,'loc_country'] = le.transform(X['loc_country'])
#
#    cates = X['cate_slug']
#    X.loc[:,'category'] = [item.split('/')[0] for item in cates]
#    X.drop('cate_slug', axis=1, inplace=True)
    
#    le.fit(X.category)
#    X.loc[:,'category'] = le.transform(X['category'])
    
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#    
#    train_contents = X_train.content
#    test_contents = X_test.content
#    
#    X_train.drop(['content', 'blurb', 'pledged'], axis=1, inplace=True)
#    X_test.drop(['content', 'blurb', 'pledged'], axis=1, inplace=True)
#    
##    train_contents = tokenization_and_stemming(train_contents)
##    test_contents = tokenization_and_stemming(test_contents)
#
#    vectorizer = TfidfVectorizer(max_df=0.8,
#                             min_df=0.2, stop_words='english',
#                             use_idf=True, ngram_range=(1,1))
#    vectorizer.fit(train_contents)
#    counts_train = vectorizer.transform(train_contents)
#    counts_test = vectorizer.transform(test_contents)
#    
##    counter = CountVectorizer()
##    counter.fit(train_contents)
##    counts_train = counter.transform(train_contents)
##    counts_test = counter.transform(test_contents)
#    
#    X_train = sp.csr_matrix(X_train)
#    X_test = sp.csr_matrix(X_test)
#    
#    X_train = sp.hstack((counts_train, X_train), format='csr')
#    X_test = sp.hstack((counts_test, X_test), format='csr')
#    
#    clf = MultinomialNB(alpha=0.1)
#    clf.fit(X_train,y_train)
#    pred = clf.predict(X_test)
#    print (accuracy_score(pred,y_test))
##    showCON('DTC', clf, X_test, y_test)
#    
#    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train,y_train)
#    pred = clf.predict(X_test)
#    print (accuracy_score(pred,y_test))
##    showCON('DTC', clf, X_test, y_test)
#
##    perm = PermutationImportance(clf, random_state=1).fit(X_train,y_train)
##    features = ['pledged','loc_state', 'loc_country','category','videoCount', 'imageCount','len_blurb']
##    eli5.show_weights(perm, feature_names = features)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = pd.read_csv('/Users/shirleyhu/Desktop/data2.csv')


    

f['category']=np.nan
for i in range(0,2400):
    f['category'][i]=f['cate_slug'][i].split('/')[0]
    
#cate = sns.barplot(x="category",y="success",data=f)
#cate = g.set_ylabel("Success Probability")

c = f['category']
result = pd.value_counts(c)
print (result)




















