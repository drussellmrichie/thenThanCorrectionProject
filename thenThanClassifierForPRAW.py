"""
Get POS contexts for 'then' and 'than'. Then see if a classifier can predict whether 
'then' or 'than' should be used given POS contexts. Works reasonably well at the moment: 
94% accuracy on test set.

This the obtained classifier will then be used in conjunction with PRAW to get reddit
comments which used 'then' or 'than', and see whether they should be corrected.
"""

from nltk.corpus import brown
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def make_thenThan_classifier(window = 2, n_estimators = 20):
    """
    Get a classifier that predicts, given the POS tags for the window words on either side
    of an instance of then/than, whether then or than should have been used.
    
    Right now, this uses a random forest classifier, and n_estimators specifies how many
    trees should be in the classifie. More trees = better performance (w/ diminishing
    returns), but potentially slower.
    
    Also returns the data used to train the classifier, which is useful for testing it,
    and also used when classifying new reddit comments.
    
    Could improve this by building the pandas table while looping through tagged_sents,
    instead of building lists in the loop, and then later converting to table?    
    """
    taggedSentTotal = len(brown.tagged_sents())

    thenSentTags = []
    thanSentTags = []

    for sentIndex, tagged_sent in enumerate(brown.tagged_sents()):
        sent = [x[0] for x in tagged_sent]
        if ('then' in sent):
            thenInd = sent.index('then')
            tags = [x[1] for x in tagged_sent[max(0,thenInd-window):min(thenInd+window+1,len(tagged_sent))]]
            #tags.extend(0)
            thenSentTags.append(tags)
        if ('than' in sent):
            thanInd = sent.index('than')
            tags = [x[1] for x in tagged_sent[max(0,thanInd-window):min(thanInd+window+1,len(tagged_sent))]]
            #tags.extend(1)
            thanSentTags.append(tags)

    # Convert the lists of then and than tag contexts to pandas dataframes, which we'll
    # then feed to our classifier for training

    thenData = pd.DataFrame(thenSentTags)
    thenData.columns = ["Slot{}".format(x-window) for x in thenData.columns]
    thenData['th{e|a}n'] = 0

    thanData = pd.DataFrame(thanSentTags)
    thanData.columns = ["Slot{}".format(x-window) for x in thanData.columns]
    thanData['th{e|a}n'] = 1

    allData = thenData.append(thanData)
    allData.drop('Slot0', axis=1, inplace=True)

    # convert categorical labels to one-hot encoding/dummy variables and specify the input
    # and output of the model

    dummyData = pd.get_dummies(allData)

    X = dummyData.loc[:,"Slot-{}_'".format(window):]
    y = dummyData['th{e|a}n']
    
    # now select and fit a model
    clf = RandomForestClassifier(n_estimators = n_estimators)
    clf.fit(X,y)
    
    return (clf, dummyData, allData)
    
if __name__ == "__main__":

    import time
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import confusion_matrix

    testingWindow = 2
    startTime = time.time()
    print("Now making and training a classifier")
    clf, dummyData, allData = make_thenThan_classifier(window = testingWindow)
    duration = time.time() - startTime
    print("Done making and training a classifier. That took this long:", duration)

    X = dummyData.loc[:,"Slot-{}_'".format(testingWindow):]
    y = dummyData['th{e|a}n']
    
    # do a test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # now select and fit a model on training data
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)

    # Now evalute on test data
    print("On test data, classifier accuracy was:", clf.score(X_test,y_test))
    # Should give accuracy around 94%

    # make a confusion matrix so we can see TP, FP, etc rates.
    y_pred = clf.predict(X_test)
    y_pred = pd.Series(y_pred, name="th{e|a}n'_pred", index=y_test.index)
    for ys in [y_pred, y_test]:
        ys.replace({0:'then',1:'than'}, inplace=True)
        
    print("\nA confusion matrix between predictions and actual 'thens' and 'thans':")
    print(pd.crosstab(y_test, y_pred))
    # evenly split between false negatives and false positives...and very few of each!

    """
    Let's look at feature importances to make sure those are sensible.
    Look at http://www.comp.leeds.ac.uk/ccalas/tagsets/brown.html for brown corpus tag
    interpretations
    """

    colsWithImportances = list(zip(dummyData.columns[1:],clf.feature_importances_))
    colsWithImportances = sorted(colsWithImportances, key = lambda x: x[1], reverse=True)

    importances = pd.DataFrame(colsWithImportances, columns = ['Feature','Importance'])
    print("\n",importances.head(10))        
    """
    Completely sensible results here -- a conjunction (CC) or comma in slot -1 is very
    informative, strongly predicting usage of 'then'. Contrast 'and then' or ', then'
    with 'and than' or ', than'. The latter two are never grammatical, I think!
    
    Conversely, a comparative (JJR) or in slot -1  predicts 'than'. Contrast 'longer than'
    with 'longer then'. The latter is possible but should be much less common.
    
    So the random forest classifier is solving this classification problem in reasonable
    ways.
    """