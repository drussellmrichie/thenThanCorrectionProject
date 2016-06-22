import pandas as pd

def detectThenThanComment(word, 
						  window, 
						  pos_tagger, 
						  commentText, 
						  tokenizeFunction,
						  allData, 
						  dummyData,
						  clf,
						  confidenceLevel = 0.3):
    if (word not in tokenizeFunction(commentText)):
        return
    
    #time.sleep(3)
    tokens = tokenizeFunction(commentText)
    wordInd = tokens.index(word)
    sliceObject = slice(max(0,wordInd-window), min(wordInd+window+1,len(tokens)))
    wordContext = [x for x in tokens[sliceObject]]
    wordTagPairs   = pos_tagger.tag(wordContext)
    print("\n",wordTagPairs)
    tags = [x[1] for x in wordTagPairs]

    # now have to convert tags to dummy encoding before finally predicting probability
    # of then vs than...
    # This dataframe wrangling is horrible. Horrible, dreadful, no good. Should find
    # a better way to do this eventually.
    dfTags = pd.DataFrame(tags).transpose()
    dfTags.columns = ["Slot{}".format(x-window) for x in dfTags.columns]
    dfTags['th{e|a}n'] = 0 if word == 'then' else 1
    dfTags.drop('Slot0', axis=1, inplace=True)

    tempData = allData.append(dfTags)
    tempDummyData = pd.get_dummies(tempData)
    try:
        if not all(tempDummyData.columns == dummyData.columns):
            print("Can't process this comment because it has a tag new to the clf")
            return
    except ValueError:
        print("Can't process this comment because it has a tag new to the clf")
        return
    
    dfTags = tempDummyData.tail(1)    
    dfTags.drop('th{e|a}n', axis=1, inplace=True)
    probs = clf.predict_proba(dfTags)
    print(probs)
    
    # remember that 0 is 'then' and 1 is 'than'...so if the first prob is high, the
    # clf predicts that 'then' should be used    
    
    if word == 'then':
        probsInd = 0
        otherWord = 'than'
    else:
        probsInd = 1
        otherWord = 'then'
    
    if probs[0][probsInd] < confidenceLevel:
        print("You said '", ' '.join(wordContext), "'.",  sep='')
        wordContext[window] = otherWord
        print("Did you mean '", ' '.join(wordContext), "'?", sep='')
    else:
        print("Commenter (probably) correctly used '{}'".format(word))