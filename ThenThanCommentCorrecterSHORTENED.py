"""
Getting comments with 'then' or 'than' in them, and then deciding whether or not
the usage of 'then' or 'than' is incorrect.

I'm following the steps here:
http://praw.readthedocs.io/en/stable/pages/oauth.html?highlight=authenticate#step-2-setting-up-praw
"""

import praw, pprint, webbrowser, time, pickle, nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

if ('make_thenThan_classifier' not in dir()) or ('detectThenThanComment' not in dir()):
    try:
        from thenThanClassifierForPRAW import make_thenThan_classifier
        from detectThenThanComment import detectThenThanComment
    except ImportError:
        print("Could not import functions. Are you running this script from Rodeo IDE? "
              "Either run this from terminal, or run the scripts from which you are "
              "importing, and then run this script again")

# In other testing, I determined that a unigram tagger with NN default backoff trained
# on brown corpus got about 90% of tags right. I suspect this will work pretty well here.
#
# I decided to try the unigram tagger with regex backoff for reasons.
patterns = [
            (r'.*ing$', 'VBG'),               # gerunds
            (r'.*ed$', 'VBD'),                # simple past
            (r'.*es$', 'VBZ'),                # 3rd singular present
            (r'.*ould$', 'MD'),               # modals
            (r'.*\'s$', 'NN$'),               # possessive nouns
            (r'.*s$', 'NNS'),                 # plural nouns
            (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
            (r'.*', 'NN')                     # nouns (default)
            ]

t0         = nltk.RegexpTagger(patterns)
#t0         = nltk.DefaultTagger('NN')
pos_tagger = nltk.UnigramTagger(brown.tagged_sents(), backoff=t0)
print("Tagger for Reddit comments has been trained")

# Now do some stuff to get access to reddit
user_agent = ("then/than comments by /u/squirreltalk")
r = praw.Reddit(user_agent)
r.set_oauth_app_info(client_id='_T1U9A4zHwgzvA',
                     client_secret='sIPlWZpZYPXknlB9_Yor02sDtto',
                     redirect_uri='http://127.0.0.1:65010/'
                                  'authorize_callback')
                                  
# need to pass the below method all the 'scopes' that I need: the kinds of 
# things I want to do when interacting with reddit...
#url = r.get_authorize_url('uniqueKey', 'history identity read submit', True)
url = r.get_authorize_url('uniqueKey', 'identity read', True)
webbrowser.open(url)

try:
    uniqueKey = input("Please enter the uniqueKey & Code from the page that popped up: ")
except:
    print("This frontend does not support input requests. Manually enter the key on the \
    	   next line of the script, select the appropriate portion of the script below, \
    	   and run again.")
    uniqueKey = 'DBfCQINOj6zn22wf1aHih3ywje8'

access_information = r.get_access_information(uniqueKey)
authenticated_user = r.get_me()

# Now should have OAuth and be able to access
window = 2
n_estimators = 20
clf, dummyData, allData = make_thenThan_classifier(window=window, 
                                                   n_estimators=n_estimators
                                                   )

print("Finished making then/than classifier...")
print("Now getting reddit comments and looking for then/than mistakes...")

r.refresh_access_information(access_information['refresh_token'])
all_comments = r.get_comments('all', limit = None)

for comment in all_comments:
    for word in ['then', 'than']:
        detectThenThanComment(word=word, window=window, pos_tagger=pos_tagger, 
                              commentText=comment.body, allData=allData, 
                              dummyData=dummyData, tokenizeFunction=word_tokenize,
                              clf=clf, confidenceLevel=0.3)