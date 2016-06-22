# Then/Than Correction Project

### The problem

Writers of English, especially when writing quickly and in informal settings, often confuse 'then' and 'than'. For example, someone might mistakenly write "I'm a better artist *then* you" or "Walk the dog, *than* feed him". How can we correct this?

It turns out that 'then' and 'than' often occur in different sentence contexts. For example, 'than' is often preceded by comparatives like 'better', while 'then' is often preceded by a comma. Can we capitalize on this and find likely misuses of 'then' and 'than'? Maybe. The current project is an attempt at this.

### Package requirements

I wrote this while using Python 3.5.1. In addition to some standard libraries, you'll need NLTK, Pandas, Scikit-Learn, and PRAW.

### The present scripts

This repo contains the following:

* **thenThanClassifierForPRAW.py** trains a Random Forest Classifier (from Scikit-Learn) to predict usage of 'then' or 'than' based on surrounding part of speech tags. This gets about 94% accuracy on a held-out test set of 'then' and 'than' usages. If you run this script by itself, you'll execute some tests/exploration of the classifier.
* **detectThenThanComment.py** uses that classifier to predict 'then' or 'than' usage for new sentences.
* **ThenThanCommentCorrecterSHORTENED.py** uses PRAW to grab comments from [reddit.com/r/all](http:/reddit.com/r/all), and then uses the above detector to determine whether 'then' or 'than' usage was correct.

So if you just clone this repo and run:

`python ThenThanCommentCorrecterSHORTENED.py`

Then it should work. If you haven't used PRAW before, you may have to do some setup there beforehand. In addition, in the middle of the script, a browser window will pop up and give you an access code which you'll give back to the script (as user input) to allow you to access reddit via PRAW.

### Brief assessment / Future plans

So how well does this work? Decently, but not super great. It turns out (from manual inspection) that most usages of 'then' and 'than' on /r/all are correct, and thankfully, these scripts don't turn up too many corrections. For the few corrections that are identified, some are true, and some are false (though I haven't done any systematic analysis of this yet). So hmmm.

We can probably do better. I could train the classifier on a bigger corpus, maybe on one drawn from social media like reddit (cf the Brown corpus, which I'm using now). I could train the classifier on a bigger context/window around 'then'/'than' (right now, we just look at two words to the left and two words to the right). I could make a better PoS (noun, verb, adjective, etc.) tagger (see the comments in the scripts themselves for discussion of how to approach PoS tagging).

We might also try only counting corrections that the classifier is VERY sure of. Right now, we only 'trust' corrections when the classifier's predicted probability clears some predetermined confidence level (so far, 70%). This means that if a user said 'then', but the classifier predicted 'than' with 75% probability, we count that as a correction. If we were more conservative (i.e., set the confidence level to 80%), we'd get fewer false corrections (false positives), but we'd suffer more failures to correct (false negatives). Since reddit users are a prickly bunch and don't take well to being corrected in general, we probably only want to correct them when we're *really* sure of it, so we'd probably do well to err on the side of tolerating failures to correct to ensure more of our corrections are true.

Maybe at some point I'll also put this into a proper Reddit bot that will respond to incorrect usages of 'then'/'than'.

### Acknowledgment(s)

Many thanks to [Brad Ziolko](https://github.com/bradziolko/) for inspiring this project. He implemented a then/than correcting Reddit bot in a somewhat different way, not utilizing NLP and machine learning. Check out [this reddit thread](https://www.reddit.com/r/cscareerquestions/comments/4o7r89/would_a_reddit_bot_be_an_appropriate_personal/) for some explanation of how his bot works (pretty well, it turns out!).
