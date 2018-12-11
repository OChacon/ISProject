# ISProject

The whole project can be run using the projectDriver.py file and it has a help function to aid in the use
of the driver. To access the help menu just run the command
    "python projectDriver.py -h"
If all you would like to do is test the accuracy of the models then you can run the command
    "python projectDriver.py -algs SVC LSTM RFC -load -eval"
This will load the preexisting models and train it on the evaluation vectors created when you download
the data with the following command.
    "python projectDriver.py -d"
Some pretrained model have been provided for you along with the method to download the test data.
If you would like to supply your own test data it must be in the form of
{
    "some identifier":
    {
        "desc": "subreddit post description"
        "subReddit": "post subreddit",
        "topComment": "The single top comment from the post",
        "title": "Post Title"
    }
}
The download script may provide some insight as to how to go about doing this unfortunately this is the only
format the code will accept.

If you would like to provide your own training data just place a file named "la_pf_100.json" into the local directory
and run
    "python SeperateData.py"
    "python vectors.py"
then just run
    "python projectDriver.py -algs ALGS [ALGS ...] -train"
and the training of the ALGS should then begin and sae the new models localy

In order to use the Download function you must create a reference.py file that looks like this:

import praw
reddit = praw.Reddit(client_id='Something', client_secret="something else",
                     password='password', user_agent='username',
                     username='username')

Note: This poject makes and uses "ALOT" of local files so I strongly recommend running from inside the directory
or in a dedicated directory

Dependencies:
    nltk
    tensorflow
    scipy
    scikit
    numpy
    pickle
    praw

Thanks,
Mike Hurlbutt and Oscar Chacon