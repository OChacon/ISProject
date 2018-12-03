import praw
import sys
import json
from reference import reddit

def download(subredditA, subredditB, outfile, count):
    #p = reddit.subreddit('tifu').random()
    pd = {}
    c = 0
    while len(pd) < count:
        #for post in [reddit.subreddit(subredditA).random() for x in range(count)]:
        post = reddit.subreddit(subredditA).random()
        p = {}
        p['title'] = post.title
        p['desc'] = post.selftext
        p['topComment'] = ""
        if len(post.comments) > 0:
            p['topComment'] = post.comments[0].body
        p['subReddit'] = post.subreddit.display_name
        pd[post.id] = p
    while len(pd)<2*count:
        #for post in [reddit.subreddit(subredditB).random() for x in range(count)]:
        post = reddit.subreddit(subredditB).random()
        p = {}
        p['title'] = post.title
        p['desc'] = post.selftext
        p['topComment'] = ""
        if len(post.comments) > 0:
            p['topComment'] = post.comments[0].body
        p['subReddit'] = post.subreddit.display_name
        pd[post.id] = p
    of = open(outfile, 'w+')
    json.dump(pd,of,indent=4,separators=(',', ': '))

if __name__ == '__main__':
    download('legaladvice','personalfinance','la_pf_100.json',100)