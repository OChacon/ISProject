import praw
import sys
import json
from reference import reddit

def download(subreddit, outfile, count):
    #p = reddit.subreddit('tifu').random()
    pd = {}
    for post in [reddit.subreddit(subreddit).random() for x in range(count)]:
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
    download('tifu','tifu_10.json',10)