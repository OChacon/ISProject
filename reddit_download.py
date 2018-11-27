import praw
from reference import reddit
#p = reddit.subreddit('tifu').random()

for post in [reddit.subreddit('tifu').random() for x in range(100)]:
    p = {}
    p['title'] = post.title
    p['desc'] = post.selftext
    p['topComment'] = post.comments[0].body
    post.count(0)