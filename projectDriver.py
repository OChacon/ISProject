import reddit_download
import os

def run():
    # Training Data
    if not os.path.exists('training.txt'):
        reddit_download.download('legaladvice','personalfinance','training.txt',50)
    # Develop Data
    if not os.path.exists('developing.txt'):
        reddit_download.download('legaladvice','personalfinance','developing.txt',25)
    # Eval Data
    if not os.path.exists('eval.txt'):
        reddit_download.download('legaladvice', 'personalfinance', 'eval.txt', 25)
