import sqlite3
import re

cleanedTweets = []

conn = sqlite3.connect('twitter-data-collector/database.sqlite')
cur = conn.cursor()

hashtagRe = re.compile('#[a-zA-Z]+')
mentionRe = re.compile('@[a-zA-Z\d]+')
twitterLinkRe = re.compile('https://t.co/[a-zA-Z\d]+')

tweets = cur.execute('SELECT * FROM tweets')
for tweet in tweets:
    cleanedTweet = (hashtagRe.sub('', mentionRe.sub('', twitterLinkRe.sub('', tweet[2])))).strip()
    cleanedTweets.append((cleanedTweet, tweet[0]))

cur.close()

cur = conn.cursor()
cur.executemany('UPDATE tweets SET body = ? WHERE id = ?', cleanedTweets)
cur.close()
conn.commit()
