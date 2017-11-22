import tweepy
import sys

auth = tweepy.AppAuthHandler('PTkRZpwIXTIXs2rFpC8bU7dN1',
						   'Rz80z98fAlVwU6a4ckCzZ7U7pXXWYVGwRyxw7VkH48b7Qc4gtE')

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
query = 'a OR e OR i OR o OR u'
limit = 10000000

count = 0
with open('es_dump.txt', 'w') as f:
	for tweet in tweepy.Cursor(api.search,q=query,lang='es').items(limit):
		f.write("#{} {}\n".format(tweet.id, tweet.text))
		count += 1
		if count % 10000 == 0:
			sys.stdout.write("Scraped {} tweets\n".format(count))

sys.stdout.write("FINAL: Scraped {} tweets\n".format(count))

