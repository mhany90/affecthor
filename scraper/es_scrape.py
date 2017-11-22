import tweepy
import sys

auth = tweepy.AppAuthHandler('FMwQZ2L8MM5Gzi0NVwWhsQFLD',
						     'qciRPPQ9mEVm4RHIFlxos5lHNvm3Nq1yINcoUBHoOKxpYw5osH')

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
query = 'a OR e OR i OR o OR u OR y OR el OR la OR los OR las OR que OR q OR no OR es OR un OR por OR una OR lo OR de'
limit = 10000000

count = 0
with open('es_dump.txt', 'a') as f:
	for tweet in tweepy.Cursor(api.search,q=query,lang='es').items(limit):
		print(tweet.text)
		f.write("#{} {}\n".format(tweet.id, tweet.text))
		count += 1
		f.flush()
		if count % 10000 == 0:
			sys.stdout.write("Scraped {} tweets\n".format(count))

sys.stdout.write("FINAL: Scraped {} tweets\n".format(count))

