from google_play_scraper import Sort, reviews as play_reviews
import datetime
import time

app_id = "com.zhiliaoapp.musically"
print("Fetching for", app_id)

t0 = time.time()
try:
    result, _ = play_reviews(
        app_id,
        lang='tr',
        country='tr',
        sort=Sort.NEWEST,
        count=15000
    )
    print("Found total limit:", len(result))
    if result:
        print("Oldest date limit:", result[-1]['at'])
except Exception as e:
    print("Error:", e)
print("Time taken:", time.time() - t0)
