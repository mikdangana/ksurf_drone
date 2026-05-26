from searchtweets import load_credentials
from print_utils import print_data
import requests, sys, json



def tokens():
    access_tok = "1461124379211972619-qtTay8A5nDCTO58RGfTXkfANc1BIkl"
    access_secret = "99DElCh3VAu847AlMrWkQqPAVKDVynqxWLYJROsk9TXKf"
    bearer_tok = "AAAAAAAAAAAAAAAAAAAAAPjGnAEAAAAApOy5ZpFxznXBulzNefLd1rEhV68%3DUKB5XNhwfdL6pT5J2X9ke6StVr4RTLrUfxB65ZUTSfRgVTrCbt"
    api_key = "9qUvKwqpycGmGZ4krTicH2n60"
    api_secret = "4YUP8Y0EmmzgmWjLZYC2qweu0yHfcUMLKqUkVeHvtcYD95vNIw"
    return access_tok, access_secret, bearer_tok, api_key, api_secret


def fetch_recent():
    #url = "https://api.twitter.com/2/tweets/counts/all"
    #url = "https://api.twitter.com/2/tweets/search/recent"
    url = "https://api.twitter.com/2/tweets/counts/recent"
    #qparams = {'query': '(from:twitterdev -is:retweet) OR #twitterdev',
    #           'tweet.fields': 'author_id'}
    toks, csv_file = tokens(), 'tweets.csv'
    headers = { 'User-Agent': "v2RecentSearchPython",
                'Authorization': f"Bearer {toks[2]}" }
    #data = { 'client_id': toks[3], 'grant_type': 'client_credentials' }
    qparams = { 'query': '#Ukraine', 'granularity': 'minute' }
               #'start_time': '2019-03-05T00:00:00Z' }
               #'end_time': '2019-03-06T00:00:00Z'} 
    print(f"url = {url}, hdrs = {headers}, params = {qparams}")
    resp = requests.get(url, params=qparams, headers=headers)
    resp = json.loads(resp.text)
    None if 'data' in resp else print(f"message = {resp}") 
    data = [f"{d['start']},{d['end']},{d['tweet_count']}" \
            for d in resp['data']] if 'data' in resp else []
    print_data(data, csv_file)
    print(f"Output: {csv_file}, resp.keys = {resp.keys()}, " \
          f"meta = {resp['meta'] if 'meta' in resp else None}")
    return resp


def refresh_token():
    url = "https://api.twitter.com/2/oauth2/token"
    toks = tokens()
    headers = { 'Content-Type': 'applicaiton/x-www-form-urlencoded',
                'User-Agent': f"{toks[3]}:{toks[4]}",
                'Authorization': f"Bearer {toks[2]}" }
    data = { 'client_id': toks[3], 'grant_type': 'client_credentials',
             'grant_type': 'refresh_token' }
    body = { 'text': 'Test message' }
    print(f"url = {url}, hdrs = {headers}, data = {data}")
    resp = requests.post(url, params=data, headers=headers, json=body)
    print(f"resp = {resp}")
    return resp.json()['refresh_token']


def post_tweet():
    url = "https://api.twitter.com/2/tweets"
    toks = tokens()
    headers = { 'User-Agent': f"{toks[3]}:{toks[4]}",
                'Authorization': f"Bearer {toks[2]}" }
    data = { 'client_id': toks[3], 'grant_type': 'client_credentials' }
    body = { 'text': 'Test message' }
    print(f"url = {url}, hdrs = {headers}, data = {data}")
    resp = requests.post(url, params=data, headers=headers, json=body)
    print(f"resp = {resp.json()}")
    return resp



if __name__ == "__main__":
    if "-p" in sys.argv or "--post" in sys.argv:
        #post_tweet()
        print(f"refresh_token = {refresh_token()}")
    else:
        fetch_recent()
