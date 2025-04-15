## Task

NLP Task: Tweet Enhancement for Higher Engagement

Task Overview

Develop an NLP-based enhancement model that analyzes tweet text and suggests improvements to increase engagement (likes, retweets, and reach). The system should provide actionable recommendations to optimize tweets for better performance.

Dataset
- Name: Tweets and User Engagement Dataset
- File: [Twitterdatainsheets.csv](https://www.kaggle.com/datasets/thedevastator/tweets-and-user-engagement)

Relevant Features:
- text → The actual tweet content (to be optimized)
- Likes, RetweetCount, Reach → Engagement metrics (used to evaluate tweet effectiveness)
- Sentiment → Determines the emotional tone of the tweet
- Klout → Measures the influence of the user posting the tweet
- Weekday, Hour → Useful for posting time recommendations

Task Breakdown

1. Preprocessing
- Clean tweets: Remove links, special characters, and unnecessary whitespace.
- Normalize text: Convert to lowercase and remove stopwords.
- Extract keywords and hashtags: Identify key terms influencing engagement.
2. Engagement Prediction
- Train an NLP model using tweet text and engagement metrics (Likes, RetweetCount, Reach).
- Implement a regression model to predict the engagement score of a given tweet.
- Identify text patterns that correlate with higher engagement.
3. Tweet Enhancement & Recommendations
- Suggest content improvements based on high-performing tweet patterns:
- Optimize phrasing: Recommend changes in wording or tone.
- Hashtag recommendations: Suggest effective hashtags.
- Call-to-action enhancements: Improve engagement triggers (e.g., “Retweet if you agree!”).
- Suggest best posting times based on dataset trends.

### Example Input & Output

```
Input Tweet:
{
  "text": "New job opening at our company! Apply here: https://link.com #hiring #jobs",
  "Weekday": "Monday",
  "Hour": 15
}
```
​
- Predicted Engagement Score:
  - 📊 3.2 / 10 (Low Engagement)

- Suggested Enhancements:
  - ✅ Optimized Tweet:"Exciting opportunity! 🚀 We're hiring for a new role – apply today! 🌟 → [link] #JobSearch #CareerGrowth"

- ✅ Best Posting Time:
  - Tuesday at 12 PM (historically higher engagement)

- ✅ Hashtag Improvements:
  - Use #JobSearch #CareerGrowth instead of generic #hiring #jobs (low-performing hashtags)

Expected Deliverables
- Python Codebase with:
- NLP preprocessing
- Engagement prediction model
- Tweet enhancement algorithm
- README (setup instructions & methodology)
- Short Report (approach, findings, and sample results)


## Solution

#### EDA

EDA.ipynb

- A lot of tweets with empty texts. Should remove them. 100k tweets with not empty texts
- let's remove non English texts as outliers. They are about 8% as sum of data, But every other language is less then 1.5%. Can drop it without losing signal.
- There are some outliers in data with axis Reach, Likes, RetweetCount. Should remove outliers.
- As a base-line let's see 95% percentile for each axis.
- All commercial posts are not liked in most. Can neglect this value as mark of engagement.
- Also RetweetCount is very small for most of tweets. Can neglect it too.
- The real valuable mark of engagements id Reach.
- Will use it as target for modeling and will use only it for removing outliers.
- Set threshold as tr_reach = 15000 (remove all tweets with Reach > 1500 ~ 6% of tweets)
- See that distribution of Reach is log normal. it's good to predict log_normal value of target.
- let's cut left tail or distribution. Cut log_Reach < 2.
- Log_Reach is in range 2 - 9.5
- Tweets with commercial tematics in IT services (seen 3 times for 100 examples).
- Links are full with http, https.
- Create clean texts - extract links and hashtags
- Reach analysis
  - all work-days have similar quantity of tweets. is about 15k.
  - weekend is about 6k - 3 times lower.
  - weekend has lower Reach values then workdays. It's statistically significant.
  - Tuesday, Wednesday gives less reach then others weekdays
  - Monday, Thursday, Friday - are equal by reach and best for posting
- Let's take for further analysis Monday, Thursday, Thursday, Friday tweets only for excluding weekdays season-effects
- best hours for posting
  - 9, 10, 11, 14, 16
- month days are not all in dataset. Only 1-23
  - best days for posting 9-16 (2nd week of months)
  - last week of month is off in this data and not in analysis
- extract only tweets in prime-time (popular weekdays and popular hours) - 12k tweets for correlation analysis
  - sentiment, count of links, tags, text length are not correlate with Reach
  - the only parameter correlate with Reach is Klout
  - klout is normal distributed
  - extract only tweets with similar Klout value - close to mean Klout, for more deep correlation analysis
  	- We take tweets with similar Klout and see again that there is no correlation with Reach and sentiment, count of links, tags, text length


- Features for tweets reach are
  - Klout
  - weekday
  - day of month
  - hour


- May be we can extract some text features
  - text tematic
  - call to action
  - may be some text embedding


- let's find best and worst tweets to see if text of theme differ
- tokenize clean texts - lower, extract only words, remove stop words
- get most popular tweets (above 95% percentile random sample 2000)
- get worst tweets (lower 5% percentile random sample 2000)
- get top 100 popular words from popular and worst
- most popular tweets and most worst tweets has the same words (63% of 100 most popular words in both groups are the same).


- Hypothesis: there is no difference in tematic for popular and unpopular tweets in this dataset
- Let's check this with bert model for semantic similarity.
- Use Sentence Transformers STS model all-MiniLM-L6-v2
- There is no difference in tematic for popular and unpopular tweets in this dataset
  - for random samples for 20 and 100 tweets from most popular and most unpopular tweets they have similar semantic vector (cos-sim 0.96 for every average-100 vectors, cos-sim 0.88 for every average-20 vectors)
  -
#### Conclusion of EDA

- for this dataset text key words, or hashtags doesn't influence on engagement.
- for such commercial short IT messages the only valuable things are
  - account Klout
  - weekday
  - day of month
  - hour

- Then we can use Tree model for such data to predict engagement.
- we will predict lor_reach data as target.
- we can use lor_reach data for scale from 1 to 10 as engagement score for tweet

- There is no need to learn NLP model to predict engagement. Engagement does't depends on content and hashtags in this domain (It staff).
- The key insight
  - you can write what you want about IT services, Job offers, webinars and others that are in this dataset - content not valuable for engagement
  - you will have very small amount of likes and will not have retweets
  - if you want to have more reach
	- tweet in prime-time:  Monday, Thursday, Friday and hours: 9, 10, 11, 14, 16
	- ask someone with most Klout retweet you to deliver your message

### Model Learning

learn_model.ipynb

- let's create scale function and training data for model.
- Exclude Day data from dataset because it's not full monts. We will get not full data model - it wouldn't work with data of other days: 24-31
- log_reach to score function:
```
def scale_function(x):
	sc =  (x-2)*10/7.5
	return min(max(0, sc), 10)
```

- dataset features
  - WeekDay
  - Hour
  - Klout
- target
  - log_Reach

- use OneHotEncoder for WeekDay and Hour

- First try - RandomForestRegressor(n_estimators=20, max_depth=5)
  - neg_mean_absolute_error - 0.75 on cross validation
- target 0 is normal distributed with mean = 6, std = 1.53
- 0.76 MAE - is vell enough for this task. If we scale target to 0-10 we have error not more then 1 scale-grade.
- for task of predicting scale-score it's ok - you differentiate bad-mid-good objects.

- Make grid search cross-validate for finding better RandomForestRegressor
  - best params: {'max_depth': 10, 'n_estimators': 300}
  - best score: 0.7592471231979313
  - see that best model score is close to first-try model
 
- Let's check more simple models - LinearRegression and Lasso
- Linear Regression MAE cross-val score is 0.86
  - not more worse then RandomForestRegressor
- Lasso Grid-search gives similar cross-val MAE
  - best params: {'alpha': 0.1}
  - best score: 0.859382925125782
- Lasso on diferent alpha gives us all coefficient 0, except Klout coefficient and CV-MAE is about 0.86.
- Because of data it's reasonable use only Klout to predict engagement score:
```
def predict_popularity_klout(klout):
	"""
	x is Klout parameter of tweet
	"""
	pr = 0.08*klout + 2.85
	return round(min(max(0,(pr-2)*10/7.5), 10), 0)
```
- i.e. engagement depends on account popularity and not on text for this dataset, and account popularity is more valuable then time and day of posting


### Improve Tweet

improve_tweet.ipynb

-  Because engagement not depend of text and hashtags in this data then we can neglect this data for improving tweets and use only common sense
-  let's use LLM as tool for improving texts and set hashtags
-  will use One-Shot prompting, as one-shot use example from task
-  for LLM connector use openrouter for being LLM independent
-  if it will work not well - it's possible add more few-shots for text improvement and hashtags
-  if LLM will fail a lot with next day and time - it's easy to calculate it separately with python function.

### General thoughts
- It's common to train NLP model for predicting anything for text based on pretrained BERT models.
- But for this task it is redundant
- It's common way to use BERT models to extract key words from texts for analyze topics
- Also it's possible to normalize words and do LDA topic modeling, but for this data it's not necessary.
- Suggested above simple solution is enough for task solving
- It's simple baseline for easy and fast prototype solution - to show it to business and discuss and test it.
 
