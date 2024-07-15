import nltk
import numpy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


def Tokenization (text):
    tokens = word_tokenize (text)
    token = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stopwords.words('english')]
    return token
        
def P_Indicators (tokenize_text, base_keyword):
    base_found = 0
    ind_found = 0
    base_found = False
    indicators = ['looking for', 'looking', 'recommend', 'recommendation', 'recommendations', 'advice', 'compare', 'best', 'need', 'trying to find',
                  'options for','considering', 'searching for', 'seeking', 'opinions on', 'experiences with', 'pros and cons of', 'affordable', 'reliable', 'i want', 'alternative'
                 ]
    for token in tokenize_text:
        for ind in indicators: 
            if token == base_keyword:
                base_found += 1
            if ind == token:
                ind_found += 1
            if base_found and ind_found >= 1:
                return True
    return False
        
def Sentiment_Analysis (model, text):
    tokenize = AutoTokenizer.from_pretrained (model)
    Model = AutoModelForSequenceClassification.from_pretrained (model)
    encode = tokenize (text, return_tensors = 'pt')
    Modelout = Model (**encode)
    numpyout = Modelout[0][0].detach().numpy()
    softout = softmax(numpyout)
    return softout
    
    
def Selection (model, text, base_keyword):
    Selection_Success = False
    ind_success = P_Indicators (Tokenization (text), base_keyword)
    sentiment = Sentiment_Analysis (model, text)
    if ind_success and (sentiment[2] > sentiment[0] or sentiment[1] > sentiment[0]):
        Selection_Success = True
        return Selection_Success
    return Selection_Success
    
    
    
    
base_keyword = "hosting"
model = "cardiffnlp/twitter-roberta-base-sentiment"

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Testing On Sample Questions Data of 75 Questions of (complex and medium strucure), With Time and Accuracy Analysis..

from tqdm import tqdm
base_keyword = "hosting"
model = "cardiffnlp/twitter-roberta-base-sentiment"
sample_texts = [
    "I'm thrilled with the performance of my new web hosting service. Pages load lightning-fast!",
    "Just signed up with a hosting provider recommended by a friend. Excellent customer support and uptime so far.",
    "Impressed by the features and ease of use of my cloud hosting service. Definitely worth the investment.",
    "My current hosting service keeps crashing, causing downtime for my business. Looking for a reliable alternative.",
    "Frustrated with the slow response time from my web hosting support team. Need better customer service.",
    "Disappointed with the lack of security features in my shared hosting plan. Considering a switch.",
    "Exploring different hosting options for my new project. Any recommendations?",
    "What are the key factors to consider when choosing a hosting provider for an e-commerce site?",
    "Learning about the differences between cloud hosting and VPS for hosting my personal blog.",
    "I am looking for a good web hosting service.",
    "Can anyone recommend a video editor?",
    "I'm having trouble with my current hosting provider.",
    "What is the best hosting service for a small business?",
    "I don't need any hosting services.",
    "Anyone here using XYZ hosting? How's the uptime and support?",
    "I'm thinking of switching to a new hosting provider. What are your experiences with ABC hosting?",
    "Need a reliable hosting service for my new blog. Any suggestions?",
    "My website has been down several times this month due to hosting issues. Looking for a better option.",
    "What's the best hosting service for high traffic sites?",
    "Considering moving to a dedicated server. What hosting providers do you recommend?",
    "I've been using shared hosting but considering VPS. Is it worth the upgrade?",
    "What are the most important features to look for in a hosting service?",
    "Anyone used both SiteGround and Bluehost? Which one is better?",
    "How's the customer support with HostGator? Thinking of switching.",
    "What hosting service do you use for your WordPress site?",
    "I'm looking for an eco-friendly hosting provider. Any recommendations?",
    "What do you think of managed WordPress hosting vs. regular hosting?",
    "How do you handle backups with your hosting service?",
    "Looking for a hosting provider that offers good scalability. Suggestions?",
    "Any horror stories with cheap hosting services? Need advice.",
    "Is it worth paying extra for premium hosting plans?",
    "My hosting renewal is coming up. Should I stay with my current provider or switch?"
    "My website frequently experiences slow loading times with my current hosting provider. Any recommendations for a faster alternative?",
    "I'm a beginner looking to set up my first blog. What hosting provider would you recommend for someone new to web development?",
    "Need advice on choosing a reliable hosting service for an e-commerce site. Any suggestions?",
    "Experiencing frequent downtime with my current hosting service. Looking for a more stable option. Thoughts?",
    "Considering migrating my website to a cloud hosting platform. What are the benefits compared to traditional shared hosting?",
    "Looking for a cost-effective hosting solution for a small business website. Any affordable yet reliable options?",
    "Confused between managed and unmanaged hosting. Which one is better for a growing startup?",
    "My current hosting plan is expiring soon. Any good deals on hosting services right now?",
    "How do I transfer my website from one hosting provider to another without losing data?",
    "What are the security measures I should consider when choosing a hosting provider for a WordPress site?",
    "I've heard about VPS hosting but not sure if it's suitable for my personal blog. Any advice?",
    "Experienced issues with customer support from my hosting provider. How important is 24/7 support?",
    "Thinking about starting a podcast. Which hosting service would be best for uploading and distributing episodes?",
    "Need recommendations for hosting providers that support Python-based web applications.",
    "Is it worth investing in a dedicated server for a high-traffic website, or should I stick with shared hosting?",
    "Looking for hosting options that offer good scalability for future growth. Suggestions?",
    "Looking for a reliable hosting service. Any recommendations?",
    "Just started my blog and need advice on hosting options.",
    "Experiencing slow load times with my current hosting. Any better alternatives?",
    "Considering switching hosting providers due to frequent downtime.",
    "What are the pros and cons of cloud hosting versus shared hosting?",
    "Looking to upgrade my hosting plan. Any suggestions for affordable options?",
    "Seeking recommendations for hosting a small e-commerce site.",
    "Need advice on choosing the best hosting service for a startup.",
    "Exploring different hosting options for my personal website.",
    "Currently unhappy with my web hosting provider. Looking for alternatives.",
    "Can anyone suggest a good web hosting provider for a beginner?",
    "Thinking about switching to a VPS. Any recommendations?",
    "My current hosting plan doesn't meet my needs. Any better alternatives?",
    "Need help deciding between shared hosting and dedicated hosting.",
    "I am not satisfied with my current hosting provider. Any suggestions?",
    "Looking for reliable hosting options. Any advice would be appreciated.",
    "What hosting service would you recommend for a high-traffic website?",
    "Looking for affordable hosting options with good customer support.",
    "Considering different hosting options for a new project. Any tips?",
    "I need a reliable hosting service with good uptime. Any suggestions?",
    "Exploring cloud hosting solutions. Any recommendations?",
    "Seeking recommendations for hosting my personal blog.",
    "Can anyone suggest a hosting provider with strong security features?",
    "Looking for a hosting service that supports multiple domains.",
    "Need help choosing a hosting provider with excellent customer service.",
    "Considering a switch from my current hosting due to performance issues.",
    "Looking for hosting options that offer scalability for future growth.",
    "Need advice on migrating my website to a new hosting provider."

]

selected = 0
not_selected = 0
for text in tqdm(sample_texts, desc = "Processing"):
    select = Selection (model, text, base_keyword)
    if select == True:
        #print(f"Index {index} is {select} ✔️")
        selected += 1
    else:
        #print(f"Index {index} is {select} ✖️")
        not_selected += 1
        # Accuracy ..
Accuracy = (selected / not_selected) * 100
print(f"Accuracy is  ≈ {Accuracy:.3f}%")

# Specific Results: 
# Processing: 100%|██████████| 75/75 [01:17<00:00,  1.04s/it]
# Accuracy is  ≈ 87.500%
# Time Taken: 01:17s
        
