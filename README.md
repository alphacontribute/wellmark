```# Session 1
# Introduction to NLP with spacy

import spacy

nlp=spacy.blank('en')

type(nlp)

 doc1=nlp('Today is Tuesday, 3rd January. We are starting NLP course using spacy')

type(doc1)

## Tokenization

![image-2.png](attachment:image-2.png)

for token in doc1:
    print(token)

doc1[0]

doc1[1]

doc1[2:7]

doc1[:8]

doc1[0]='Yesterday'

## Stop words

from spacy.lang.en.stop_words import STOP_WORDS

print(STOP_WORDS)

for token in doc1:
    print(token,'-',token.is_stop)

# session 2
# Preprocessing of text using spacy

import spacy

nlp=spacy.load('en_core_web_sm')

doc1=nlp('THE WRITING was on the wall — as early as almost half a century ago, when an 18-member committee warned that the town of Joshimath is “geologically unstable”, and suggested several restrictions and remedial measures. The committee, under the chairmanship of then Commissioner, Garhwal Mandal, Mahesh Chandra Mishra, was set up to probe the cause of landslides and sinking of Joshimath town. In its report dated May 7, 1976, it suggested restrictions on heavy construction work, agriculture on slopes, felling of trees; construction of pucca drainage to stop seepage of rainwater, proper sewage system, and cement blocks on river banks to prevent erosion.Amid the current crisis, the Congress and BJP are now blaming each other for failing to implement the recommendations of the report.')

type(doc1)

## Tokenization

for token in doc1:
    print(token)

# No of tokens

t_count=0
for token in doc1:
    t_count=t_count+1
    print(token)
print('\n\n The no of tokens:',t_count)

len(doc1)

## Is it stop word

from spacy.lang.en.stop_words import STOP_WORDS

print(STOP_WORDS)

len(STOP_WORDS)

# No of stop words in the documnt

s_count=0
for token in doc1:
    print(token,'==>',token.is_stop)
print('\n \n The non- stop words in the doc \n')
for token in doc1:
    if token.is_stop == False:
        s_count=s_count+1
        print(token)
print('\n The no of non-stop words:',s_count)

## Is it a punctuation

# No of non - punctations in the documnt

p_count=0
for token in doc1:
    print(token,'==>',token.is_punct)
print('\n \n The non- punctuations in the doc \n')
for token in doc1:
    if token.is_punct == False:
        p_count=p_count+1
        print(token)
print('\n The no of non-punctuations:',p_count)

## Is it a left puncutation

# No of left - punctations in the documnt

lp_count=0
for token in doc1:
    print(token,'==>',token.is_left_punct)
print('\n \n The left punctuations in the doc \n')
for token in doc1:
    if token.is_left_punct == True:
        lp_count=lp_count+1
        print(token)
print('\n The no of left punctuations:',lp_count)

## Is it a right punctuation

# No of right - punctations in the documnt

rp_count=0
for token in doc1:
    print(token,'==>',token.is_right_punct)
print('\n \n The right punctuations in the doc \n')
for token in doc1:
    if token.is_right_punct == True:
        rp_count=rp_count+1
        print(token)
print('\n The no of right punctuations:',rp_count)

## Is it an alphabet

# No of alphabets in the documnt

a_count=0
for token in doc1:
    print(token,'==>',token.is_alpha)
print('\n \n The alphabets in the doc \n')
for token in doc1:
    if token.is_alpha == True:
        a_count=a_count+1
        print(token)
print('\n The no of alphabets:',a_count)

## Is it a digit?

# No of digits in the documnt

d_count=0
for token in doc1:
    print(token,'==>',token.is_digit)
print('\n \n The digits in the doc \n')
for token in doc1:
    if token.is_digit == True:
        d_count=d_count+1
        print(token)
print('\n The no of digits:',d_count)

## Is it of lower case?

# No of lower cases in the documnt

l_count=0
for token in doc1:
    print(token,'==>',token.is_lower)
print('\n \n The lower cases in the doc \n')
for token in doc1:
    if token.is_lower == True:
        l_count=l_count+1
        print(token)
print('\n The no of lower case tokens:',l_count)

## Is it in upper case?

# No of upper cases tokens in the documnt

u_count=0
for token in doc1:
    print(token,'==>',token.is_upper)
print('\n \n The upper case tokens in the doc \n')
for token in doc1:
    if token.is_upper == True:
        u_count=u_count+1
        print(token)
print('\n The no of upper case tokens:',u_count)

## Is it a title case?

# No of title cases in the documnt

t_count=0
for token in doc1:
    print(token,'==>',token.is_title)
print('\n \n The title cases in the doc \n')
for token in doc1:
    if token.is_title == True:
        t_count=t_count+1
        print(token)
print('\n The no of title cases:',t_count)

## Is it a quote?


# No of quotes in the documnt

q_count=0
for token in doc1:
    print(token,'==>',token.is_quote)
print('\n \n The quotes in the doc \n')
for token in doc1:
    if token.is_quote == True:
        q_count=q_count+1
        print(token)
print('\n The no of quotes:',q_count)

## Is it like a number

# No of Numbers in the documnt

n_count=0
for token in doc1:
    print(token,'==>',token.like_num)
print('\n \n The numbers in the doc \n')
for token in doc1:
    if token.like_num == True:
        n_count=n_count+1
        print(token)
print('\n The count of numbers:',n_count)

doc2=nlp(' 2 is the second lowest positive integer')

for token in doc2:
    print(token,'=>',token.like_num)

for token in doc2:
    print(token,'=>',token.is_digit)

## Is it like a url?

doc3=nlp('The url of Times of India is www.timesofindia.com, which we tried to access')

for token in doc3:
    print(token.text,'==>',token.like_url)

## Is it like an Email ID?


doc4=nlp(' My Email ID is abc123@nmims.edu')

for token in doc4:
    print(token.text, '=>', token.like_email)

doc5=nlp(' My Email id is abc123@nmims.edu')

for token in doc5:
    print(token.text, '=>', token.like_email)

# Parts of Speech - POS

for token in doc1:
    print(token.text,'==>',token.pos_)

spacy.explain('ADP')

## POS in a DF

cols=['Token','POS','Explain_POS','Tag','Explain_Tag']
cols

rows=[]
for token in doc1:
    row=token,token.pos_,spacy.explain(token.pos_),token.tag_,spacy.explain(token.tag_)
    rows.append(row)
rows

import pandas as pd

token_df=pd.DataFrame(rows,columns=cols)
token_df

token_df['POS'].value_counts()

# session 3
# Spacy Pipeline

![image.png](attachment:image.png)

import spacy

nlp=spacy.load('en_core_web_sm')

doc_1=nlp('NMIMS is the all-encompassing educational platform for diverse fields of career such as Engineering, Science and Technology, Management, Commerce, Architecture, Law, Economics, Pharmacy, Aviation, Design, Performing Arts, Liberal Arts, Hospitality Management, Mathematical Science, Branding and Advertising, Agricultural Sciences and much more..Ideal for forward-thinking and new-age young professionals.Strong industry interface & corporate relationships with 1348+ companies.Research-backed updated curriculum, along with 2400+ published papers to its credit.Continuing the legacy of quality education.Focus on employability of students through evolved pedagogy.Pursuing excellence in technological and management research')

type(doc_1)

## Tokenizer

for token in doc_1:
    print(token)

## Stream of strings as input 

When there is a stream of strings as input, we need to 

use nlp.pipe() instead of nlp().



### List of strings

text_2=['Today is Monday','Tomorrow is Tuesday',
       'Yesterday was a holiday']

type(text_2)

text_2[0]

text_2[1]

for sentence in nlp.pipe(text_2):
    print(sentence)

# Tokens

for sentence in nlp.pipe(text_2):
    print(sentence)
    for token in sentence:
        print(token)

### Tuple of strings

text_3=('Today is Monday','Tomorrow is Tuesday',
       'Yesterday was Sundaya,a holiday')

type(text_3)

text_3[0]

for sent in nlp.pipe(text_3):
    print(sent)
    for token in sent:
        print(token)

### List of tuples

text_4=[('Today is Monday'),('Tomorrow is Tuesday'),
       ('Yesterday was Sundaya,a holiday')]

type(text_4)

text_4[0]

sent_count=0
for sent in nlp.pipe(text_4):
    sent_count=sent_count+1
    print(sent_count,'=>',sent)
    for token in sent:
        print(token)

### A DataFrame

text_2

import pandas as pd

text_df=pd.DataFrame(text_2,columns=['Sentence'])
text_df

text_df['Sentence']

for sent in nlp.pipe(text_df['Sentence']):
    print(sent)
    for token in sent:
        print(token)

doc_1

for token in doc_1:
    print(token)

## Separating doc into sentences

for sent in doc_1.sents:
    print(sent)

sent_count=0
for sent in doc_1.sents:
    sent_count=sent_count+1
    print(sent_count,'==>',sent)

## Tagger

for token in doc_1:
    print(token.text,'==>',token.tag_)

spacy.explain('NNS')

spacy.explain('VBG')

for token in doc_1:
    print(token.text,'==>',token.tag)

## POS

for token in doc_1:
    print(token.text,'==>',token.pos_)

for token in doc_1:
    print(token.text,'==>',token.pos)

## POS count

pos_count=doc_1.count_by(spacy.attrs.POS)
pos_count

for x,y in sorted(pos_count.items()):
    print(x,doc_1.vocab[x].text,y)

## Visualisation of POS

from spacy import displacy
displacy.render(doc_1,style='dep') # Dependence

options={'compact':'True','color':'blue'}

displacy.render(doc_1,style='dep',options=options)

## COnverting a text into a DF with tokens, pos 

text_df

token=[]
for sent in nlp.pipe(text_df['Sentence']):
    if sent.has_annotation('DEP'):
        token.append([word.text for word in sent] )
token

token=[]
pos=[]
for sent in nlp.pipe(text_df['Sentence']):
    if sent.has_annotation('DEP'):
        token.append([word.text for word in sent] )
        pos.append([word.pos_ for word in sent])
print(token)
print(pos)

# Updating text_df

text_df['Token']=token
text_df['POS']=pos

text_df

# session 4
# Parser and NER

![image.png](attachment:image.png)

import spacy

nlp=spacy.load('en_core_web_sm')

type(nlp)

doc1=nlp('Let’s start by differentiating between data analytics and traditional analytics. The terms are often used interchangeably, but a distinction does exist. Traditional data analytics refers to the process of analyzing massive amounts of collected data to get insights and predictions. Business data analytics (sometimes called business analytics) takes that idea, but puts it in the context of business insight, often with prebuilt business content and tools that expedite the analysis process..Specifically, business analytics refers to:.Taking in and processing historical business data.Analyzing that data to identify trends, patterns, and root causes.Making data-driven business decisions based on those insights.In other words, data analytics is more of a general description of the modern analytics process. Business analytics implies a narrower focus and has functionally become more prevalent and more important for organizations around the globe as the overall volume of data has increase')

doc1

## Tokenizer

for token in doc1:
    print(token)

## Tagger

for token in doc1:
    print(token.text,token.tag_)

## Parser

Tries to find the dependence between the tokens.

for token in doc1:
    print(token.text,'==>',token.dep_)

from spacy import displacy

displacy.render(doc1,style='dep')

spacy.explain('nsubj')

for token in doc1:
    print(token.text,'==>',token.head)

## Noun chunks

for chunk in doc1.noun_chunks:
    print(chunk.text, '==>',chunk.label_)

## NER

Named Entity Recognizer


doc1

doc2=nlp('Sport is a significant part of life in India. The country has a very long sports history, with sports being a part of tradition, culture, finance and entertainment. People in India closely follow various sports and enthusiastically participate in them. Cricket is the most popular spectator sport in the country, and citizens often play it as a recreational activity; it generates the highest television viewership, and features full-capacity audiences in stadiums during international and Indian Premier League (IPL) matches. It is part of popular culture. However, in more recent decades, football has also become another popular sport in terms of broadcast viewership and stadium audience attendance.[2][b] Kabaddi has grown into the mainstream, as well as badminton, tennis, and athletics. India are the one of the power houses in field hockey. India won World Cup & multiple medals in field hockey in Olympics. During that time, Dhyan Chand was a notable player. Sports such as swimming and badminton are played as recreational activities and for exercise.[')

doc2

for token in doc2:
    print(token.text)

for ent in doc2.ents:
    print(ent.text,'==>',ent.label_)

spacy.explain('NORP')

# List of entities

ent_list=[]
for ent in doc2.ents:
    ent_list.append(ent.label_)

print(ent_list)

# List of tuples of text and the respective entities

for ent in doc2.ents:
    print(ent.text, ent.label_)

ent_list=[(ent.text,ent.label_) for ent in doc2.ents]

print(ent_list)

## NER for web data

import requests

from bs4 import BeautifulSoup

url='https://en.wikipedia.org/wiki/India'

print(url)

request=requests.get(url)

print(request)

request=request.text
print(request)

soup_request=BeautifulSoup(request)
print(soup_request)

text= soup_request.body.text
print(text)

type(text)

### Converting str to doc using nlp

doc3=nlp(text)

type(doc3)

doc3

### Tokenizer

for token in doc3:
    print(token.text)

len(doc3)

## # List of entities

for ent in doc3.ents:
    print(ent.text,'==>',ent.label_)

displacy.render(doc3, style='ent')

ent_list=[]
for ent in doc3.ents:
    ent_list.append(ent.label_)
print(ent_list)

from collections import Counter
Counter(ent_list)

### Entities most appeared

most_ent=[]
for ent in doc3.ents:
    most_ent.append(ent.text)
print(most_ent)

# Most common

Counter(most_ent).most_common()

Counter(most_ent).most_common(10)

Counter(most_ent).most_common(20)

print(len(doc3.ents))

EXercise:

Scrap https://en.wikipedia.org/wiki/History_of_India

Find 10 most occurred named entities in the text.


# session 5
# Rule based matching

import spacy

nlp=spacy.load('en_core_web_sm')

type(nlp)

## Accessing a text

doc=nlp('''ChatGPT (Chat Generative Pre-trained Transformer)[1] is a chatbot launched by OpenAI in November 2022. It is built on top of OpenAI's GPT-3 family of large language models, and is fine-tuned (an approach to transfer learning)[2] with both supervised and reinforcement learning techniques.

ChatGPT was launched as a prototype on November 30, 2022, and quickly garnered attention for its detailed responses and articulate answers across many domains of knowledge. Its uneven factual accuracy was identified as a significant drawback.[3] Following the release of ChatGPT, OpenAI was valued at $29 billion.[4]

Training

Pioneer Building, San Francisco, headquarters of OpenAI

Sam Altman, CEO of OpenAI
ChatGPT was fine-tuned on top of GPT-3.5 using supervised learning as well as reinforcement learning.[5] Both approaches used human trainers to improve the model's performance. In the case of supervised learning, the model was provided with conversations in which the trainers played both sides: the user and the AI assistant. In the reinforcement step, human trainers first ranked responses that the model had created in a previous conversation. These rankings were used to create 'reward models' that the model was further fine-tuned on using several iterations of Proximal Policy Optimization (PPO).[6][7] Proximal Policy Optimization algorithms present a cost-effective benefit to trust region policy optimization algorithms; they negate many of the computationally expensive operations with faster performance.[8][9] The models were trained in collaboration with Microsoft on their Azure supercomputing infrastructure.

In addition, OpenAI continues to gather data from ChatGPT users that could be used to further train and fine-tune ChatGPT. Users are allowed to upvote or downvote the responses they receive from ChatGPT; upon upvoting or downvoting, they can also fill out a text field with additional feedback.[10][11][12]

Features and limitations

Cropped screenshot of a conversation with ChatGPT, December 30, 2022
Although the core function of a chatbot is to mimic a human conversationalist, ChatGPT is versatile. For example, it has the ability to write and debug computer programs; to compose music, teleplays, fairy tales, and student essays; to answer test questions (sometimes, depending on the test, at a level above the average human test-taker);[13] to write poetry and song lyrics;[14] to emulate a Linux system; to simulate an entire chat room; to play games like tic-tac-toe; and to simulate an ATM.[15] ChatGPT's training data includes man pages and information about Internet phenomena and programming languages, such as bulletin board systems and the Python programming language.[15]

In comparison to its predecessor, InstructGPT, ChatGPT attempts to reduce harmful and deceitful responses.[16] In one example, whereas InstructGPT accepts the premise of the prompt "Tell me about when Christopher Columbus came to the US in 2015" as being truthful, ChatGPT acknowledges the counterfactual nature of the question and frames its answer as a hypothetical consideration of what might happen if Columbus came to the U.S. in 2015, using information about Columbus' voyages and facts about the modern world – including modern perceptions of Columbus' actions.[6]

Unlike most chatbots, ChatGPT remembers previous prompts given to it in the same conversation; journalists have suggested that this will allow ChatGPT to be used as a personalized therapist.[17] To prevent offensive outputs from being presented to and produced from ChatGPT, queries are filtered through OpenAI's company-wide moderation API,[18][19] and potentially racist or sexist prompts are dismissed.[6][17]

ChatGPT suffers from multiple limitations. OpenAI acknowledged that ChatGPT "sometimes writes plausible-sounding but incorrect or nonsensical answers".[6] This behavior is common to large language models and is called hallucination.[20] The reward model of ChatGPT, designed around human oversight, can be over-optimized and thus hinder performance, otherwise known as Goodhart's law.[21] ChatGPT has limited knowledge of events that occurred after 2021. According to the BBC, as of December 2022 ChatGPT is not allowed to "express political opinions or engage in political activism".[22] Yet, research suggests that ChatGPT exhibits a pro-environmental, left-libertarian orientation when prompted to take a stance on political statements from two established voting advice applications.[23] In training ChatGPT, human reviewers preferred longer answers, irrespective of actual comprehension or factual content.[6] Training data also suffers from algorithmic bias, which may be revealed when ChatGPT responds to prompts including descriptors of people. In one instance, ChatGPT generated a rap indicating that women and scientists of color were inferior to white and male scientists.[24][25]

Service
ChatGPT was launched on November 30, 2022, by San Francisco-based OpenAI, the creator of DALL·E 2 and Whisper. The service was launched as initially free to the public, with plans to monetize the service later.[26] By December 4, OpenAI estimated ChatGPT already had over one million users.[10] CNBC wrote on December 15, 2022, that the service "still goes down from time to time".[27] The service works best in English, but is also able to function in some other languages, to varying degrees of success.[14] Unlike some other recent high-profile advances in AI, as of December 2022, there is no sign of an official peer-reviewed technical paper about ChatGPT.[28]

According to OpenAI guest researcher Scott Aaronson, OpenAI is working on a tool to attempt to watermark its text generation systems so as to combat bad actors using their services for academic plagiarism or for spam.[29][30] The New York Times relayed in December 2022 that the next version of GPT, GPT-4, has been "rumored" to be launched sometime in 2023.[17]

Reception and implications
Positive reactions
ChatGPT was met in December 2022 with generally positive reviews; The New York Times labeled it "the best artificial intelligence chatbot ever released to the general public".[31] Samantha Lock of The Guardian noted that it was able to generate "impressively detailed" and "human-like" text.[32] Technology writer Dan Gillmor used ChatGPT on a student assignment, and found its generated text was on par with what a good student would deliver and opined that "academia has some very serious issues to confront".[33] Alex Kantrowitz of Slate magazine lauded ChatGPT's pushback to questions related to Nazi Germany, including the claim that Adolf Hitler built highways in Germany, which was met with information regarding Nazi Germany's use of forced labor.[34]

In The Atlantic's "Breakthroughs of the Year" for 2022, Derek Thompson included ChatGPT as part of "the generative-AI eruption" that "may change our mind about how we work, how we think, and what human creativity really is".[35]

Kelsey Piper of the Vox website wrote that "ChatGPT is the general public's first hands-on introduction to how powerful modern AI has gotten, and as a result, many of us are [stunned]" and that ChatGPT is "smart enough to be useful despite its flaws".[36] Paul Graham of Y Combinator tweeted that "The striking thing about the reaction to ChatGPT is not just the number of people who are blown away by it, but who they are. These are not people who get excited by every shiny new thing. Clearly, something big is happening."[37] Elon Musk wrote that "ChatGPT is scary good. We are not far from dangerously strong AI".[36] Musk paused OpenAI's access to a Twitter database pending a better understanding of OpenAI's plans, stating that "OpenAI was started as open-source and non-profit. Neither is still true."[38][39] Musk had co-founded OpenAI in 2015, in part to address existential risk from artificial intelligence, but had resigned in 2018.[39]


Google CEO Sundar Pichai upended the work of numerous internal groups in response to the threat of disruption by ChatGPT.[40]
In December 2022, Google internally expressed alarm at the unexpected strength of ChatGPT and the newly discovered potential of large language models to disrupt the search engine business, and CEO Sundar Pichai "upended" and reassigned teams within multiple departments to aid in its artificial intelligence products, according to The New York Times.[40] The Information reported on January 3, 2023 that Microsoft Bing was planning to add optional ChatGPT functionality into its public search engine, possibly around March 2023.[41][42]

Stuart Cobbe, a chartered accountant in England & Wales, decided to the test the ChatGPT chatbot by entering questions from a sample exam paper on the ICAEW website and then entering its answers back into the online test. ChatGPT scored 42% which, while below the 55% pass mark, was considered a reasonable attempt.''')

doc

## How many sentences?


sent_count=0
for sent in doc.sents:
    sent_count=sent_count+1
    print(sent_count,'=>',sent)
print('Total no of sentences:',sent_count)

## Tokenization

for token in doc:
    print(token.text)

print(len(doc))

## Tagger

for token in doc:
    print(token.text,'=>',token.pos_)

## NER

for ent in doc.ents:
    print(ent.text,'=>',ent.label_)

## Rule based matching

doc

### Matching

  1) Token Matching 
  
  2) Phrase Matching
  
  3) Entity Matching

How ?

   1) Create an object/instance of the Matcher class.
   
   2) Define a pattern/rule.
   
   3) Add the pattern to the object
   
   4) Pass the document to the object

### Token matching

#### Occurance of the text ' ChatGPT'

from spacy.matcher import Matcher

# Create an instance of Matcher

matcher_1=Matcher(nlp.vocab)

# Define a pattern or a rule.

# A pattern is a list of dictionaries.

pattern_1=[{'text':'ChatGPT'}]

# Add pattern to the object

matcher_1.add('Pattern1',[pattern_1])

# Pass the doc to the object

match_1= matcher_1(doc)

print(len(match_1))

for match_id,start,end in match_1:
    span=doc[start:end]
    print(span.text)

## Phrase matching

#### Occurance of 'ChaptGPT is'

matcher_2=Matcher(nlp.vocab)
pattern_2=[{'text':'ChatGPT'},
          {'text':'is'}]
matcher_2.add('Pattern2',[pattern_2])
match_2=matcher_2(doc)

print(len(match_2))

for match_id,start,end in match_2:
    span=doc[start:end]
    print(span)

### Occurances of language/s, model's

matcher_3=Matcher(nlp.vocab)
pattern_3=[{'LEMMA':'language'},
          {'LEMMA':'model'}]


matcher_3.add('Pattern3',[pattern_3])
match_3=matcher_3(doc)

print(len(match_3))

for match_id,start,end in match_3:
    span=doc[start:end]
    print(span)

## Occurances of alphabets, digits 

matcher_4=Matcher(nlp.vocab)
pattern_4=[{'IS_ALPHA': True},
           {'IS_DIGIT':True}]


matcher_4.add('Pattern4',[pattern_4])
match_4=matcher_4(doc)

print(len(match_4))

for match_id,start,end in match_4:
    span=doc[start:end]
    print(span)

## Occurance of launch, discovery, find,,....

matcher_5=Matcher(nlp.vocab)
pattern_5=[{'LEMMA':
            {'IN':['launch','discover','find',
        'invent','create','develop','innovate',
                  'form','initiate']}}]

matcher_5.add('Pattern5',[pattern_5])
match_5=matcher_5(doc)

print(len(match_5))

for match_id,start,end in match_5:
    span=doc[start:end]
    print(span)

## Occurance of words of having length>15

matcher_6=Matcher(nlp.vocab)
pattern_6=[{'LENGTH':{'>=':15}}]
matcher_6.add('Pattern6',[pattern_6])
match_6=matcher_6(doc)

print(len(match_6))

for match_id,start,end in match_6:
    span=doc[start:end]
    print(span)

## Words of length 2

matcher_7=Matcher(nlp.vocab)
pattern_7=[{'LENGTH':{'==':2}}]
matcher_7.add('Pattern7',[pattern_7])
match_7=matcher_7(doc)

for match_id,start,end in match_7:
    span=doc[start:end]
    print(span)

# Occurance of Elon Musk

### Entity Matching

### Occurance of ent-type 'PERSON'

matcher_10=Matcher(nlp.vocab)
pattern_10=[{"ENT_TYPE":'PERSON'}]
matcher_10.add('Pattern10',[pattern_10])
match_10=matcher_10(doc)

print(len(match_10))

for match_id,start,end in match_10:
    span=doc[start:end]
    print(span.text)

Refer:
    https://spacy.io/usage/rule-based-matching


# session 6
# Vectorisation of tokens and similarity of documents

# !pip install gensim

import gensim
import spacy
nlp=spacy.load('en_core_web_sm')

## Create a list of texts

doc_1='''Chat GPT is a highly popular AI-based program that people use for generating dialogues. The chatbot has a language-based model that the developer fine-tunes for human interaction in a conversational manner. 
It’s a simulated chatbot primarily designed for customer service; people use it for various other purposes. But what is it? If you are new to this Chat GPT, this guide is for you, so continue reading. 
What’s Chat GPT?
Chat GPT is an AI chatbot auto-generative system created by Open AI for online customer care. It is a pre-trained generative chat, which makes use of (NLP) Natural Language Processing. The source of its data is textbooks, websites, and various articles, which it uses to model its own language for responding to human interaction.'''

doc_1

doc_2='''What is Chat GPT and why is everyone talking about it? On Twitter, blogs, and at the office, Chat GPT has taken over the conversation in marketing. However, not everyone is a fan.
So what is Chat GPT? Who better to ask than Chat GPT itself? 
ChatGPT is a variant of the GPT (Generative Pre-training Transformer) language model specifically designed for generating text in a chatbot-like manner. It is trained on a large dataset of human-human conversations and can generate natural language responses to input prompts.
In other words, it is a smart AI technology that will spit out factual, informative and well-written responses to given prompts. The technology presents endless potential with many applications to marketing including customer service, eCommerce, entertainment, resourcing and more! Along with these benefits, many professionals are questioning what such a helpful tool means for working freelancers and industry professionals.'''

doc_2

doc_3='''ChatGPT is a large language learning model that was designed to imitate human conversation. It can remember things you have said to it in the past and is capable of correcting itself when wrong.
It writes in a human-like way and has a wealth of knowledge because it was trained on all sorts of text from the internet, such as Wikipedia, blog posts, books, and academic articles.
It's easy to learn how to use ChatGPT, but what is more challenging is finding out what its biggest problems are. Here are some that are worth knowing about.
1. ChatGPT Isn't Always Right
It fails at basic math, can't seem to answer simple logic questions, and will even go as far as to argue completely incorrect facts. As social media users can attest, ChatGPT can get it wrong on more than one occasion.'''

doc_3

doc_4='''Texting, chatting and online messaging can be used for much more than simply communicating with your friends. Online communication can help young people build and develop social skills and gives them a platform to share their skills and help each other out.
Messaging and texting are among the most popular methods of communication among children and teenagers. A study by Common Sense Media in 2018 found that 70% of teenagers report using social media multiple times a day.
Messaging and texting can be much more than ways to communicate. They can also be tools that help young people learn and master important skills.'''

doc_4

# Creating the list

docs=[doc_1,doc_2,doc_3,doc_4]

print(docs)

## Choosing the tokens

texts=[]# List of all tokens
for document in docs:
    doc=nlp(document)
    text=[] # List of tokens in the document
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.like_num:
            text.append(token.lemma_)
    texts.append(text)

print(texts)

print(len(texts))

print(len(texts[0]))

print(len(texts[1]))

print(len(texts[2]))

print(len(texts[3]))

## Creation of a corpus

Corpus is a collection of tokens in a dictionary format.

from gensim.corpora import Dictionary

dict_1=Dictionary(texts)
print(dict_1)

## Giving an ID to each token

print(dict_1.token2id)

print(len(dict_1))

## Bag of words

bow_vec=[]
for token in texts:
    bow_vec.append(dict_1.doc2bow(token))

print(bow_vec)

## Creating BOW Matrix

from gensim.matutils import corpus2dense

bow_matrix=corpus2dense(bow_vec,num_terms=len(dict_1))

print(bow_matrix)

bow_matrix.shape

## TFIDF Vectorisation

Term Frequency Inverse Document Frequency

from gensim.models import TfidfModel
tfidf=TfidfModel(bow_vec)

print(tfidf)

tfidf_vec=[]
for vec in bow_vec:
    tfidf_vec.append(tfidf[vec])

print(tfidf_vec)

print(len(tfidf_vec))

print(gensim.__version__)

## Similarity of documents

from gensim.similarities import MatrixSimilarity

sim=MatrixSimilarity(tfidf_vec,num_features=len(dict_1))

print(sim)

print(sim[tfidf_vec[0]])

print(sim[tfidf_vec[3]])

# session 7
# Topic Modelling

#!pip install pyLDAvis
import pyLDAvis

import pyLDAvis.gensim_models

import spacy
import gensim

nlp=spacy.load('en_core_web_sm')

## Accessing texts

text_1='''chess, one of the oldest and most popular board games, played by two opponents on a checkered board with specially designed pieces of contrasting colours, commonly white and black. White moves first, after which the players alternate turns in accordance with fixed rules, each player attempting to force the opponent’s principal piece, the King, into checkmate—a position where it is unable to avoid capture.
Chess first appeared in India about the 6th century AD and by the 10th century had spread from Asia to the Middle East and Europe. Since at least the 15th century, chess has been known as the “royal game” because of its popularity among the nobility. Rules and set design slowly evolved until both reached today’s standard in the early 19th century. Once an intellectual diversion favoured by the upper classes, chess went through an explosive growth in interest during the 20th century as professional and state-sponsored players competed for an officially recognized world championship title and increasingly lucrative tournament prizes. Organized chess tournaments, postal correspondence games, and Internet chess now attract men, women, and children around the world.
This article provides an in-depth review of the history and the theory of the game by noted author and international grandmaster Andrew Soltis.
Characteristics of the game
Chess is played on a board of 64 squares arranged in eight vertical rows called files and eight horizontal rows called ranks. These squares alternate between two colours: one light, such as white, beige, or yellow; and the other dark, such as black or green. The board is set between the two opponents so that each player has a light-coloured square at the right-hand corner.
Algebraic notation
Individual moves and entire games can be recorded using one of several forms of notation. By far the most widely used form, algebraic (or coordinate) notation, identifies each square from the point of view of the player with the light-coloured pieces, called White. The eight ranks are numbered 1 through 8 beginning with the rank closest to White. The files are labeled a through h beginning with the file at White’s left hand. Each square has a name consisting of its letter and number, such as b3 or g8. Additionally, files a through d are referred to as the queenside, and files e through h as the kingside. See Figure 1.
Get a Britannica Premium subscription and gain access to exclusive content.
Subscribe Now
Moves
The board represents a battlefield in which two armies fight to capture each other’s king. A player’s army consists of 16 pieces that begin play on the two ranks closest to that player. There are six different types of pieces: king, rook, bishop, queen, knight, and pawn; the pieces are distinguished by appearance and by how they move. The players alternate moves, White going first.
King
White’s king begins the game on e1. Black’s king is opposite at e8. Each king can move one square in any direction; e.g., White’s king can move from e1 to d1, d2, e2, f2, or f1.
Rook
Each player has two rooks (formerly also known as castles), which begin the game on the corner squares a1 and h1 for White, a8 and h8 for Black. A rook can move vertically or horizontally to any unobstructed square along the file or rank on which it is placed.
Bishop
Each player has two bishops, and they begin the game at c1 and f1 for White, c8 and f8 for Black. A bishop can move to any unobstructed square on the diagonal on which it is placed. Therefore, each player has one bishop that travels only on light-coloured squares and one bishop that travels only on dark-coloured squares.
Queen
Each player has one queen, which combines the powers of the rook and bishop and is thus the most mobile and powerful piece. The White queen begins at d1, the Black queen at d8.
Knight
Each player has two knights, and they begin the game on the squares between their rooks and bishops—i.e., at b1 and g1 for White and b8 and g8 for Black. The knight has the trickiest move, an L-shape of two steps: first one square like a rook, then one square like a bishop, but always in a direction away from the starting square. A knight at e4 could move to f2, g3, g5, f6, d6, c5, c3, or d2. The knight has the unique ability to jump over any other piece to reach its destination. It always moves to a square of a different colour.
Capturing
The king, rook, bishop, queen, and knight capture enemy pieces in the same manner that they move. For example, a White queen on d3 can capture a Black rook at h7 by moving to h7 and removing the enemy piece from the board. Pieces can capture only enemy pieces.
Pawns
Each player has eight pawns, which begin the game on the second rank closest to each player; i.e., White’s pawns start at a2, b2, c2, and so on, while Black’s pawns start at a7, b7, c7, and so on. The pawns are unique in several ways. A pawn can move only forward; it can never retreat. It moves differently than it captures. A pawn moves to the square directly ahead of it but captures on the squares diagonally in front of it; e.g., a White pawn at f5 can move to f6 but can capture only on g6 or e6. An unmoved pawn has the option of moving one or two squares forward. This is the reason for another peculiar option, called en passant—that is, in passing—available to a pawn when an enemy pawn on an adjoining file advances two squares on its initial move and could have been captured had it moved only one square. The first pawn can take the advancing pawn en passant, as if it had advanced only one square. An en passant capture must be made then or not at all. Only pawns can be captured en passant. The last unique feature of the pawn occurs if it reaches the end of a file; it must then be promoted to—that is, exchanged for—a queen, rook, bishop, or knight.
Castling
The one exception to the rule that a player may move only one piece at a time is a compound move of king and rook called castling. A player castles by shifting the king two squares in the direction of a rook, which is then placed on the square the king has crossed. For example, White can castle kingside by moving the king from e1 to g1 and the rook from h1 to f1. Castling is permitted only once in a game and is prohibited if the king or rook has previously moved or if any of the squares between them is occupied. Also, castling is not legal if the square the king starts on, crosses, or finishes on is attacked by an enemy piece.
Relative piece values
Assigning the pawn a value of 1, the values of the other pieces are approximately as follows: knight 3, bishop 3, rook 5, and queen 9. The relative values of knights and bishops vary with different pawn structures. Additionally, tactical considerations may temporarily override the pieces’ usual relative values. Material concerns are secondary to winning.
Object of the game
When a player moves a piece to a square on which it attacks the enemy king—that is, a square from which it could capture the king if the king is not shielded or moved—the king is said to be in check. The game is won when one king is in check and cannot avoid capture on the next move; this is called checkmate. A game also can end when a player, believing the situation to be hopeless, acknowledges defeat by resigning.
There are three possible results in chess: win, lose, or draw. There are six ways a draw can come about: (1) by mutual consent, (2) when neither player has enough pieces to deliver checkmate, (3) when one player can check the enemy king endlessly (perpetual check), (4) when a player who is not in check has no legal move (stalemate), (5) when an identical position occurs three times with the same player having the right to move, and (6) when no piece has been captured and no pawn has been moved within a period of 50 moves.
In competitive events, a victory is scored as one point, a draw as half a point, and a loss as no points.'''

text_1

text_2='''Chess computers were first able to beat strong chess players in the late 1980s. Their most famous success was the victory of Deep Blue over then World Chess Champion Garry Kasparov in 1997, but there was some controversy over whether the match conditions favored the computer.
In 2002–2003, three human–computer matches were drawn, but, whereas Deep Blue was a specialized machine, these were chess programs running on commercially available computers.
Chess programs running on commercially available desktop computers won decisive victories against human players in matches in 2005 and 2006. The second of these, against then world champion Vladimir Kramnik is (as of 2019) the last major human-computer match.
Since that time, chess programs running on commercial hardware—more recently including mobile phones—have been able to defeat even the strongest human players.
MANIAC (1956)
In 1956 MANIAC, developed at Los Alamos Scientific Laboratory, became the first computer to defeat a human in a chess-like game. Playing with the simplified Los Alamos rules, it defeated a novice in 23 moves.[1]
Mac Hack VI (1966–1968)
In 1966 MIT student Richard Greenblatt wrote the chess program Mac Hack VI using MIDAS macro assembly language on a Digital Equipment Corporation PDP-6 computer with 16K of memory. Mac Hack VI evaluated 10 positions per second.
In 1967, several MIT students and professors (organized by Seymour Papert) challenged Dr. Hubert Dreyfus to play a game of chess against Mac Hack VI. Dreyfus, a professor of philosophy at MIT, wrote the book What Computers Can’t Do, questioning the computer's ability to serve as a model for the human brain. He also asserted that no computer program could defeat even a 10-year-old child at chess. Dreyfus accepted the challenge. Herbert A. Simon, an artificial intelligence pioneer, watched the game. He said, "it was a wonderful game—a real cliffhanger between two woodpushers with bursts of insights and fiendish plans ... great moments of drama and disaster that go in such games." The computer was beating Dreyfus when he found a move which could have captured the enemy queen. The only way the computer could get out of this was to keep Dreyfus in checks with its own queen until it could fork the queen and king, and then exchange them. That is what the computer did. Soon, Dreyfus was losing. Finally, the computer checkmated Dreyfus in the middle of the board.
In the spring of 1967, Mac Hack VI played in the Boston Amateur championship, winning two games and drawing two games. Mac Hack VI beat a 1510 United States Chess Federation player. This was the first time a computer won a game in a human tournament. At the end of 1968, Mac Hack VI achieved a rating of 1529. The average rating in the USCF was near 1500.[2]
Chess x.x (1968–1978)
In 1968, Northwestern University students Larry Atkin, David Slate and Keith Gorlen began work on Chess (Northwestern University). On 14 April 1970 an exhibition game was played against Australian Champion Fred Flatow, the program running on a Control Data Corporation 6600 model. Flatow won easily. On 25 July 1976, Chess 4.5 scored 5–0 in the Class B (1600–1799) section of the 4th Paul Masson chess tournament in Saratoga, California. This was the first time a computer won a human tournament. Chess 4.5 was rated 1722. Chess 4.5 running on a Control Data Corporation CDC Cyber 175 supercomputer (2.1 megaflops) looked at less than 1500 positions per second. On 20 February 1977, Chess 4.5 won the 84th Minnesota Open Championship with 5 wins and 1 loss. It defeated expert Charles Fenner rated 2016. On 30 April 1978, Chess 4.6 scored 5–0 at the Twin Cities Open in Minneapolis. Chess 4.6 was rated 2040.[3] International Master Edward Lasker stated that year, "My contention that computers cannot play like a master, I retract. They play absolutely alarmingly. I know, because I have lost games to 4.7."[4]
David Levy's bet (1978)
Main article: David Levy (chess player) § Computer chess bet
For a long time in the 1970s and 1980s, it remained an open question whether any chess program would ever be able to defeat the expertise of top humans. In 1968, International Master David Levy made a famous bet that no chess computer would be able to beat him within ten years. He won his bet in 1978 by beating Chess 4.7 (the strongest computer at the time).
Cray Blitz (1981)
In 1981, Cray Blitz scored 5–0 in the Mississippi State Championship. In round 4, it defeated Joe Sentef (2262) to become the first computer to beat a master in tournament play and the first computer to gain a master rating (2258).[5]
HiTech (1988)
In 1988, HiTech won the Pennsylvania State Chess Championship with a score of 4½–½. HiTech defeated International Master Ed Formanek (2485).[6]
The Harvard Cup Man versus Computer Chess Challenge was organized by Harvard University. There were six challenges from 1989 until 1995. They played in Boston and New York City. In each challenge the humans scored higher and the highest scorer was a human'''

text_2

text_3='''"It's like a game of chess," we used to say in days gone by.
Every move our politicians made could be analysed and interpreted, not only for its significance in the wider electoral tournament but also for the possible moves, or false moves, it might induce from opponents.
That all seems rather quaint now, viewed from our present standpoint where the table on which the chess board so precariously sits is being shaken by a constant bombardment of violent impacts: the very survival of the UK, the war on Islamic State and our future in or out of the EU.
No wonder the chess pieces are wobbling already and could soon be simply tossed up in the air to land who knows where.
And now, of course, there is a new player in the game. UKIP thinks the contest has for too long been the preserve of the same exclusive club of elite players.
"This is an unpredictable election," Ed Miliband told me in what could well go down as the understatement of last week in Manchester.
There really are so many imponderables piled one on top of another that assessing the likely outcomes next May looks less and less like a chess game and more and more like a mug's game.
For a start, this era of coalition government means that with the two biggest parties short of a Commons majority, they both have rival sets of target seats on the go at the same time. So they are both in the position of having to fight defensive and offensive campaigns simultaneously.
Our list of marginal seats here in the West Midlands shows that behind their ultra-close knife-edge marginals, Labour have some other narrow majorities to worry about before they start taking the Conservatives' chessmen off the board.
These numbers underline the extend to which the Midlands has traditionally been a predominantly two-party contest.
Liberal Democrat Lorely Burt stunned the Conservatives when she "crept in under the radar" in 2005 but now she has her work cut out to defend a majority of just 175 over the Conservatives.
How will the emergence of the Green Party, now the official opposition on Solihull Council, affect the chances of the larger parties?
Lorely Burt's party colleague John Hemming has turned Birmingham Yardley into something of a personal fiefdom, but Labour will be fighting hard to overturn his 3,002 majority and regain the seat he captured from former Education Secretary Estelle Morris in 2005.
The Liberal Democrats' only other Midlands seat is in Cheltenham where Martin Horwood will defend a more comfortable majority of 4,920 over the Conservatives in what is still the home of the Midlands' only Liberal Democrat-controlled council.
With confident predictions that the Liberal Democrats will lose many seats, perhaps it is to their traditional "core" constituencies that they may have to turn as they fight to limit the damage.
And into this otherwise two-way political street comes the new kid on the blog, UKIP, arguably the biggest imponderable of all.
If, as Nigel Farage says, they are not a repository simply for disillusioned Tory votes but a genuine mass party with broad appeal with their "tanks on Labour's lawn", how might they upset the two-party chess board?
Take Dudley North, for example, where Labour's wafer-thin majority over the Conservatives faces the additional challenge of the UKIP candidate Bill Etheridge MEP, who has turned his borough into a local power base.
'Politically toxic'
On the other hand, how will the Conservatives' tiny majority over Labour in Warwickshire North fare against not just an experienced Labour candidate, former minister Mike O'Brien, but also UKIP who have made great play of opposition to high-speed rail in an area where HS2 has become, to mix my analogies, politically toxic?
Or could the Midlands UKIP vote shrink, as it did in 2010 to just 4% after a performance in the previous summer's European elections which was impressive but not as emphatic as their clear victory in this year's EU poll?
The evidence of the local elections on the same day was that if you simply weigh the votes, it is the Conservatives who lose the most. But if you apply those numbers to real council areas (and Parliamentary constituencies?} you see it was Labour who suffered last May, failing to win majorities in target councils like Tamworth, Walsall and Worcester.
But in the remaining eight months before polling, perhaps the greatest imponderable of all is what politicians call "events" in a region which has always been particularly prone to the ups and downs of the economy.
The Birmingham and Solihull Chamber of Commerce has just reported a continuing surge in business confidence.
"Try telling that to young people in my constituency" says Ian Austin, the Labour MP for that key seat of Dudley North, where unemployment remains well above the national average and where wages continue to lag behind prices.
Put all these chess pieces together and you can see why even those politics watchers with the longest memories say they have never known an election as difficult to predict as this.
It helps to explain why the mood clearly detectable in Labour's ranks last week in Manchester was more uncertain than hopeful; why the Conservatives, behind Labour in the polls for so long, nevertheless closed their conference in Birmingham with more than a sneaking feeling that they could confound the sooth-sayers; and why I have it on good authority that senior Liberal Democrats are preparing to embark on their conference in Glasgow in a mood of innermost trepidation.
As the end game draws near, a game of chess has never looked more like a game of chance.....'''

text_3

# Create a list of texts

texts=[text_1,text_2,text_3]

print(texts)

## Creating a word list

words_list=[]
for text in texts:
    doc=nlp(text)
    text_words=[]
    for token in doc:
        if token.is_stop==False and token.is_punct==False and token.like_num==False and token.text!='\n':
            text_words.append(token.lemma_)
    words_list.append(text_words)

print(words_list)

print(len(words_list))

print(len(words_list[0]))

print(len(words_list[1]))

print(len(words_list[2]))

## Creating a corpus

corpus=[]
from gensim.corpora import Dictionary

dict=Dictionary(words_list)
type(dict)

for word in words_list:
    corpus.append(dict.doc2bow(word))

print(corpus)

len(corpus)

len(corpus[0])

len(corpus[1])

len(corpus[2])

## Creating an LDA model

lda=gensim.models.ldamodel.LdaModel(corpus=corpus,
                                   num_topics=5,
                                   id2word=dict)

type(lda)

## Displaying topics

lda.print_topics()

lda.print_topics()[:2]

## Getting topics for a word

lda.get_term_topics('game')

lda.get_term_topics('politics')

lda.get_term_topics('famous')

lda.get_term_topics('player')

## Visualisation of topics

pyLDAvis.enable_notebook()

plot=pyLDAvis.gensim_models.prepare(lda,
                                    corpus=corpus,
                                   dictionary=lda.id2word)

plot

# session 8
# Sentiment Analysis and Text Classification 

#!

from textblob import TextBlob

text1='Anuj looks bit sad today'

blob1=TextBlob(text1)

type(blob1)

# We want to have the sentiment

blob1.sentiment

# Another example

text2='The movie was the worst movie I have watched recently. I felt like cheated'

blob2=TextBlob(text2)

blob2.sentiment

text3='The movie was the worst and horrible movie I have watched recently. I felt like cheated. I recommend not to wacth'

blob3=TextBlob(text3)

blob3.sentiment

text4=' I was elated to receive the wonderful news that I have got a call from the bext company I wanted to join. I am so happy and want to share with all you my happiness by throwing a party'

blob4=TextBlob(text4)

blob4.sentiment

text5=' The sun is going to set at 6 PM'

blob5=TextBlob(text5)

blob5.sentiment



text6='''The Union Budget 2023, presented by the Finance Minister Nirmala Sitharaman proposed to extend the period of limitation on the assessment orders in her budget speech. It was proposed that an order of assessment may be passed within a period of 12 months from the end of the relevant assessment year or the financial year in which updated return is filed, as the case may be. It was also proposed that in cases where search under section 132 of the Income Tax Act, 1961 or requisition under section 132A of the Income Tax Act has been made, the period of limitation of pending assessments shall be extended by twelve months. In the Finance Bill of 2023, it was proposed to amend the section of 153 of the Income Tax Act.  In  sub-section (1A), for the words “nine months”, the words “twelve months” shall be substituted. It was proposed to extend the time limit to 12 months. Additionally, the time limit for completing an assessment or reassessment, as applicable, is extended by 12 months if it is ongoing on the date that a search under Section 132 is commenced or a requisition under Section 132A is made. In this proposal, mainly the time limit has been extended and which would be helpful for the officers to figure out the things and proceed with proper procedure. Additionally, it was proposed to substitute “Principal Chief Commissioner or Chief Commissioner or Principal Commissioner or Commissioner, as the case may be,” for the phrases “Principal Commissioner or Commissioner” at both of their places. As reported before in the case Smt. Rashidaben Taher Morawala Badri Mohalla vsThe DCIT, CITATION: 2022 TAXSCAN (ITAT) 1772, the Division Bench of the Income Tax Appellate Tribunal (ITAT), Ahmedabad quashed the assessment order passed in violation of time limit under Section 153(1) of the Income Tax Act, 1961.

Read More: https://www.taxscan.in/union-budget-2023-period-of-limitation-on-pending-assessment-extended-by-12-months/250888/'''

blob6=TextBlob(text6)

blob6.sentiment

# Text Classification

from sklearn.datasets import fetch_20newsgroups

train=fetch_20newsgroups(subset='train')

test=fetch_20newsgroups(subset='test')

type(train)

train.keys()

train



train['target_names']

train['target']

import numpy as np

np.unique(train['target'])

train['data']

print(train['data'][1])

print(train['target_names'][1])

print(train['data'][10])

print()

print(train['target_names'][10])

print(train['data'][100])

print()



## Building the model

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import make_pipeline

mnb=make_pipeline(TfidfVectorizer(),MultinomialNB())

type(mnb)

# Training

mnb.fit(train['data'],train['target'])

# Prediction

y_pred=mnb.predict(test['data'])

y_pred

## Performance of the model

from sklearn.metrics import classification_report,confusion_matrix

report=classification_report(test['target'],y_pred)

print('The report:\n',report)

# Confusion Matrix

cm=confusion_matrix(test['target'],y_pred)

print('The confusion Matrix:\n',cm)

## Testing with our own data

def news_group_prediction(doc):
    group_pred=mnb.predict([doc])
    return test['target_names'][group_pred[0]]

news_group_prediction('Computer technology is becoming more user friendly. Windows operating systems are easy to operate')

news_group_prediction('The soocer is the one sport most of Eurpeaons follow. It is quite a big business, which involved lot of anlaytucs as well  ')


# session 9
# Text Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Accessing the dataset

bbc=pd.read_csv('bbc-text.csv')

bbc

bbc['category'].value_counts()

## Selecting the data

data=bbc['text']

data

data[0]

data[1]

data[10]

data[100]

data[1000]

## Converting text into vectors

from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer()

features=tf.fit_transform(data)

features

print(features)

## Clustering using K Means

from sklearn.cluster import KMeans


SSD=[]
for k in range(1,10):
    kmeans=KMeans(n_clusters=k, random_state=10)
    kmeans.fit(features)
    SSD.append(kmeans.inertia_)
plt.plot(range(1,10),SSD);

## Applying silhouette_score
from sklearn.metrics import silhouette_score
SS=[]
for k in range(2,11):
    kmeans=KMeans(n_clusters=k, random_state=10)
    kmeans.fit(features)
    SS.append(silhouette_score(features,kmeans.predict(features)))
    
plt.plot(range(2,11),SS);

## Building the model with 5 clusters

kmeans=KMeans(n_clusters=5,random_state=10)
kmeans.fit(features)


kmeans.labels_

## Updating the DF with cluster labels

bbc['Cluster']=kmeans.labels_

bbc

## Dimensionality reduction using TSNE

from sklearn.manifold import TSNE

tsne=TSNE(n_components=2,perplexity=30,random_state=10)

features_tsne=tsne.fit_transform(features)

features_tsne

features_tsne.shape

## Visualisation 

plt.figure(figsize=(10,8))
plt.scatter(features_tsne[:,0],features_tsne[:,1]);

plt.figure(figsize=(10,8))
plt.scatter(features_tsne[:,0],features_tsne[:,1],c=bbc['Cluster']);

