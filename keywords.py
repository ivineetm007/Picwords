import re
from collections import Counter

output_file = open('most_freq_words.txt','w') 

with open('stopwords.txt') as f:
	aStopWords = f.read().splitlines()

text_file = open('textfile.txt','r') 
sText = text_file.read() 
sText = re.sub('[\W_]+', ' ', sText, flags=re.UNICODE) 
sText = sText.lower()

aWords = sText.split(" ")
aWords  = [word for word in aWords if word.lower() not in aStopWords]

counts = Counter(aWords)
words = sorted(counts, key=lambda word: (-counts[word], word))


for word in words:
	if word !="":
		output_file.write(word+'\n')

output_file.close()
text_file.close()