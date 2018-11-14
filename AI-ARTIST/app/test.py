import codecs

words=[]
with codecs.open('sth.txt','r','utf-8') as f:
    lines=f.read()
    print lines
    words=lines.split(',')