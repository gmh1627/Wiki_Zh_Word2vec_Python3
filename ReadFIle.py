import codecs,sys
import chardet

#rawdata = open('wiki.zh.simp.seg.txt', 'rb').read()
#result = chardet.detect(rawdata)
#print(result)
with open('wiki.zh.simp.seg.only_chinese.txt', 'r', encoding='utf-8') as f:
    for line in f:
        print(line)
#with open('wiki.zh.simp.seg.txt', 'r', encoding='GBK') as f:
#    for i in range(5):
#        print(repr(f.readline()))
#print(os.path.getsize('wiki.zh.simp.seg.txt'))
# 打开原文件，读取内容
"""
with codecs.open('wiki.zh.simp.seg.txt', 'r',encoding='ascii') as f:
    content = f.read()

# 以UTF-8编码写入新文件
with codecs.open('wiki.zh.simp.utf8.txt', 'w', encoding='utf-8') as f:
    f.write(content)
f = codecs.open('wiki.zh.simp.utf8.txt','r',encoding="utf-8")    
line = f.readline()
print(line)
"""