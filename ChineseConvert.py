from opencc import OpenCC

# 初始化转换器，t2s表示从繁体转简体
cc = OpenCC('t2s')

# 打开简体中文文档进行写入
with open('wiki.zh.simp.txt', 'w', encoding='utf-8') as out_f:
    # 分批读取繁体中文文档
    with open('wiki.zh.txt', 'r', encoding='utf-8') as in_f:
        for line in in_f:
            # 转换为简体中文
            simplified_chinese = cc.convert(line)
            # 写入简体中文
            out_f.write(simplified_chinese)