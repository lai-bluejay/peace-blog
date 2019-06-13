#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
source/trans2zhihu.py was created on 2019/06/13.
file in :relativeFile
Author: Charles_Lai
Email: lai.bluejay@gmail.com
"""
import os

root = os.path.dirname(os.path.abspath(__file__))
input_dir = root + '/_posts/深度学习'
output_dir = root + '/../zhihu_posts/'
import re
def repl(m):
    inner_word = m.group(1)
    return '<br><br>$' + inner_word + '$<br><br>'

def trans_head(line):
    if line.startswith('###'):
        line = line.replace('#', '').strip()
        line = "<h4>" + line + "</h4>"
    elif line.startswith('##'):
        line = line.replace('#', '').strip()
        line = "<h3>" + line + "</h3>"
    elif line.startswith('#'):
        line = line.replace('#', '').strip()
        line = "<h2>" + line + "</h2>"
    return line

for file in os.listdir(input_dir):
    with open(input_dir + '/' + file) as fi:
        text = fi.readlines()
        k = 0
        match = False
        flag = 0
        start_k = None
        end_k = None
        tmp_formula = []
        new_text = []
        while k < len(text):
            item = text[k]
            
            if item.strip() == "":
                k += 1
                continue

            elif item.startswith('#'):
                item = trans_head(item)
                new_text.append(item+'<br>')
                k += 1

            elif item.startswith('$$'):
                # new_text.append('<br>')
                if flag == 0:
                    flag = 1
                    k = k + 1
                elif flag == 1 and match is False:
                    match = True
                    new_text.append('<br>$' + " ".join(tmp_formula) + '$<br>')
                    tmp_formula = []
                    flag = 0
                    match = False
                    k += 1
                else:
                    print(k)
                    print(text[k+1])
                    print(text[k-1])
                    print(item)
                    print(flag)
                    print(match)
                    break
            else:
                if flag == 1:
                
                    tmp_formula.append(item.strip())
                    k += 1
                else:
                    new_text.append(item+'<br>')
                    k += 1
        
    with open(output_dir + '/' + file, 'w') as f_write:
        f_write.writelines(new_text)
