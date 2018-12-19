"""
pandas的基本使用
"""

import pandas as pd
import numpy as np

# 创建一个基本序列是一个空序列。pandas.Series( data, index, dtype, copy)。
sEmprty = pd.Series()
print(sEmprty)

# 通过一个数组创建序列，如果传入索引长度要跟数组一样，不传的话就是0...n
data = np.array(['a','b','c','d'])
sArr = pd.Series(data)
print(sArr)

# 通过字典创建一个序列
data = {
    'a':0., 
    'b':1., 
    'c':2.}
sDic = pd.Series(data)
print(sDic)
 