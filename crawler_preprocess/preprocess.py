import pandas as pd
import re

# 读取CSV文件
df = pd.read_csv('sets/output.csv')

# 移除链接
df['Content'] = df['Content'].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))

# 移除转发内容
df['Content'] = df['Content'].apply(lambda x: re.sub(r'RT @\S+', '', x))

# 删除Content字数少于20的行
df = df[df['Content'].str.len() >= 20]

# 添加新列'label'，并将其值设为0
df['label'] = 0

# 将修改后的数据保存回CSV文件
df.to_csv('sets/output_filtered.csv', index=False)

