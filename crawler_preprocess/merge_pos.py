import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('rmrb.csv')
df2 = pd.read_csv('rmw.csv')

# 合并两个数据框
merged_df = pd.concat([df1, df2], ignore_index=True)

# 选择label为1的前30000条数据
selected_data = merged_df[merged_df['label'] == 1].head(30000)

# 将结果保存到新的CSV文件
selected_data.to_csv('pos_posts_weibo.csv', index=False)
