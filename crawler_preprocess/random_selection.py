import csv
import random

def random_sample(input_file, output_file, sample_size=20):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 读取列名

        rows = list(reader)  # 读取所有数据行

        if len(rows) <= sample_size:
            print("样本数量超过CSV文件中的数据行数！")
            return

        random.shuffle(rows)  # 随机打乱数据行

        sample = rows[:sample_size]  # 选择前 sample_size 条数据行作为样本

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 写入列名
        writer.writerows(sample)  # 写入样本数据行

    print("随机抽样完成！")

# 示例用法
input_file = 'output.csv'  # 替换为实际的输入CSV文件路径
output_file = 'neg_sample.csv'  # 替换为实际的输出CSV文件路径
sample_size = 20  # 要抽样的数量

random_sample(input_file, output_file, sample_size)
