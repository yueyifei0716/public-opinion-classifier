import os
import csv

def merge_csv_files(directory, output_file):
    file_list = [file for file in os.listdir(directory) if file.endswith('.csv')]

    if not file_list:
        print("目录中没有CSV文件！")
        return

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        header_written = False

        for filename in file_list:
            with open(os.path.join(directory, filename), 'r') as infile:
                reader = csv.reader(infile)
                if not header_written:
                    header = next(reader)
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)

    print("CSV文件合并完成！")

# 示例用法
directory = 'sets/'  # 替换为实际的目录路径
output_file = 'sets/output.csv'  # 替换为输出的合并后的CSV文件路径

merge_csv_files(directory, output_file)
