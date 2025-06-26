import pandas as pd

# 读取 CSV 文件（替换为你的文件路径）
csv_path = 'do_425012.csv'
df = pd.read_csv(csv_path)

# 检查列是否存在，避免报错
if 'MeanDischargeRate' in df.columns:
    # 删除指定列
    df = df.drop(columns=['MeanDischargeRate'])
    print("成功移除 'MeanDischargeRate' 列")
else:
    print("列 'MeanDischargeRate' 不存在")

# 保存处理后的数据到新文件
output_path = '425012.csv'
df.to_csv(output_path, index=False)
print(f"处理后的数据已保存至 {output_path}")