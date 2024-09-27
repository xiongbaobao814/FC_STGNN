import os  
import pandas as pd  

# 设置你的文件夹路径  
folder_path = '你的文件夹路径'  

# 使用pandas创建一个空的DataFrame来存储文件名  
df = pd.DataFrame(columns=['文件名'])  

# 遍历文件夹中的所有文件  
for filename in os.listdir(folder_path):  
    # 检查文件扩展名是否为.docx或.pdf  
    if filename.endswith(('.docx', '.pdf')):  
        # 将文件名添加到DataFrame中  
        df = df.append({'文件名': filename}, ignore_index=True)  
  
# 将DataFrame保存为Excel文件  
excel_path = '文件名列表.xlsx'  
df.to_excel(excel_path, index=False)  
  
print(f"文件名已保存到{excel_path}")