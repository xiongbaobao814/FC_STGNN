# 打开文件  
with open('./X_train.txt.txt', 'r') as file:  
    # 读取第一行  
    first_line = file.readline().strip()  
    # 如果文件为空，则直接结束  
    if not first_line:  
        print("文件为空或没有数据行")  
        exit()  
    # 以逗号分隔第一行，并计算分隔后的元素数量，即列数  
    num_columns = len(first_line.split(' '))  
    print(f'列数: {num_columns}')