import json
from bs4 import BeautifulSoup
import pandas as pd


def document_merge() -> dict:
    data = {}
    # 完善代码
    mouth = ['january', 'february', 'may']
    
    # json数据处理
    json_data = open('/home/project/2022_february.json', 'r', encoding='utf-8').read()

    data[mouth[1]] = json.loads(json_data)['february']
    # xlsx数据处理
    xlsx_data = pd.read_excel('/home/project/2022_january.xlsx',usecols='C:I',header=3)
    table_head = xlsx_data[0:1].values[0]
    table_data = xlsx_data[1:].values
    data[mouth[0]] ={}
    for i in range(len(table_data)):
        date = str(table_data[i][0].month) + '-' +str(table_data[i][0].day)
        tmp ={}
        for j in range(1,len(table_head)):
            tmp[table_head[j]] = table_data[i][j]
        data['january'][date] = tmp
    
    # html数据处理
    # 读取html数据
    html_data = open('/home/project/2022_may.html', 'r', encoding='utf-8').read()
    # 解析html数据
    soup = BeautifulSoup(html_data, 'html.parser')
    # 选取每个tr元素
    trs = soup.find_all('tr')
    # print(trs)
    # 划分数据 把字典的选取出来
    head = trs[0].text.replace('\n',' ').strip().split()[1:]
    # print(head)
    content = trs[1:]
    # 处理数据
    # i.text= ['\n5-1\n673\n620\n1891\n668\n766\n3145\n', '\n5-2\n522\n2025\n139\n569\n890\n2159\n',……]
    solution_content = [i.text.replace('\n', ' ').strip() for i in content]
    # print(solution_content)
    data[mouth[2]] ={}
    
    for i in solution_content:
        temp = i.split()
        data[mouth[2]][temp[0]]={key:int(val) for key,val in zip(head,temp[1:])}
        
    return data