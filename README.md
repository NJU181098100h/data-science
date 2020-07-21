# 数据科学基础 大作业

[toc]

## 小组信息

### 小组成员

| 学号      | 姓名   | 邮箱                       | Python练习完成题目数量 |
| --------- | ------ | -------------------------- | ---------------------- |
| 181250194 | 赵一鸣 | 181250194@smail.nju.edu.cn | 200                    |
| 181870006 | 蔡晓星 | 181870006@smail.nju.edu.cn | 200                    |
| 181098100 | 胡家铭 | 181098100@smail.nju.edu.cn | 190                    |

### 组员分工职责



## 研究问题

### 题目视角

详细分析在"研究方法"模块

1. 题目难度分析

综合题目的各项指标，如平均分数、完成率、代码平均运行时间等指标，对每道题目给出一个难度系数的评估分数，分数越高说明题目越难。这一分析有助于老师合理控制布置的题目的难度，使其在一个合理的区间范围内。

2. 基于学生答题表现的聚类分析

综合题目的各项指标，运用主成分分析、深度自编码器、变分自编码器对数据进行降维，接着再对题目进行聚类分析，聚类后在同一类中的题目，学生在这些题目上的表现更加接近。这一分析能够帮助老师分析学生在哪些题目上的表现相似，从而制定更加合理的教学计划。

3. "好题"、"坏题"鉴定器

分析一道题目出的好不好，比如说，如果一道题目的得分的方差太小或做题人数过少，则说明这道题出的不好，实际操作时，考虑到的指标不止上述两个。这一研究能够帮助老师挑选好的编程题，从而更加有利于学生编程能力的提升。

4. 面向测试用例编程检测器

此次编程作业中，有很多人会套测试用例，从而"面向测试用例编程"，这种行为不利于学生真正掌握编程，所以写了一个面向测试用例编程检测器，从而能够筛选出这些提交记录并记为0分。

### 学生视角

详细分析在"研究方法"模块

1. 编程能力评估

2. 生成编程学习路线

3. 寻找编程搭档

4. 自动推荐代码

## 开源代码地址

https://github.com/SWargrave/data-science

## 研究方法

### 开发环境

#### 集成开发环境

<img src="https://i.loli.net/2020/07/21/2X5MkvGUrDdnQHq.png" alt="1200px-Jupyter_logo.svg.png" style="zoom:12.5%;" />

<img src="https://i.loli.net/2020/07/21/byHoIBA5wmdar8l.png" alt="PyCharm_Logo.svg.png" style="zoom:12.5%;" />

由于此次大作业是一个数据分析项目，不同于传统的编程或web开发，所以决定采用"最适合数据科学"的`jupyter`+`PyCharm`作为开发环境，不同于传统的python开发，`jupyter`有更好的交互体验，可以实时查看内存中的变量，这些变量都会被保存在`Jupyter Runtime`中，而且代码运行结果可以实时反馈，不仅可以保存代码，还可以保存中间输出结果，充分利用了`python`的解释性语言的特性。所以这个项目的所有源代码都是`.ipynb`格式的文件而非`.py`。

#### 导入必要的python库

```python
import json # 处理json数据
import numpy as np 
from sklearn.preprocessing import OneHotEncoder,StandardScaler # one-hot编码处理类别数据，数据归一化工具
from sklearn.decomposition import PCA # 主成分分析
import csv # 处理csv数据
import pandas as pd 
import time # 时间数据处理
import datetime # 时间数据处理
import torch # numpy数组数据转为tensor张量形式，方便后续处理
from torch import nn # 用于帮助构建深度自编码器和变分自编码器
import torch.nn.functional as F # 用于帮助构建深度自编码器和变分自编码器
from torch.autograd import Variable # 可求导变量
from torch.utils.data import DataLoader,TensorDataset,RandomSampler # 数据集制作工具
import os
import urllib.request,urllib.parse # 下载题目
import zipfile # 解压缩文件
import win32api
import random
```

![download.jpg](https://i.loli.net/2020/07/21/1B2WrXFQ7feO6vd.jpg)

![download.png](https://i.loli.net/2020/07/21/pzAhuLsawOoTDvj.png)

`NumPy`可以用来存储和处理大型矩阵，支持大量的维度数组和矩阵运算，`pandas`是为了解决数据分析任务而创建的，能够高效地操作大型数据集。在后续的分析建模中，我们会把数据集转换成csv格式的文件，`Numpy`和`pandas`可以高效地处理这些数据。

![download.png](https://i.loli.net/2020/07/21/GNuL73Y8DT2ms5n.png)

`scikit-learn`库提供了非常方便的数据预处理函数和基本的机器学习函数，比如独热编码、数据归一化、主成分分析等模块。

![download.png](https://i.loli.net/2020/07/21/MwbY5P4jzqZUQ2x.png)

`PyTorch`模块提供了构建神经网络、自动求导等工具，本项目中会使用深度自编码器和变分自编码器，所以需要引入这个库，此外，`PyTorch`还提供了便捷地构建数据集的工具。

### 数据预处理

#### 学生编号、题目编号、题目类型处理

在`test_data.json`中，`user_id`和`case_id`都是字符串类型的数据，且没有从0开始每次加1编号，于是我们就把`user_id`和`case_id`重新编号，都从0开始，每次加1，一共有271名学生，882道题，所以处理过后的学生编号为0到270，题目编号为0到881。

```python
userIds=[str(i) for i in sorted([int(i) for i in list(test_data.keys())])]
updateUserIds={}
getOldUserId=[]
for i in range(len(userIds)):
    updateUserIds.update({userIds[i]:i})
    getOldUserId.append(userIds[i])
```

```python
for i in test_data:
    for j in test_data[i]['cases']:
        case_ids.add(j['case_id'])
case_ids=sorted(list(case_ids))
updateCaseIds={}
getOldCaseId=[]
for i in range(len(case_ids)):
    updateCaseIds.update({case_ids[i]:i})
    getOldCaseId.append(case_ids)
```

`updateUserIds`和`updateCaseIds`是字典，字典的key是原来的`id`，即`test_data.json`中的`user_id`和`case_id`，字典的value是重新编号后的`user_id`和`case_id`，由这两个字典可以根据旧的编号获得重新编号后的新编号；`getOldUserId`和`getOldCaseId`是list类型的变量，长度分别为271和882，由这两个list可以根据新的编号获取旧的编号。

题目类型一共有8种，分别为排序算法、查找算法、图结构、树结构、数字操作、字符串、线性表、数组，把这八种类型分别用数字0~7表示。

```python
type_dict={'排序算法':0,'查找算法':1,'图结构':2,'树结构':3,'数字操作':4,'字符串':5,'线性表':6,'数组':7}
```

#### `test_data.json`数据集重构

由于user_id已经重新编号为0~270，所以可以把`test_data.json`所包含的所有内容保存到一个数组中，该数组名为`new_data`。

```python
new_data=[]
for i in range(271):
    new_data.append(test_data[getOldUserId[i]]['cases'])
del test_data
```

假设`x`是0~270中的整数，则`new_data[x]`为`user_id`为`x`的学生所做的所有题目，即`new_data[x]`和下图中的`"cases"`对应的数组相对应。

![image.png](https://i.loli.net/2020/07/21/VFNoqMpTEjrAZQP.png)

#### 变量说明：`userFinishCaseIds`

由于使用了交互式开发环境，所以可以在`Jupyter Runtime`中保存一些变量为后续数据整理和清洗提供方便，下面就说明一下这些有用的变量。

`userFinishCaseIds`是一个列表，长度为271，记录了每个学生所做的题目的编号，这个列表的每一个元素也是一个列表。比如`userFinishCaseIds[x]`是一个列表，列表的元素是整数类型，这个列表保存了编号为`x`的学生完成的题目的编号。

```python
userFinishCaseIds=[]
for i in range(271):
    tempList=[]
    for j in range(len(new_data[i])):
        tempList.append(new_data[i][j]['case_id'])
    userFinishCaseIds.append(sorted(tempList))
```

#### 变量说明：`groupCaseIds`

`groupCaseIds`是一个列表，由于学生和题目都分成了五组，也就是说不是每个学生都要做882道题目，每道题目也不是都应该被271名学生做。所以`groupCaseIds`记录了每组的题目编号，`groupCaseIds`的长度为5，`groupCaseIds[x]`是一个长度为200的列表（其中`x`的取值范围为0~4），表示第`x`组的所有题目的编号。

提供的数据集`test_data.json`中并未包含分组信息，具体如何分组的方法如下：首先找出完成了200道题目的学生（利用`userFinishCasesIds`即可），共有80名学生完成了全部的题目，接着把这80名学生按所做的题目的编号分类，即把做的题目编号完全相同的学生归为一类，结果得到这80名学生一共可以归为四组，于是我们就得到了前四组的题目编号，即`groupCaseIds`的前四个元素已经求出来了。下面找出第五组的题目编号：所有的题目一共有882道，把这882道题目中出现在前四组的题目都删去，剩下的题目一定在第五组中，接着利用`userFinishCaseIds`找出做了第五组的题目学生编号，这些学生一定是第五组的，再把这些学生做的所有题目编号纳入`groupCaseIds[4]`，即第五组的题目编号列表中，这样就可以得到每一组的题目编号了。

具体的代码如下，最后需要把中间变量`fifthGroupIds`、`theFifthGroupCaseIds`、`ids882`、`caseIds200`删除以节省内存。

```python
caseIds200=[]
groupCaseIds=[]
ids882=[i for i in range(882)]
theFifthGroupCaseIds=set()
for i in range(271):
    ids='.'.join([str(i) for i in userFinishCaseIds[i]])
    if len(userFinishCaseIds[i])==200 and ids not in caseIds200:
        caseIds200.append(ids)
        groupCaseIds.append(userFinishCaseIds[i])
        for j in userFinishCaseIds[i]:
            ids882[j]=-1
for i in ids882:
    if i!=-1:
        theFifthGroupCaseIds.add(i)
fifthGroupIds=list(theFifthGroupCaseIds)
for i in range(271):
    if 190<=len(userFinishCaseIds[i])<=200:
        flag=False
        for j in theFifthGroupCaseIds:
            if j in userFinishCaseIds[i]:
                flag=True
                break
        if flag:
            for j in userFinishCaseIds[i]:
                fifthGroupIds.append(j)
groupCaseIds.append(sorted(list(set(fifthGroupIds))))
del fifthGroupIds
del theFifthGroupCaseIds
del ids882
del caseIds200
```

#### 变量说明：`groupUserIds`、`ctGroupUserIds`、`validUserIds`、`groupUserNum`

由于已经对题目编号进行了分组，下面可以利用`groupCaseIds`和`userFinishCaseIds`来对学生进行分组，如果学生所做的所有题目都包含在某一组中，则该学生一定是该组的，在对学生进行分组的过程中，若某个学生做的题目太少，则可能会出现该学生可以属于不止一个组的情况，所以解决方法是声明两个数组变量`ctGroupUserIds`和`validUserIds`，分别表示无法分类或做题太少的学生编号，这些学生的数据被标记为无效数据，而其余的学生编号会被加入到`validUserIds`中，表示是有效的数据。具体的操作为：如果一个学生做题数量小于等于10或者无法被分到唯一的一组，则会被标记为无效，其编号会被加入到`ctgroupUserIds`，否则加入到`validUserIds`，最后的结果为有效学生数量为254，无效学生数量为17，之后的所有分析都将忽略这17条数据。`groupUserNum`是一个长度为5的列表，记录了每组的学生人数。

```python
groupUserIds=[[] for i in range(5)]
ctGroupUserIds=[]
validUserIds=[]
for i in range(271):
    if len(userFinishCaseIds[i])>200:
        groupUserIds[4].append(i)
        continue
    if len(userFinishCaseIds[i])<=10:
        ctGroupUserIds.append(i)
        continue
    if i==261:
        groupUserIds[3].append(i)
        continue
    g=[]
    for j in range(5):
        flag=True
        for k in userFinishCaseIds[i]:
            if not k in groupCaseIds[j]:
                flag=False
                break
        if flag:
            g.append(j)
    if len(g)==1:
        groupUserIds[g[0]].append(i)
    else:
        ctGroupUserIds.append(i)
for i in groupUserIds:
    for j in i:
        validUserIds.append(j)
validUserIds=sorted(validUserIds)
groupUserNum=[len(i) for i in groupUserIds]
```

#### 变量说明：`caseGroups`、`caseUserNum`、`caseUserNumInFact`、`caseFinishRate`、`caseScoreIgnoreUndo`、`caseScoreCountUndo`、`caseIdsByType`

由于一道题目可能出现在不止一组中，所以就用`caseGroups`来记录每道题所属的组，`caseGroups`是一个列表，长度为882，列表的元素也是列表，其元素为整数，表示所在的组，长度最大为5，比如，若`caseGroups[x]`的值为`[1,3]`，表示编号为`x`的题目被包含在第1、3组中。

`caseUserNum`是一个长度为882的列表，记录了每道题应该完成的学生人数，即把每道题所在的小组的学生人数相加的结果，可以利用`caseGroups`和`groupUserNum`求得到。

`caseUserNumInFact`是一个长度为882的列表，记录了每道题实际上有多少人做了，因为不是每个学生都做满了200道题。该变量可利用`userFinishCaseIds`求得。

`caseFinishRate`是一个长度为882的列表，记录了每道题的完成率，即用实际做了这道题的人数/应该做这道题的人数。

`caseScoreIgnoreUndo`是一个长度为882的列表，记录了每道题的平均得分，会忽略没有做这道题的人，即用这道题目的总得分/实际做的人。

`caseScoreCountUno`是一个长度为882的列表，记录了每道题的平均得分，如果本应做而没有做这道题的学生，此题得分会以0分计入。

`caseIdsByType`是一个长度为8的列表（8：一共由8种类型的题目），列表元素也是列表，记录了每种类型的所有题目的编号，`caseIdsByType[x]`表示所有种类为`x`的题目的编号。

```python
caseGroups=[[] for i in range(882)]
caseUserNum=[0 for i in range(882)]
caseUserNumInFact=[0 for i in range(882)]
caseScoreIgnoreUndo=[0 for i in range(882)]
caseScoreCountUndo=[0 for i in range(882)]
caseIdsByType=[set() for i in range(8)]
for i in range(882):
    for j in userFinishCaseIds:
        if i in j:
            caseUserNumInFact[i]+=1
    for j in range(len(groupCaseIds)):
        if i in groupCaseIds[j]:
            caseGroups[i].append(j)
            caseUserNum[i]+=groupUserNum[j]
caseFinishRate=list(np.array(caseUserNumInFact)/np.array(caseUserNum))
for i in range(len(new_data)):
    if i not in validUserIds:
        continue
    for j in range(len(new_data[i])):
        caseScoreIgnoreUndo[new_data[i][j]['case_id']]+=new_data[i][j]['final_score']
caseScoreCountUndo=list(np.array(caseScoreIgnoreUndo)/np.array(caseUserNum))
caseScoreIgnoreUndo=list(np.array(caseScoreIgnoreUndo)/np.array(caseUserNumInFact))
for i in range(len(new_data)):
    for j in range(len(new_data[i])):
        new_data[i][j]['case_type']=type_dict[new_data[i][j]['case_type']]
        caseIdsByType[new_data[i][j]['case_type']].add(new_data[i][j]['case_id'])
for i in range(len(caseIdsByType)):
    caseIdsByType[i]=sorted(list(caseIdsByType[i]))
```

#### 变量说明：`caseAllScores`、`casesScoreVar`

`caseAllScores`是一个列表，长度为882，记录了每道题目的所有得分，其元素也是列表，比如，`caseAllScores[x]`是编号为`x`的题目的所有学生的分数。

`casesScoreVar`是一个列表，长度为882，记录了每道题目的得分的方差，利用`caseAllScores`可以求出。

```python
caseAllScores=[[] for i in range(882)]
for i in validUserIds:
    for j in range(len(new_data[i])):
        caseAllScores[new_data[i][j]['case_id']].append(new_data[i][j]['final_score'])
casesScoreVar=[np.var(np.array(caseAllScores[i])) for i in range(882)]
```

#### 函数说明：`getGroupIdByUserId`

由学生的编号获取所在的组号

```python
def getGroupIdByUserId(userId):
    for i in range(len(groupUserIds)):
        if userId in groupUserIds[i]:
            return i
    return -1
```

#### 函数说明：`getGroupIdsByCaseId`

由题目的编号获取所在的组号，可以不止一个

```python
def getGroupIdsByCaseId(caseId):
    gs=[]
    for i in range(len(groupCaseIds)):
        if caseId in groupCaseIds[i]:
            gs.append(i)
    return gs
```

#### 函数说明：`getTypesByCaseId`

由题目的编号获取类型编号，可能不止一个

```python
def getTypesByCaseId(caseId):
    ts=[]
    for i in range(len(caseIdsByType)):
        if caseId in caseIdsByType[i]:
            ts.append(i)
    return ts
```

#### 函数说明：`getScoreByUserIdAndCaseId`

获取学生在某道题上的得分，如果该学生不需要做这道题（即不是同一组的），则返回-1。

```python
def getScoreByUserIdAndCaseId(userId,caseId):
    """
    返回某个用户在某道题上的得分,如果用户应该做这道题而没有做,则返回0分,如果这道题这个用户本来就不需要做(即用户所在的group内不包含这道题),就返回-1
    :param userId:
    :param caseId:
    :return:
    """
    if not caseId in groupCaseIds[getGroupIdByUserId(userId)]:
        return -1
    for i in new_data[userId]:
        if i['case_id']==caseId:
            return i['final_score']
    return 0
```

#### 函数说明：`getCaseIdsByGroupAndType`

根据组号和类型获取题目的编号，即获取某一组的某种类型的所有题目编号。

```python
def getCaseIdsByGroupAndType(groupId,typeId):
    return sorted(list(set(caseIdsByType[typeId])&set(groupCaseIds[groupId])))
```

#### 函数说明：`getUploadNumByUserAndCase`

获取某个学生在某道题上的提交次数，如果这个学生不需要做这道题，则返回-1

```python
def getUploadNumByUserAndCase(userId,caseId):
    if not caseId in groupCaseIds[getGroupIdByUserId(userId)]:
        return -1
    for i in new_data[userId]:
        if i['case_id']==caseId:
            return len(i['upload_records'])
    return 0
```

#### 函数说明：`getFinalUploadCodeUrlByUserAndCase`

获取某个学生在某道题上的最后一次提交记录的代码链接地址，如果没有提交，则返回空串。

```python
def getFinalUploadCodeUrlByUserAndCase(userId,caseId):
    if not caseId in groupCaseIds[getGroupIdByUserId(userId)]:
        return ''
    for i in new_data[userId]:
        if i['case_id']==caseId:
            return i['upload_records'][-1]['code_url']
    return ''
```

#### 函数说明：`getUploadSumByCaseId`

获取某道题的总的提交次数

```python
def getUploadSumByCaseId(caseId):
    r=0
    for i in validUserIds:
        r+=(getUploadNumByUserAndCase(i,caseId) if getUploadNumByUserAndCase(i,caseId)>0 else 0)
    return r
```

#### 函数说明：`getTimeSpanByUserAndCase`、`getTimeSpanByUserAndCaseInMinute`

获取某个学生做某道题目的时间跨度，即最后一次提交时间减去第一次提交时间，若该学生不需要做这道题，则返回-2，如果该学生需要做而没有做这道题，则返回-1。

```python
def getTimeSpanByUserAndCase(userId,caseId):
    if not caseId in groupCaseIds[getGroupIdByUserId(userId)]:
        return -2
    for i in new_data[userId]:
        if i['case_id']==caseId:
            return i['upload_records'][-1]['upload_time']-i['upload_records'][0]['upload_time']
    return -1
```

`getTimeSpanByUserAndCaseInMinute`：以分钟为单位返回时间跨度

```python
def getTimeSpanByUserAndCaseInMinute(userId,caseId):
    if not caseId in groupCaseIds[getGroupIdByUserId(userId)]:
        return -2
    for i in new_data[userId]:
        if i['case_id']==caseId:
            return time_diff_minute(i['upload_records'][0]['upload_time'],i['upload_records'][-1]['upload_time'])
    return -1
```

#### 函数说明：`getAvgTimeSpanByCase`、`getAvgTimeSpanByCaseInMinute`

获取某道题的平均时间跨度，需要利用`getTimeSpanByUserAndCase`函数

```python
def getAvgTimeSpanByCase(caseId):
    r=0
    n=0
    for i in validUserIds:
        if getTimeSpanByUserAndCase(i,caseId)>-1:
            r+=getTimeSpanByUserAndCase(i,caseId)
            n+=1
    return r/n
```

`getAvgTimeSpanByCaseInMinute`：以分钟为单位返回平均时间跨度

```python
def getAvgTimeSpanByCaseInMinute(caseId):
    r=0
    n=0
    for i in validUserIds:
        if getTimeSpanByUserAndCaseInMinute(i,caseId)>-1:
            r+=getTimeSpanByUserAndCaseInMinute(i,caseId)
            n+=1
    return r/n
```

#### 函数说明：`getCodeRunTime`

下载`code_url`对应的代码，解压缩，并且运行python脚本，计算运行时间并返回，传入的`userId`和`caseId`仅用于给下载的相应的文件命名，防止文件名冲突。

```python
def getCodeRunTime(code_url,userId,caseId):
    try:
        os.chdir('allcases')

        dirname=str(userId)+'_'+str(caseId)+'_dir'#存放原压缩包解压物的，每道题都有专属的文件夹名
        name=str(userId)+'_'+str(caseId)+'_zip'#原压缩包名

        urllib.request.urlretrieve(code_url,name)#下载原压缩包
        url_file=zipfile.ZipFile(name)#为原压缩包解压做准备

        os.mkdir(dirname)#原压缩包解压目录
        os.chdir(dirname)
        url_file.extractall()#原压缩包解压

        tmp=os.listdir(os.curdir)#当前目录为原压缩包解压目录，即获取原压缩包解压出来的压缩包名
        temp=tmp[0]#第二个压缩包名
        tempp=zipfile.ZipFile(temp)
        tempp.extractall()
        #第二个压缩包在此解压

        tmp=os.listdir(os.curdir)#再次获取当前目录内的所有文件名，以获得py文件进行运行
        code_name=''
        for i in tmp:
            if i[-3::]=='.py':
                code_name=i#py文件名

        start_time=time.clock()
        win32api.ShellExecute(0,'open',code_name,'','',0)
        end_time=time.clock()

        os.chdir('..')#返回至allcases目录
        os.chdir('..')#返回主目录

        return end_time-start_time
    except:
        os.chdir('..')#返回至allcases目录
        os.chdir('..')#返回主目录
        return -1
```

#### 函数说明：`getUserIdsByCaseId`

获取做了某道题目

```python
def getUserIdsByCaseId(caseId):
    r=[]
    for i in validUserIds:
        if caseId in userFinishCaseIds[i]:
            r.append(i)
    return r
```

#### 类别数据处理

题目的类型和组号现在分别以0~8和0~5的整数表示，整数和整数之间有天然的大小关系，但是对于这种类别数据，不应该有这种大小关系，所以利用`sklearn`对题目类型和组号进行独热编码，类型数据转换为长度为8的数组，如果这道题属于某一类，则相应的值为1，其余为0，组号数据同理。

```python
typeOneHot=OneHotEncoder(categories='auto').fit([[i] for i in range(8)]).transform([[i] for i in range(8)]).toarray()
groupOneHot=OneHotEncoder(categories='auto').fit([[i] for i in range(5)]).transform([[i] for i in range(5)]).toarray()
```

#### 数据初步处理完成，把`test_data.json`中有关题目的数据写入一个csv文件

利用上面定义的变量和函数，对与题目相关的数据进行整合，并保存为`cases_analysis_result.csv`。

![image.png](https://i.loli.net/2020/07/21/XK5xnmkFLr3U8sy.png)

下面对该文件做出一些解释：该文件共有23列，大小为(882,23)，每列的含义如下：

- `id`：题目的编号
- `type0~type7`：如果这道题目的类型属于这一类，相应的列的值为1，其余为0
- `finishRate`：题目的完成率
- `userNum`：应该做这道题的学生人数
- `userNumInFact`：实际上做了这道题的学生人数
- `scoreIgnoreUndo`：题目的平均得分，忽略未做的人
- `scoreCountUndo`：题目的平均分，不忽略未做的人
- `group0~group4`：如果这道题出现在某一组，则相应的列为1，其余为0
- `uploadSum`：题目的提交总次数
- `uploadAvg`：平均提交次数，不忽略未做的人
- `uploadAvgInFact`：平均提交次数，忽略未做的人
- `timeSpan`：做这道题目的学生所花的平均时间跨度

```python
for i in range(882):
    typeId=np.zeros((8,))
    for j in getTypesByCaseId(i):
        typeId+=np.array(typeOneHot[j])
    finishRate=caseFinishRate[i]
    userNum=caseUserNum[i]
    userNumInFact=caseUserNumInFact[i]
    scoreIgnoreUndo=caseScoreIgnoreUndo[i]
    scoreCountUndo=caseScoreCountUndo[i]
    groupId=np.zeros((5,))
    for j in getGroupIdsByCaseId(i):
        groupId+=np.array(groupOneHot[j])
    uploadSum=getUploadSumByCaseId(i)
    uploadAvg=uploadSum/userNum
    uploadAvgInFact=uploadSum/userNumInFact
    timeSpan=getAvgTimeSpanByCaseInMinute(i)
    caseLine=np.concatenate((np.array([i]),typeId,np.array([finishRate]),np.array([userNum]),np.array([userNumInFact]),np.array([scoreIgnoreUndo]),np.array([scoreCountUndo]),groupId,np.array([uploadSum]),np.array([uploadAvg]),np.array([uploadAvgInFact]),np.array([timeSpan])),axis=0).reshape(1,-1)
    if i==0:
        cases_analysis_result=caseLine
    else:
        cases_analysis_result=np.concatenate((cases_analysis_result,caseLine),axis=0)
with open('cases_analysis_result.csv',mode='w',newline='') as file:
    cw=csv.writer(file)
    header=['id','type0','type1','type2','type3','type4','type5','type6','type7','finishRate','userNum','userNumInFact','scoreIgnoreUndo','scoreCountUndo','group0','group1','group2','group3','group4','uploadSum','uploadAvg','uploadAvgInFact','timeSpan']
    cw.writerow(header)
    for i in cases_analysis_result:
        cw.writerow(list(i))
```

#### 多列数据归一化

读取之前保存的`cases_analysis_result.csv`，利用`sklearn`对`timeSpan`、`uploadSum`、`scoreIgnoreUndo`、`scoreCountUndo`、`userNum`、`userNumInFact`进行归一化处理，即放缩数据使得均值为1、方差为0；此外，再添加一列`scoreVar`，表示每道题目的方差，并且进行归一化，此列可以利用`casesScoreVar`求出，

```python
cases_analysis_result=pd.read_csv('cases_analysis_result.csv')
cases_analysis_result['timeSpan']=StandardScaler().fit_transform(np.array(cases_analysis_result['timeSpan']).reshape(-1,1))
cases_analysis_result['uploadSum']=StandardScaler().fit_transform(np.array(cases_analysis_result['uploadSum']).reshape(-1,1))
cases_analysis_result['scoreIgnoreUndo']=StandardScaler().fit_transform(np.array(cases_analysis_result['scoreIgnoreUndo']).reshape(-1,1))
cases_analysis_result['scoreCountUndo']=StandardScaler().fit_transform(np.array(cases_analysis_result['scoreCountUndo']).reshape(-1,1))
cases_analysis_result['userNum']=StandardScaler().fit_transform(np.array(cases_analysis_result['userNum']).reshape(-1,1))
cases_analysis_result['userNumInFact']=StandardScaler().fit_transform(np.array(cases_analysis_result['userNumInFact']).reshape(-1,1))
cases_analysis_result['scoreVar']=StandardScaler().fit_transform(np.array(casesScoreVar).reshape(-1,1))
```

#### 下载代码、运行、获取运行时间

利用`getCodeRunTime`函数计算每道题的运行时间，具体实施为：对于每个用户做的每道题目，去最后一次提交记录提交的代码，下载、运行、记录运行时间。并把结果保存到`code_run_time.csv`，其大小为(271,882)。若时间缺少，则记为-1。

```python
codeRunTime=-np.ones((271,882))
for i in validUserIds:
    for j in range(882):
        if getFinalUploadCodeUrlByUserAndCase(i,j)!='':
            codeRunTime[i,j]=getCodeRunTime(getFinalUploadCodeUrlByUserAndCase(i,j),i,j)
            os.chdir(root_path)
with open('code_run_time.csv',mode='w',newline='') as file:
    cw=csv.writer(file)
    for i in codeRunTime:
        cw.writerow(list(i))
```

#### 计算每道题的平均运行时间

读取之前保存的`code_run_time.csv`，计算每道题目的平均运行时间，并为`cases_analysis_result`增加一列`avgRunTime`记录代码平均运行时间，此列在一定程度上可以反应平均代码质量。

```python
codeRunTime=pd.read_csv('code_run_time.csv',header=None).values
def getCaseAvgRunTime(caseId):
    sumTime=0
    for i in validUserIds:
        if codeRunTime[i,caseId]>-1:
            sumTime+=codeRunTime[i,caseId]
    return sumTime/caseUserNumInFact[caseId]
cases_analysis_result['avgRunTime']=StandardScaler().fit_transform(np.array([getCaseAvgRunTime(i) for i in range(882)]).reshape(-1,1))
```

#### 题目难度评估

定义难度系数函数，难度系数由`finishRate`、`scoreIgnoreUndo`、`scoreCountUndo`、`uploadAvg`、`uploadAvgInFact`、`timeSpan`、`avgRunTime`这些列决定，由于cases_analysis_result的各列数据大部分都已经利用`sklearn`归一化过了，所以在计算难度系数时可以把各列的权重设为一样，难度系数越大，表明题目越难。从另一个角度来说，难度系数越高的题目，说明学生对这道题的掌握情况也就越差。

```python
def difficult_degree(caseId):
    """
    题目的难度系数,值越大说明题目越难,各列的系数可能还需要调整
    :param caseId:
    :return:
    """
    return -cases_analysis_result.iloc[caseId]['finishRate']-cases_analysis_result.iloc[caseId]['scoreIgnoreUndo']-cases_analysis_result.iloc[caseId]['scoreCountUndo']+cases_analysis_result.iloc[caseId]['uploadAvg']+cases_analysis_result.iloc[caseId]['uploadAvgInFact']+cases_analysis_result.iloc[caseId]['timeSpan']+cases_analysis_result.iloc[caseId]['avgRunTime']
```

在评估完每道题目的难度后，我们可以对每种类型的题目进行整体难度评估

```python
def getTypeDifficultDegree(typeId):
    d=0
    for caseId in caseIdsByType[typeId]:
        d+=difficult_degree(caseId)
    return d/len(caseIdsByType[typeId])
```

对每组的题目进行难度评估

```python
def getGroupDifficultDegree(groupId):
    d=0
    for caseId in groupCaseIds[groupId]:
        d+=difficult_degree(caseId)
    return d/len(groupCaseIds[groupId])
```

#### 保存处理后的题目相关的数据

保存`cases_analysis_result`到`cases_analysis_final.csv`

```python
cases_analysis_result['difficultDegree']=np.array([difficult_degree(i) for i in range(882)]).reshape(-1,1)
cases_analysis_result.to_csv('cases_analysis_final.csv')
```

### 题目视角

#### 题目难度分析

#### 基于学生答题表现的聚类分析

#### "好题"、"坏题"鉴定器

#### 面向测试用例编程检测器

### 学生视角

#### 编程能力评估

#### 生成编程学习路线

#### 寻找编程搭档

#### 自动推荐代码

## 案例分析

