# 数据科学基础 大作业 研究报告

[toc]

## 小组信息

### 小组成员

| 学号      | 姓名   | 邮箱                       | Python练习完成题目数量 |
| --------- | ------ | -------------------------- | ---------------------- |
| 181250194 | 赵一鸣 | 181250194@smail.nju.edu.cn | 200                    |
| 181870006 | 蔡晓星 | 181870006@smail.nju.edu.cn | 200                    |
| 181098100 | 胡家铭 | 181098100@smail.nju.edu.cn | 190                    |

### 组员分工职责



## 项目总述

这次的数据科学项目中，我们分别站在题目的视角和学生的视角各讨论了4个、5个主题，研究的主题数量是相对来说是比较多的，但是事实上，花的时间最多的模块是数据预处理部分，即数据清理和格式化 & 探索性数据分析 & 特征工程和特征选择部分，在该部分我们对老师提供的`test_data.json`数据进行了清洗和整理，对分类数据、编号类数据进行了重新编号，使其从0开始每次加一增长，同时整理了很多列表、数组、字典等变量，这些变量在运行程序后会被保存到内存中而非磁盘中，这些列表等变量可以使我们非常方便地获取某些信息，比如说`groupCaseIds`就记录了每组的题目的编号，没有这些变量，我们依然可以从最原始的`test_data.json`获取我们需要的信息，但是这会降低编程效率，与此同时还写了一些函数，这些函数的作用和先前的变量类似，能够快速地获取关于数据集的一些信息，这些在内存中函数和变量的作用就相当于索引、视图、存储过程对于数据库的作用；此外，我们还把从`test_data.json`中提取出的有用的信息保存为一些列的csv文件，csv文件更加容易转换为`numpy`数组和`DataFrame`，为后续的建模分析打下基础。尽管数据预处理部分花费了大量的时间，但这期间并没有用到什么复杂的算法或是数学定理。然而事实上，由于在前期的数据预处理部分做了大量的工作，使得在后续的对9个主题的探究上尤为方便轻松，因为想要的数据大都已经准备好了，我们可以更加专注于算法的实施和数学知识的运用。这个大作业让我更加深刻地感受到数据清洗和特征工程的重要性。

## 关键词

`scikit-learn` `PyTorch` `NumPy` `Pandas` `matplotlib` `数据清洗` `特征工程` `数据可视化` `主成分分析` `深度自编码器` `变分自编码器` `K-means聚类分析` 

## 研究问题

### 题目视角

详细分析在"研究方法"模块

1. 题目难度分析

综合题目的各项指标，如平均分数、完成率、代码平均运行时间等指标，对每道题目给出一个难度系数的评估分数，分数越高说明题目越难。这一分析有助于老师合理控制布置的题目的难度，使其在一个合理的区间范围内。

2. 基于学生答题表现的聚类分析

综合题目的各项指标，运用主成分分析、深度自编码器、变分自编码器对数据进行降维，接着再对题目进行聚类分析，聚类后在同一类中的题目，学生在这些题目上的表现更加接近。这一分析能够帮助老师分析学生在哪些题目上的表现相似，从而制定更加合理的教学计划。

3. 题目质量鉴定器

分析一道题目出的好不好，比如说，如果一道题目的得分的方差太小或做题人数过少，则说明这道题出的不好，实际操作时，考虑到的指标不止上述两个。这一研究能够帮助老师挑选好的编程题，从而更加有利于学生编程能力的提升。

4. 面向测试用例编程检测器

此次编程作业中，有很多人会套测试用例，从而"面向测试用例编程"，这种行为不利于学生真正掌握编程，所以写了一个面向测试用例编程检测器，从而能够筛选出这些提交记录并记为0分。

5. 代码抄袭检测器

为了更好地评估学生的做题情况、代码的原创性以及公平性考虑，我们写了一个基于`pycode_similar`和`difflib`的代码抄袭检测器，即判断两个`python`文件的代码的重复率，从而给出是否有抄袭嫌疑的判断。

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
import matplotlib.pyplot as plt # 画图工具
import random
import difflib # 用于检测代码是否抄袭的工具
import seaborn as sns # 画图工具
```

![download.jpg](https://i.loli.net/2020/07/21/1B2WrXFQ7feO6vd.jpg)

![download.png](https://i.loli.net/2020/07/21/pzAhuLsawOoTDvj.png)

`NumPy`可以用来存储和处理大型矩阵，支持大量的维度数组和矩阵运算，`pandas`是为了解决数据分析任务而创建的，能够高效地操作大型数据集。在后续的分析建模中，我们会把数据集转换成csv格式的文件，`Numpy`和`pandas`可以高效地处理这些数据。

![download.png](https://i.loli.net/2020/07/22/swUcyFXzVROBu3n.png)

`matplotlib`是一个数据科学的画图工具，可以帮助我们更好地展示分析结果、判断分析方法是否正确等。

![download.png](https://i.loli.net/2020/07/23/BgjG6LrEfDyxhbe.png)

`seaborn`是一个基于`matplotlib`的数据可视化工具，可以画出丰富的统计图形。

![download.png](https://i.loli.net/2020/07/21/GNuL73Y8DT2ms5n.png)

`scikit-learn`库提供了非常方便的数据预处理函数和基本的机器学习函数，比如独热编码、数据归一化、主成分分析等模块。

![download.png](https://i.loli.net/2020/07/21/MwbY5P4jzqZUQ2x.png)

`PyTorch`模块提供了构建神经网络、自动求导等工具，本项目中会使用深度自编码器和变分自编码器，所以需要引入这个库，此外，`PyTorch`还提供了便捷地构建数据集的工具。

### 数据清理和格式化 & 探索性数据分析 & 特征工程和特征选择

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

#### 函数说明：`getAvgTimeSpanByCase`、`getAvgTimeSpanByCaseInMinute`、`getGroupAvgTimeSpanInMinute`、`getTypeAvgTimeSpanInMinute`

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

`getGroupAvgTimeSpanInMinute`、`getTypeAvgTimeSpanInMinute`：按组或按题目类型返回平均做题时间跨度

```python
def getGroupAvgTimeSpanInMinute(groupId):
    s=0
    for caseId in groupCaseIds[groupId]:
        s+=getAvgTimeSpanByCaseInMinute(caseId)
    return s/len(groupCaseIds[groupId])
def getTypeAvgTimeSpanInMinute(typeId):
    s=0
    for caseId in caseIdsByType[typeId]:
        s+=getAvgTimeSpanByCaseInMinute(caseId)
    return s/len(caseIdsByType[typeId])
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

部分下载的代码如下。

![image.png](https://i.loli.net/2020/07/22/5jcPa9iz3lgWIK8.png)

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

#### 函数说明：`getTypeAvgScoreIgnoreUndo`、`getGroupAvgScoreIgnoreUndo`、`getTypeAvgScoreCountUndo`、`getGroupAvgScoreCountUndo`

按小组或题目类型获取平均分（忽略没做的人或不忽略没做的人）

```python
def getTypeAvgScoreIgnoreUndo(typeId):
    temp=pd.read_csv('cases_analysis_result.csv')
    s=0
    for caseId in caseIdsByType[typeId]:
        s+=temp.iloc[caseId]['scoreIgnoreUndo']
    return s/len(caseIdsByType[typeId])
def getGroupAvgScoreIgnoreUndo(groupId):
    temp=pd.read_csv('cases_analysis_result.csv')
    s=0
    for caseId in groupCaseIds[groupId]:
        s+=temp.iloc[caseId]['scoreIgnoreUndo']
    return s/len(groupCaseIds[groupId])
def getTypeAvgScoreCountUndo(typeId):
    temp=pd.read_csv('cases_analysis_result.csv')
    s=0
    for caseId in caseIdsByType[typeId]:
        s+=temp.iloc[caseId]['scoreCountUndo']
    return s/len(caseIdsByType[typeId])
def getGroupAvgScoreCountUndo(groupId):
    temp=pd.read_csv('cases_analysis_result.csv')
    s=0
    for caseId in groupCaseIds[groupId]:
        s+=temp.iloc[caseId]['scoreCountUndo']
    return s/len(groupCaseIds[groupId])
```

#### 函数说明：`getGroupAvgFinishRate`、`getTypeAvgFinishRate`

按小组或题目类型获取平均完成率

```python
def getGroupAvgFinishRate(groupId):
    s=0
    for caseId in groupCaseIds[groupId]:
        s+=caseFinishRate[caseId]
    return s/len(groupCaseIds[groupId])
def getTypeAvgFinishRate(typeId):
    s=0
    for caseId in caseIdsByType[typeId]:
        s+=caseFinishRate[caseId]
    return s/len(caseIdsByType[typeId])
```

#### 函数说明：`getGroupAvgUploadNum`、`getTypeAvgUploadNum`、`getGroupAvgUploadNumInFact`、`getTypeAvgUploadNumInFact`

按小组或题目类型获取平均提交次数（忽略或不忽略没有提交的人）

```python
def getGroupAvgUploadNum(groupId):
    temp=pd.read_csv('cases_analysis_result.csv')
    s=0
    for caseId in groupCaseIds[groupId]:
        s+=temp.iloc[caseId]['uploadAvg']
    return s/len(groupCaseIds[groupId])
def getTypeAvgUploadNum(typeId):
    temp=pd.read_csv('cases_analysis_result.csv')
    s=0
    for caseId in caseIdsByType[typeId]:
        s+=temp.iloc[caseId]['uploadAvg']
    return s/len(caseIdsByType[typeId])
def getGroupAvgUploadNumInFact(groupId):
    temp=pd.read_csv('cases_analysis_result.csv')
    s=0
    for caseId in groupCaseIds[groupId]:
        s+=temp.iloc[caseId]['uploadAvgInFact']
    return s/len(groupCaseIds[groupId])
def getTypeAvgUploadNumInFact(typeId):
    temp=pd.read_csv('cases_analysis_result.csv')
    s=0
    for caseId in caseIdsByType[typeId]:
        s+=temp.iloc[caseId]['uploadAvgInFact']
    return s/len(caseIdsByType[typeId])
```

#### 类别数据处理

题目的类型和组号现在分别以0到8和0到5的整数表示，整数和整数之间有天然的大小关系，但是对于这种类别数据，不应该有这种大小关系，所以利用`sklearn`对题目类型和组号进行独热编码，类型数据转换为长度为8的数组，如果这道题属于某一类，则相应的值为1，其余为0，组号数据同理。

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

对每组的每种类型的题目进行难度评估

```python
def getDifficultDegreeByGroupAndType(groupId,typeId):
    d=0
    n=0
    for caseId in groupCaseIds[groupId]:
        if caseId in caseIdsByType[typeId]:
            n+=1
            d+=difficult_degree(caseId)
    return d/n
```

#### 保存处理后的题目相关的数据

保存`cases_analysis_result`到`cases_analysis_final.csv`

```python
cases_analysis_result['difficultDegree']=np.array([difficult_degree(i) for i in range(882)]).reshape(-1,1)
cases_analysis_result.to_csv('cases_analysis_final.csv')
```

![image.png](https://i.loli.net/2020/07/22/4mt6wI5r7qPKMNs.png)

与题目相关的数据基本处理完毕，接下来可以利用这些数据做一些站在题目视角的特征分析。

#### 降维：PCA

既然已经把每道题目的信息都提取出来了，即每道题目的信息可以用一个26维的向量表示，那么为什么不直接分析这些数据，而要降维？原因有以下几点：

- 每道题目用26个特征表示，但是这些特征并不是两两正交的，某些特征之间存在一定的线性关系，比如`finishRate`、`userNum`、`userNumInFact`等特征之间存在一定的关系，所以就需要抽取出一组能够综合代表这些特征的新特征，而且这些新特征在一定程度上是相互正交的，且能够反映出原数据的大部分特征，这就要求降维之后再重构得到的26维特征与原数据的差距不能太大。
- 随着维度的升高，同样数量的数据集在空间中的分布会越来越稀疏，比如说，如果在原有数据的基础上增加一个维度，即使这个维度只是一个仅有两种取值的类别特征，如果要覆盖住这个维度，我们需要的数据量是原来的两倍，也就是说，随着维度的升高，我们需要的数据集数量呈指数倍增长，否则可能会造成过拟合。
- 后续我们需要对题目数据进行聚类分析，降低维度、抽取主成分会加快聚类的速度、提高聚类的效果，如果把数据降维到三维及以下，就可以把数据集的分布画出来，结合后续聚类，能直观的看出聚类效果，从而更方便地调整算法

首先对`cases_analysis_result.csv`的数据进行主成分分析，分别降维到2维和4维，下图为PCA的基本原理图。

![image.png](https://i.loli.net/2020/07/22/EhLtup4y86ZFRKm.png)

降维到2维后再重构，均方误差为`0.2976`，把降维之后的结果保存到`cases_pca_2_dim_result.csv`。

```python
new_dim=2
# new_dim=4
# cases_data_pca=PCA(n_components=new_dim).fit_transform(cases_result_array)
model_pca=PCA(n_components=new_dim).fit(cases_result_array)
cases_data_pca=model_pca.transform(cases_result_array)
cases_reconstructed=model_pca.inverse_transform(cases_data_pca)
with open('cases_pca_2_dim_result.csv',mode='w',newline='') as file:
    cw=csv.writer(file)
    header=['id','dim0','dim1']
    cw.writerow(header)
    for i in range(882):
        cw.writerow([i,cases_data_pca[i,0],cases_data_pca[i,1]])
pca_loss=np.mean(np.square(cases_result_array-cases_reconstructed))
print('PCA 2 dim mean square loss:{}'.format(pca_loss))
```

```
PCA 2 dim mean square loss:0.2976193041640051
```

![image.png](https://i.loli.net/2020/07/22/YWaRz7smeHbQtP1.png)

降维到4维后再重构，均方误差为`0.1387`，把降维之后的结果保存到`cases_pca_4_dim_result.csv`。

```python
# new_dim=2
new_dim=4
# cases_data_pca=PCA(n_components=new_dim).fit_transform(cases_result_array)
model_pca=PCA(n_components=new_dim).fit(cases_result_array)
cases_data_pca=model_pca.transform(cases_result_array)
cases_reconstructed=model_pca.inverse_transform(cases_data_pca)
with open('cases_pca_4_dim_result.csv',mode='w',newline='') as file:
    cw=csv.writer(file)
    header=['id','dim0','dim1','dim2','dim3']
    cw.writerow(header)
    for i in range(882):
        cw.writerow([i,cases_data_pca[i,0],cases_data_pca[i,1],cases_data_pca[i,2],cases_data_pca[i,3]])
pca_loss=np.mean(np.square(cases_result_array-cases_reconstructed))
print('PCA 4 dim mean square loss:{}'.format(pca_loss))
```

```
PCA 4 dim mean square loss:0.13871950667820518
```

![image-20200722142635213.png](https://i.loli.net/2020/07/22/7fLxmkINzbGD1dM.png)

从均方误差上来看，主成分分析的效果还不错。

#### 降维：AE&VAE

AE，即自编码器，其原理比较简单，是在主成分分析的基础上加以改造得到的方法，在PCA中，第一层的输入层和最后一层的重构后的层之间只有一层，现在增大中间的层数，使得这个结构变得更深。PCA其实可以看做两层网络结构的自编码器。

![image.png](https://i.loli.net/2020/07/22/SclYO572x1VKa6P.png)

![image.png](https://i.loli.net/2020/07/22/7o2zD5d1OqyIxXr.png)

构建深度自编码器需要使用python的`torch`库，所以首先需要把从`cases_analysis_result.csv`读取到的`numpy`数组格式的数据转化为`PyTorch`中的`Tensor`格式，这样更加方便后续训练自编码器。

```python
batch_size=32
cases_tensor=torch.tensor(cases_result_array,dtype=torch.float)
cases_dataset=TensorDataset(cases_tensor)
cases_sampler=RandomSampler(cases_dataset)
cases_dataloader=DataLoader(cases_dataset,sampler=cases_sampler,batch_size=batch_size)
```

定义自编码器模型，把原数据由26维降到2维。

```python
dim=cases_result_array.shape[1]
class AE(nn.Module):
    def __init__(self):
        super(AE,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(dim,20),
            nn.ReLU(True),
            nn.Linear(20,16),
            nn.ReLU(True),
            nn.Linear(16,8),
            nn.ReLU(True),
            nn.Linear(8,4),
            nn.ReLU(True),
            nn.Linear(4,2)
        )
        self.decoder=nn.Sequential(
            nn.Linear(2,4),
            nn.ReLU(True),
            nn.Linear(4,8),
            nn.ReLU(True),
            nn.Linear(8,16),
            nn.ReLU(True),
            nn.Linear(16,20),
            nn.ReLU(True),
            nn.Linear(20,dim)
        )
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x
```

定义损失函数`criterion`、优化器`optimizer`、学习率`learning_rate`、迭代次数`num_epochs`，并开始训练模型，为了防止过拟合，在训练中会保存损失最小的模型到`best_model_AE_2_dim.pt`，训练结束后读取保存的模型，用它对原数据进行降维，把降维后得到的数据保存到`cases_ae_encode_2_dim.csv`。

```python
model_classes={'AE':AE(),'VAE':VAE()}
learning_rate=1e-3
num_epochs=2000
model_type='AE'
# model_type='VAE'
model=model_classes[model_type].cuda()
criterion=nn.MSELoss()
optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
best_loss=np.inf
model.train()
for epoch in range(num_epochs):
    for data in cases_dataloader:
        batch_cases=data[0].cuda()
        output=model(batch_cases)
        if model_type=='VAE':
            loss=loss_vae(output[0],batch_cases,output[1],output[2],criterion)
        else:
            loss=criterion(output,batch_cases)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item()<best_loss:
            best_loss=loss.item()
            torch.save(model,'best_model_{}_2_dim.pt'.format(model_type))
    # if (epoch+1)%50==0:
    # print('epoch [{}/{}],loss:{:.4f}'.format(epoch+1,num_epochs,loss.item()))
    if (epoch+1)%100==0 or epoch==0:
        print('epoch ['+str(epoch+1)+'/'+str(num_epochs)+'],loss:'+str(loss.item()))
print('Best '+model_type+' model\' loss:'+str(best_loss))
final_model=torch.load('best_model_AE_2_dim.pt')
cases_ae_encoder=final_model.encoder(cases_tensor.cuda()).detach().cpu().numpy()
with open('cases_ae_encode_2_dim.csv',mode='w',newline='') as file:
    cw=csv.writer(file)
    header=['id','dim0','dim1']
    cw.writerow(header)
    for i in range(882):
        cw.writerow([i,cases_ae_encoder[i,0],cases_ae_encoder[i,1]])
```

![image-20200722142732912.png](https://i.loli.net/2020/07/22/Oq3I24JDahmb8C6.png)

降维到2维的深度自编码器的最好的模型的均方误差为`0.1361`。

![image.png](https://i.loli.net/2020/07/22/ELV7epADuxZjJFH.png)

除了PCA和深度自编码器，我们采用的第三种降维方式是变分自编码器（VAE），VAE是一种生成模型，它主要对数据的结构进行建模，捕捉数据不同维度之间的关系。变分自编码器能够从高维变量中学习到低维的潜变量，数据集中的每个数据`x`都有一个相对应的潜变量`z`，在此我们利用VAE寻找潜变量的作用来实现对数据的降维。

![download.png](https://i.loli.net/2020/07/22/j2p6KTsHuyctqJB.png)

定义VAE模型，把原数据由26维降至2维，该模型的损失函数维均方误差加上KL散度。

```python
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1=nn.Linear(dim,16)
        self.fc21=nn.Linear(16,2)
        self.fc22=nn.Linear(16,2)
        self.fc3=nn.Linear(2,16)
        self.fc4=nn.Linear(16,dim)
    def encoder(self,x):
        h1=F.relu(self.fc1(x))
        return self.fc21(h1),self.fc22(h1)
    def reparametrize(self,mu,logvar):
        std=logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps=torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps=torch.FloatTensor(std.size()).normal_()
        eps=Variable(eps)
        return eps.mul(std).add_(mu)
    def decode(self,z):
        h3=F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))
    def forward(self,x):
        mu,logvar=self.encoder(x)
        z=self.reparametrize(mu,logvar)
        return self.decode(z),mu,logvar
def loss_vae(recon_x,x,mu,logvar,criterion):
    """
    recon_x: generating values
    x: origin values
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD
```

下面开始训练VAE模型，训练部分的代码与深度自编码器部分类似，此处不再展示，所有代码详见`workspace.ipynb`，以下为VAE在训练过程中的损失函数的变化情况。保存损失函数最小的epoch，保存模型到`base_model_VAE_2_dim.pt`，训练完成之后，读取最佳模型，并用它把原来26维的数据降维到2维，然后把结果保存到`cases_vae_encode_2_dim.csv`。

![image-20200722142401855.png](https://i.loli.net/2020/07/22/INE6yklrCqvBX8R.png)

![image.png](https://i.loli.net/2020/07/22/uZKGdq53IQVUXzg.png)

上面用深度自编码器和变分自编码器把数据从26维降到了2维，因为2维的数据点更容易可视化，为了使得在后续对题目数据进行更好地聚类分析，2维的维度未免过低，所以这里就和PCA一样，也用AE和VAE把数据降维到4维，这需要微调两个模型后再进行训练，具体的代码不在这里展示，所有代码详见`workspace.ipynb`，训练得到的两个模型保存为了`best_model_AE_4_dim.pt`、`best_model_VAE_4_dim.pt`，用最佳的模型降维的结果分别保存为了`cases_ae_encode_4_dim.csv`、`cases_vae_encode_4_dim.csv`。下面四幅图分别为两个模型的损失函数值、降维后的结果，损失函数值要低于降到2维的模型。

![image.png](https://i.loli.net/2020/07/22/guiEZBwNMQVJa5L.png)

![image.png](https://i.loli.net/2020/07/22/buYO9UQMEXrSfsy.png)

![image.png](https://i.loli.net/2020/07/22/OLnHSTzB65iRruX.png)

![image.png](https://i.loli.net/2020/07/22/OcQsw78vyES2uFP.png)

### 题目视角

下面开始进行从题目视角出发的特定主题的分析，在整个数据分析的过程中，其实大部分的时间还是花在数据清理和格式化 & 探索性数据分析 & 特征工程和特征选择上的，而且如果第一阶段的数据整理、清洗做到位的话，后续的进一步的研究分析将会变得更加快捷方便。

#### 题目难度分析

在前面`数据清理和格式化 & 探索性数据分析 & 特征工程和特征选择`模块中，已经定义了一个名为`diffictlt_degree`的函数，此函数会综合考虑题目的完成率、忽略未做学生的平均得分、考虑未做学生的平均得分、平均提交次数、平均做题时间跨度、代码平均运行时间等因素综合判定这道题目的难度，并且在计算难度系数时会进行归一化，使得不同的指标对难度系数的影响程度大致相同。如果只想考虑一个因素，如，只想看看各类题目或各组题目的平均得分情况，这也是可以做到的。

```python
def difficult_degree(caseId):
    """
    题目的难度系数,值越大说明题目越难,各列的系数可能还需要调整
    :param caseId:
    :return:
    """
    return -cases_analysis_result.iloc[caseId]['finishRate']-cases_analysis_result.iloc[caseId]['scoreIgnoreUndo']-cases_analysis_result.iloc[caseId]['scoreCountUndo']+cases_analysis_result.iloc[caseId]['uploadAvg']+cases_analysis_result.iloc[caseId]['uploadAvgInFact']+cases_analysis_result.iloc[caseId]['timeSpan']+cases_analysis_result.iloc[caseId]['avgRunTime']
def getTypeDifficultDegree(typeId):
    d=0
    for caseId in caseIdsByType[typeId]:
        d+=difficult_degree(caseId)
    return d/len(caseIdsByType[typeId])
def getGroupDifficultDegree(groupId):
    d=0
    for caseId in groupCaseIds[groupId]:
        d+=difficult_degree(caseId)
    return d/len(groupCaseIds[groupId])
def getDifficultDegreeByGroupAndType(groupId,typeId):
    d=0
    n=0
    for caseId in groupCaseIds[groupId]:
        if caseId in caseIdsByType[typeId]:
            n+=1
            d+=difficult_degree(caseId)
    return d/n
```

下图是对所有题目按题目类型分类的各项指标的统计雷达图，其中平均得分为该题的总得分除以应该做这道题的人数，如果有学生应该做而未做这道题，则会以0分计入；平均得分（ignore）是实际做了这道题的学生的平均得分，即会忽略没有做的人，比如，这道题只有一个人做了且得分为100，则这道题的平均得分为100；难度为上述`difficult_degree`函数计算出的难度系数，从下图中可以看出八种类型的题目在难度系数上的区分度是最明显的，其中最难的题目类型为图结构，这个结果也很符合我们的预期（图结构的题目真的难到爆，根本完全没有任何学会的可能），由此看来，我们之前定义的难度系数的计算函数还是比较合理的；处理上述指标之外，还画出了完成率、平均上传次数、平均做题时间跨度这些指标，老师可以根据不同的指标来评估学生对于不同类型的题目掌握情况，从而动态地调整教学计划。

![image.png](https://i.loli.net/2020/07/23/WMyAXxQS5ia31Nm.png)

除了可以根据预先定义的难度系数来评估题目的难度，当然我们也提供了直接基于平均分的难度评估，下面是八种类型的题目的平均分可视化结果。其中`Count`表示计算平均分时会把应该做而没有做这道题的学生的分数记为0分，`Ignore`表示只会计算做了这道题的学生的分数而忽略未做的，`Count`和`Ignore`之间存在着一定的差距。

![image.png](https://i.loli.net/2020/07/23/7y5W1XITjs2DRbv.png)

除了按题目类型分类整理相应的数据，我们也做了按小组分类整理题目的相关数据，其可视化结果如下所示，老师可以根据以下图表更好的掌握各组同学的编程能力。

![image.png](https://i.loli.net/2020/07/23/ZYJ5yks28qTrnHh.png)

具体可视化部分的代码详见`workspace.ipynb`。

#### 基于学生答题表现的聚类分析

之前已经对题目相关的数据进行了清洗整理，并得到了一个处理后的数据`cases_analysis_final.csv`，还有降维后的一些`csv`结果文件。

![image.png](https://i.loli.net/2020/07/22/9OQCwV5rIetpxWN.png)

在`cases_analysis_final.csv`中保存的关于题目的信息不是题目本身的类型或是考察的知识点，而是基于学生作答这些题目时所反映出的能力表现整理出的数据，所以说接下来做的聚类分析并不能反映出题目本身的性质，而是反映出学生的答题表现。聚类分析将在`cases_ae_encode_2_dim.csv`、`cases_ae_encode_4_dim.csv`、`cases_analysis_final.csv`、`cases_analysis_result.csv`、`cases_pca_2_dim_result.csv`、`cases_pca_4_dim_result.csv`、`cases_vae_encode_2_dim.csv`、`cases_vae_encode_4_dim.csv`的基础上进行，即原数据和采用各种手段降维得到的数据。

```python
cases_analysis_result=pd.read_csv('cases_analysis_result.csv')
cases_analysis_final=pd.read_csv('cases_analysis_final.csv').iloc[:,1:]

cases_pca_2_dim=pd.read_csv('cases_pca_2_dim_result.csv')
cases_pca_4_dim=pd.read_csv('cases_pca_4_dim_result.csv')
cases_ae_2_dim=pd.read_csv('cases_ae_encode_2_dim.csv')
cases_ae_4_dim=pd.read_csv('cases_ae_encode_4_dim.csv')
cases_vae_2_dim=pd.read_csv('cases_vae_encode_2_dim.csv')
cases_vae_4_dim=pd.read_csv('cases_vae_encode_4_dim.csv')
```

观察一下降维后的数据会发现，数据的各个维度的取值区间较大，不利于聚类取得好的效果，所以下面使用`sklearn`对降维后的数据进行归一化处理。

```python
cases_analysis_final['difficultDegree']=StandardScaler().fit_transform(np.array(cases_analysis_final['difficultDegree']).reshape(-1,1))

cases_pca_2_dim['dim0']=StandardScaler().fit_transform(np.array(cases_pca_2_dim['dim0']).reshape(-1,1))
cases_pca_2_dim['dim1']=StandardScaler().fit_transform(np.array(cases_pca_2_dim['dim1']).reshape(-1,1))

cases_ae_2_dim['dim0']=StandardScaler().fit_transform(np.array(cases_ae_2_dim['dim0']).reshape(-1,1))
cases_ae_2_dim['dim1']=StandardScaler().fit_transform(np.array(cases_ae_2_dim['dim1']).reshape(-1,1))

cases_ae_4_dim['dim0']=StandardScaler().fit_transform(np.array(cases_ae_4_dim['dim0']).reshape(-1,1))
cases_ae_4_dim['dim1']=StandardScaler().fit_transform(np.array(cases_ae_4_dim['dim1']).reshape(-1,1))
cases_ae_4_dim['dim2']=StandardScaler().fit_transform(np.array(cases_ae_4_dim['dim2']).reshape(-1,1))
cases_ae_4_dim['dim3']=StandardScaler().fit_transform(np.array(cases_ae_4_dim['dim3']).reshape(-1,1))

cases_vae_2_dim['dim0']=StandardScaler().fit_transform(np.array(cases_vae_2_dim['dim0']).reshape(-1,1))
cases_vae_2_dim['dim1']=StandardScaler().fit_transform(np.array(cases_vae_2_dim['dim1']).reshape(-1,1))

cases_vae_4_dim['dim0']=StandardScaler().fit_transform(np.array(cases_vae_4_dim['dim0']).reshape(-1,1))
cases_vae_4_dim['dim1']=StandardScaler().fit_transform(np.array(cases_vae_4_dim['dim1']).reshape(-1,1))
cases_vae_4_dim['dim2']=StandardScaler().fit_transform(np.array(cases_vae_4_dim['dim2']).reshape(-1,1))
cases_vae_4_dim['dim3']=StandardScaler().fit_transform(np.array(cases_vae_4_dim['dim3']).reshape(-1,1))
```

接着就可以对上述各组数据进行聚类分析，与此同时也会直接对原来26维的数据直接进行聚类，来作为和其他各组数据的对比，我们选择的聚类的类别数维8类，最大迭代次数维2000。

```python
km_pca_2_dim=KMeans(n_clusters=8,max_iter=2000).fit(cases_pca_2_dim.iloc[:,1:].values)
km_pca_4_dim=KMeans(n_clusters=8,max_iter=2000).fit(cases_pca_4_dim.iloc[:,1:].values)
km_ae_2_dim=KMeans(n_clusters=8,max_iter=2000).fit(cases_ae_2_dim.iloc[:,1:].values)
km_ae_4_dim=KMeans(n_clusters=8,max_iter=2000).fit(cases_ae_4_dim.iloc[:,1:].values)
km_vae_2_dim=KMeans(n_clusters=8,max_iter=2000).fit(cases_vae_2_dim.iloc[:,1:].values)
km_vae_4_dim=KMeans(n_clusters=8,max_iter=2000).fit(cases_vae_4_dim.iloc[:,1:].values)
km_raw=KMeans(n_clusters=8,max_iter=2000).fit(cases_analysis_final)
```

如何检验聚类效果呢？我们打算采用可视化的方法，把聚类后的数据分类别画在一个二维平面上来看聚类效果。部分的作图结果如下所示，完整结果及代码可在`workspace.ipynb`中查看。

![image.png](https://i.loli.net/2020/07/23/B6lcfzdO1eJD3oR.png)

![image.png](https://i.loli.net/2020/07/23/9Ft52JcfU8BpZoO.png)

![image.png](https://i.loli.net/2020/07/23/f8IaYMlGwJqupDR.png)

![image.png](https://i.loli.net/2020/07/23/yYsVEqhPjduS9NI.png)

![image.png](https://i.loli.net/2020/07/23/bP4hnfIWi86OaVZ.png)

从图中可以看出，使用不同的方案处理过后的数据进行降维会有不同的聚类结果，此项研究的意义在于，老师可以对学生在哪些题目上的表现相近有一个了解，进而再结合题目的总体得分、完成率等指标可以对学生的整体的编程能力有一个把握，从而制定更加据有针对性的教学计划。

#### 题目质量鉴定器

对于一道题目，我们该如何评判这道题出的怎么样呢？也就是说如何评判这道题的题目质量如何呢？固然可以用一个循环神经网络读一遍题目的题干部分，然后靠机器"智慧地"学习出题目的好坏，但这种做法的复杂度太高，所需要的数据集太大，而且没有考虑做题人群的特质，所以我们希望从做题人群的角度出发，即站在学生群体的角度来评判每道题目的质量，生活经验告诉我们，如果一道题目的方差太小，即拉不开差距，或者这道题目愿意做的人太少，那么这道题目的质量一定不会太高，当然，在这里我们考虑的标准还不止这些。

对于题目质量评估，我们采用的是之前保存的分析结果`cases_analysis_final.csv`，这个文件一共有882行，26列，对于取值范围较大的列，我们已经做过了归一化了，所以在计算题目质量时可以考虑给各个指标相近的权重。具体的计算方法如下面的代码所示，对于一道好的题目，我们希望它的完成率是高的，平均分、难易程度、平均提交次数都应在一个合理的范围内，不能太大或太小，同时期望方差要尽可能大。

```python
def getGoodnessOfCase(caseId):
    return 30+cases_analysis_result.iloc[caseId]['finishRate']-abs(cases_analysis_result.iloc[caseId]['scoreIgnoreUndo']-cases_analysis_result['scoreIgnoreUndo'].mean())+cases_analysis_result.iloc[caseId]['scoreVar']-abs(cases_analysis_result.iloc[caseId]['difficultDegree']-cases_analysis_result['difficultDegree'].mean())-abs(cases_analysis_result.iloc[caseId]['uploadAvgInFact']-3)
def getGoodnessOfType(typeId):
    s=0
    for caseId in caseIdsByType[typeId]:
        s+=getGoodnessOfCase(caseId)
    return s/len(caseIdsByType[typeId])
def getGoodnessOfGroup(groupId):
    s=0
    for caseId in groupCaseIds[groupId]:
        s+=getGoodnessOfCase(caseId)
    return s/len(groupCaseIds[groupId])
```

下图是按题目类型分类的题目平均质量统计条形图，八种类型的题目的平均质量差异不是很大，这也符合我们的预期。

![image.png](https://i.loli.net/2020/07/23/p3QVeablNLuZhnj.png)

下图是按小组分类的题目平均质量统计图。

![image.png](https://i.loli.net/2020/07/23/d2Snli4L9IXvBbg.png)

#### 面向测试用例编程检测器

在实际的数据分析中，我们发现有许多代码明显是面向测试用例的，即套出老师的测试用例，然后直接`if-else`来输出结果，而不做任何逻辑上的处理，违背了老师让同学们作题的初衷，也不利于学生的进步，有损公平性，为此我们特地编写了一个面向测试用例编程检测器，用来判定一个python文件中的代码是不是又面向测试用例的嫌疑，具体的检测方法是：读取一个python文件，计算这份文件的有效行数，即总行数减去空行和注释的行，有效行数记为`count`，然后数出含有`print`的行数，记为`pcount`，再数出含有`if`、`elif`或`else`的行数，记为`ifcount`，如果总行数过少、`if`语句过多或`print`语句过多，则会被标记为有面向测试用例的嫌疑，于是这道题会被交给老师做进一步的处理。具体的实现如下，有面向测试用例的嫌疑则会返回`True`。

```python
def test_cases_oriented(path):
    f=open(path,'r',encoding='utf-8')
    #以下为测行数、print数
    count=1  #行数
    pcount=0 #print数
    ifcount=0 # if elif else数
    line=f.readline()
    line=line[:-1]
    while line:
        if line.count("print")>0:
            pcount+=1
        if line.count("if")>0 or line.count("elif") or line.count("else")>0:
            ifcount+=1
        line=line.strip()  #去掉前后空格
        if line: #不为空行
            if not line[0]=="#": #不为注释行
                count+=1
        line=f.readline()
    #以下为检测
    return pcount>5 or (pcount/count)>0.3 or count<5 or ifcount>5 or ifcount/count>0.3
```

以下是两个例子，下面两段代码都会被判定为有面向测试用例编程的嫌疑。

![image.png](https://i.loli.net/2020/07/22/pFrmoYxLTiWhE6H.png)

![image.png](https://i.loli.net/2020/07/22/bPrV6kIBgClyAwu.png)

![image.png](https://i.loli.net/2020/07/22/vtIU5xc8L3SOZjJ.png)

![image.png](https://i.loli.net/2020/07/22/RcFnqte5KL7WTvP.png)

#### 代码抄袭检测器

由于python练习题有200道，数量较大，难免会有同学想要"借鉴"他人的代码，这种行为不利于老师正确评估学生的编程能力，也不利于学生提高自己的编程水平，更有损公平性，同时也可以检测代码是否抄袭了答案，所以我们编写了一个基于`pycode_similar`和`difflib`的代码抄袭检测器，具体而言，读取两个python文件并解析，利用计算差异的辅助工具`difflib`，计算两个文本的序列相似度，若该值超过预先设定的阈值，则会被判定为有抄袭的嫌疑。具体的实现如下所示。

```python
def copy_detector(path1,path2,threshold=0.6):
    with open(path1,encoding='utf-8') as file1:
        code1=''
        ls=file1.readlines()
        for line in ls:
            temp=line.lstrip()
            if temp and not temp.startswith('#'):
                code1+=temp
    with open(path2,encoding='utf-8') as file2:
        code2=''
        ls=file2.readlines()
        for line in ls:
            temp=line.lstrip()
            if temp and not temp.startswith('#'):
                code2+=temp
    return difflib.SequenceMatcher(None,code1,code2).ratio()>=threshold
```

以下是两个有抄袭嫌疑的代码示例。

![image.png](https://i.loli.net/2020/07/23/2w51QKDVJuIfpsU.png)

![image.png](https://i.loli.net/2020/07/23/gXRW8NotFpky3fU.png)

![image.png](https://i.loli.net/2020/07/23/sQUhyX4CpMYT6Ja.png)

![image-20200723115244939](C:\Users\60960\AppData\Roaming\Typora\typora-user-images\image-20200723115244939.png)

### 学生视角

#### 编程能力评估

#### 生成编程学习路线

#### 寻找编程搭档

#### 自动推荐代码

## 附录

此部分主要是一些帮助读者理解的数据可视化成果展示。

对于每一道题目，我们都提供了画了三张图，以供读者了解这道题的基本情况。分别是分数分布图，忽略未做学生的平均分、不忽略没做学生的平均分条形图，以及完成率、难度、运行时间等指标的雷达图。

```python
def getViewByCaseId(x):
    mp.figure(figsize=(15,8), dpi=80)
    mp.figure(1)
    #
    ax1=mp.subplot(131)
    label='CaseId:'+str(x)+' 分数分布'
    caseScore=pd.DataFrame(np.array(caseAllScores[x]).reshape(1,-1))
    bins = np.arange(0, 101, 10)
    ax1.set_title(label,loc='left')
    ax1.set_xticks(range(0,101,10))
    ax1.set_xlabel('分数段')
    ax1.set_ylabel('人数')
    sns.distplot(caseScore,bins=10,ax=ax1,rug=True,rug_kws = {'color' : 'r'},
            kde_kws = {'bw' : 2})
    #
    ax2=mp.subplot(132)
    group=["Count","Ignore"]
    mean=np.array([caseScoreCountUndo[x],caseScoreIgnoreUndo[x]])
    ax2.set_title('Avarage Score')
    mp.bar(group,mean,color='c',width=0.6,align='center')
    #雷达图
    ax3=mp.subplot(133, polar=True)
    labels=['运行时间','完成率','难度','时间跨度','上传次数']
    kinds=['CaseId:'+str(x)]
    CR=pd.DataFrame([[runtime[x]*100000,fr[x]*100,difficulty[x]*30,spanTime[x]/5,uploadSum[x]]]
    ,columns=['运行时间','完成率','难度','时间跨度','上传次数'])
    result=pd.concat([CR,CR['运行时间']],axis=1)
    centers=np.array(result.iloc[:,:])
    n=len(labels)
    angle=np.linspace(0,2*np.pi,n,endpoint=False)
    angle=np.concatenate((angle,[angle[0]]))
    pic=[]
    for i in range(len(kinds)):
        pic.append(ax3.plot(angle,centers[i],linewidth=2,label=kinds[i]))
    ax3.set_thetagrids(angle*180/np.pi,labels)
    ax3.legend(loc='lower left')
    mp.show()
for i in random.sample(range(882),5):
    getViewByCaseId(i)
```

下面是几个示例。

![image.png](https://i.loli.net/2020/07/23/JEKyPRgUdtsqObm.png)

![image.png](https://i.loli.net/2020/07/23/tPc24UqvdlXb7ya.png)

![image.png](https://i.loli.net/2020/07/23/eV5Xtv1Td3roKcJ.png)

![image.png](https://i.loli.net/2020/07/23/FmOGy8aQrX1Hg7w.png)

![image.png](https://i.loli.net/2020/07/23/6Dk4WRTs1Z8H9gc.png)

