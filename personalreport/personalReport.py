import streamlit as st
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False # 解决保存图像是负号'-'显示为方块的问题
user_info=pd.read_csv('user_info.csv')
type_dict={'排序算法':0,'查找算法':1,'图结构':2,'树结构':3,'数字操作':4,'字符串':5,'线性表':6,'数组':7}
sidebar = st.sidebar
type = sidebar.selectbox("选择功能", ('查看报告', '获取学习路线&推荐题目', '寻找编程搭档'))
value=sidebar.text_input(label='输入学生id')
if value!='' and type=='查看报告':
    st.markdown('# 学生个人编程报告')
    st.markdown('---')
    st.markdown('### 1.编程时间分布分析')
    st.markdown('---')
    id = int(value)
    data = user_info[user_info['id'] == id].iloc[:, 2:26]
    labels = ['0:00', '1:00', '2:00', '3:00', '4:00', '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00', '12:00',
              '13:00', '14:00', '15:00', '16:00', '17:00'
        , '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
    fig=plt.figure(figsize=(12,7))
    ax=plt.subplot(111)
    rate = user_info[user_info['id'] == id].iloc[:, 26:50]
    st.markdown(' ')
    st.markdown(' ')
    st.write('**滑动查看各时间段提交比例**')
    time = st.slider('请选择一个时间段', 0, 23, 0, 1)
    st.write('该学生在', time, ':00 - ', time + 1, ':00的提交比例为', round(rate.values[0][time],4) * 100, '%')
    st.write('**分布图及做题时间类型占比**')
    st.markdown('##### 做题时间类型分为4种')
    st.table(pd.DataFrame({' day ':['上午型'],'afternoon':['下午型'],'night':['晚上型'],'deep':['深夜型']}))
    plt.style.use('seaborn')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('time')
    ax.set_ylabel('upload_num')
    ax.bar(labels, data.values[0], width=1.0, color='purple')
    ax.plot(labels, data.values[0])
    st.pyplot()
    data = user_info[user_info['id'] == id].loc[:, 'dayRate':'deepRate']
    sizes = data.values[0]
    labels = ['morning', 'afternoon', 'night', 'deep']
    explode = (0.1, 0, 0, 0)
    ax = plt.subplot(111)
    ax.pie(sizes, explode=explode, labels=labels,
              shadow=True, startangle=90, wedgeprops={'width': 0.4})
    ax.axis('equal')
    ax.legend(loc='center right', bbox_to_anchor=(1, 0, 0.05, 1))
    st.pyplot()
    if user_info[user_info['id']== id]['timeType'].values[0]==0:
        timeType='上午型'
    elif user_info[user_info['id']== id]['timeType'].values[0]==1:
        timeType='下午型'
    elif user_info[user_info['id']== id]['timeType'].values[0]==2:
        timeType='晚上型'
    else :timeType='深夜型'
    st.write('*可以看出这是一位`'+timeType+"`选手*")
    timespan=user_info[user_info['id'] == id]['avgTimeSpan'].values[0]
    delaydegree=user_info[user_info['id'] == id]['delayDegree']
    st.markdown('**拖延症病情情况**')
    st.markdown('做题平均时间跨度为`'+str(int(timespan))+'`分钟')
    st.write('拖延症指数为`'+str(int(delaydegree))+'`')
    st.markdown('---')
    st.markdown('### 2.编程喜好分析')
    st.markdown('---')
    st.markdown(' ')
    st.markdown(' ')
    labels = list(type_dict.keys())
    color=['dimgray','burlywood','aquamarine','lightcoral','slateblue','gold','skyblue','orange']
    ax=plt.subplot(111)
    like = user_info[user_info['id'] == id].loc[:,
              ['likeDegreeOfType0', 'likeDegreeOfType1', 'likeDegreeOfType2', 'likeDegreeOfType3',
               'likeDegreeOfType4', 'likeDegreeOfType5', 'likeDegreeOfType6', 'likeDegreeOfType7']]
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    st.markdown('**各类型题目喜好程度**')
    ax.barh(np.arange(8), like.values[0], color=color)
    ax.set_yticks(np.arange(8))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('喜好程度')
    st.pyplot()
    l=like.values[0].tolist()
    maxlike=l.index(max(l))
    st.markdown('*最喜欢的类型是`'+labels[maxlike]+'`类型*')
    minlike = l.index(min(l))
    st.markdown('*最讨厌的类型是`' + labels[minlike] + '`类型*')
    st.markdown('---')
    st.markdown('### 3.编程能力')
    st.markdown('---')
    st.markdown('**各类型完成率**')
    data = user_info[user_info['id'] == id].loc[:,
           ['finishRateOfType0', 'finishRateOfType1', 'finishRateOfType2', 'finishRateOfType3',
            'finishRateOfType4', 'finishRateOfType5', 'finishRateOfType6', 'finishRateOfType7']]
    ax = plt.subplot(111)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(labels)
    ax.bar(np.arange(8), data.values[0], color=color)
    st.pyplot()
    st.markdown('**各类型平均分**')
    st.write('Count代表考虑没做的题目，Ignore则代表忽略')
    data = user_info[user_info['id'] == id].loc[:,
           ['avgScoreOfType0', 'avgScoreOfType1', 'avgScoreOfType2', 'avgScoreOfType3',
            'avgScoreOfType4', 'avgScoreOfType5', 'avgScoreOfType6', 'avgScoreOfType7']]
    datai= user_info[user_info['id'] == id].loc[:,
           ['avgScoreIgnoreUndoOfType0', 'avgScoreIgnoreUndoOfType1', 'avgScoreIgnoreUndoOfType2', 'avgScoreIgnoreUndoOfType3',
            'avgScoreIgnoreUndoOfType4', 'avgScoreIgnoreUndoOfType5', 'avgScoreIgnoreUndoOfType6', 'avgScoreIgnoreUndoOfType7']]
    ability=user_info[user_info['id'] == id].loc[:,
           ['userAbilityOfType0', 'userAbilityOfType1', 'userAbilityOfType2', 'userAbilityOfType3',
            'userAbilityOfType4', 'userAbilityOfType5', 'userAbilityOfType6', 'userAbilityOfType7']]
    l=ability.values[0].tolist()
    maxl=l.index(max(l))
    minl=l.index(min(l))
    x = np.arange(len(labels))
    width = 0.35
    ax1 = plt.subplot(111)
    ax1.set_title("Average Score")
    ax1.set_ylabel('scores')
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels)
    rect1 = ax1.bar(x - width / 2, data.values[0], width, label='Count')
    rect2 = ax1.bar(x + width / 2, datai.values[0], width, label='Ignore')
    ax1.legend([rect1, rect2], ['Count', 'Ignore'], loc='lower right')
    st.pyplot()
    result = pd.concat([ability, ability['userAbilityOfType0']], axis=1)
    centers = np.array(result.iloc[:, :])
    n = len(labels)
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angle = np.concatenate((angle, [angle[0]]))
    st.markdown('**编程能力图**')
    ax3 = plt.subplot(111, polar=True)
    ax3.plot(angle, centers[0], linewidth=2, color='cyan')
    ax3.set_thetagrids(angle * 180 / np.pi, labels)
    st.pyplot()
    st.markdown('`' + labels[maxl] + '`类型表现最好，`'+labels[minl]+'`类型还需加强')
    st.markdown('---')
def getCodeRoadForUser(userId):
    user_info=pd.read_csv('user_info.csv')
    abilities=[]
    for i in range(8):
        abilities.append(float(user_info[user_info['id']==userId]['userAbilityOfType'+str(i)]))
    abilities=np.array(abilities)
    return list(np.argsort(abilities))
caseIdsByType=np.load('caseIdByType.npy', allow_pickle=True)
def getRecommendCaseIdsUserLike(userId):
    user_info=pd.read_csv('user_info.csv')
    typeIds=[]
    typeIds1=list(np.argsort(-np.array([float(user_info[user_info['id']==userId]['userAbilityOfType'+str(i)]) for i in range(8)]))[:4])
    typeIds2=list(np.argsort(np.array([float(user_info[user_info['id']==userId]['likeDegreeOfType'+str(i)]) for i in range(8)]))[:4])
    for i in range(4):
        for j in range(i+1):
            typeIds.append(typeIds1[i])
            typeIds.append(typeIds2[i])
    recommendCaseIds=set()
    for i in range(25):
        recommendCaseIds.add(np.random.choice(caseIdsByType[np.random.choice(typeIds)]))
    return list(recommendCaseIds)

def getRecommendCaseIdsUserDislike(userId):
    user_info=pd.read_csv('user_info.csv')
    typeIds=[]
    typeIds1=np.argsort(np.array([float(user_info[user_info['id']==userId]['userAbilityOfType'+str(i)]) for i in range(8)]))[:4]
    typeIds2=np.argsort(-np.array([float(user_info[user_info['id']==userId]['likeDegreeOfType'+str(i)]) for i in range(8)]))[:4]
    for i in range(4):
        for j in range(i+1):
            typeIds.append(typeIds1[i])
            typeIds.append(typeIds2[i])
    recommendCaseIds=set()
    for i in range(25):
        recommendCaseIds.add(np.random.choice(caseIdsByType[np.random.choice(typeIds)]))
    return list(recommendCaseIds)
def geth(result):
    if len(result) % 5 == 0:
        h = 5
    elif len(result) % 4 == 0:
        h = 4
    elif len(result) % 3 == 0:
        h = 3
    elif len(result) % 2 == 0:
        h = 2
    else:
        h = 1
    return h
if value!='' and type=='获取学习路线&推荐题目':
    st.markdown('# 获取学习路线')
    st.markdown('---')
    st.markdown('**此路线为在对学生各类型题目进行客观准确评估后制定，学生可以按照返回的结果按优先级从高到低强化编程练习**')
    labels = list(type_dict.keys())
    result=getCodeRoadForUser(int(value))
    fig = plt.figure(figsize=(12, 7))
    ax = plt.subplot(111)
    colors = ['dimgray', 'burlywood', 'aquamarine', 'lightcoral', 'slateblue', 'gold', 'skyblue', 'orange']
    sizes=np.array([2000,2500,1500,750,500,1000,1200,1800])
    np.random.shuffle(sizes)
    x=np.arange(8)
    plt.style.use('seaborn-dark')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_xticks([])  # 去掉x轴
    ax.set_yticks([])  # 去掉y轴
    ax.plot(x,result,color='k')
    ax.scatter(x,result,c=colors,s=sizes)
    for x, y in zip(x, result):
        ax.text(x, y + 1, labels[y], ha='center', va='bottom', fontsize=15)
    ax.text(0,result[0]-1,'起点', ha='center', va='bottom', fontsize=15)
    ax.text(7, result[-1] - 1, '终点', ha='center', va='bottom', fontsize=15)
    st.pyplot()
    st.markdown('---')
    st.markdown('# 获取推荐题目编号')
    st.markdown('---')
    st.markdown('**自动推荐代码模块将从两个角度给学生推荐题目编号，分别是"心动模式"和"考验模式"**')
    st.markdown('"心动模式"，即给学生优先推荐他所喜欢或擅长的题型的题目')
    st.markdown('"考验模式"，即给学生优先推荐他所不擅长或不喜欢的题目类型的题目')
    mode=st.radio('选择一种模式',('心动模式','考验模式'))
    st.write('题目id列表')
    if mode=='心动模式':
        result=getRecommendCaseIdsUserLike(int(value))
        h=geth(result)
        result=np.array(result).reshape(h,-1)
        st.table(result)
    else:
        result = getRecommendCaseIdsUserLike(int(value))
        h = geth(result)
        result = np.array(result).reshape(h,-1)
        st.table(result)
if value!='' and type=='寻找编程搭档':
    st.markdown('# 获取编程搭档编号')
    st.markdown('---')
    st.markdown('**在寻找编程搭档时，我们将从4个角度出发；**')
    st.markdown('第一个角度是寻找编程时间分布与自己最接近的搭档,能增加一起学习的时间')
    st.markdown('第二个角度是寻找与自己在各类题目上分数分布最接近的同学，即各方面能力与自己最接近的人')
    st.markdown('第三个角度是寻找与自己在各类题目上分数分布差异最大的同学，这有利于大家优势互补')
    st.markdown('第四个角度是"一键寻找"与自己能力最接近的同学，这会综合各种因素，包括平均得分、编程时间等各种因素。')
    id=int(value)
    st.markdown('---')
    mode=st.radio('',('时间角度','能力相似角度','能力互补角度','综合角度'))
    user_partner=pd.read_csv('user_partner.csv')
    st.write('学生id列表')
    if mode=='时间角度':
        result=np.array(eval(user_partner[user_partner['id']==id]['time'].values[0])).reshape(1,-1)
        st.table(result)
    elif mode == '能力相似角度':
        result = np.array(eval(user_partner[user_partner['id'] == id]['scoreClose'].values[0])).reshape(1, -1)
        st.table(result)
    elif mode == '能力互补角度':
        result = np.array(eval(user_partner[user_partner['id'] == id]['scoreFar'].values[0])).reshape(1, -1)
        st.table(result)
    else:
        result = np.array(eval(user_partner[user_partner['id'] == id]['ability'].values[0])).reshape(1, -1)
        st.table(result)