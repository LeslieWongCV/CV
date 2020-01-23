import numpy as np
from collections import Counter



list = np.array([[2,2,2,2,3,3],[0,0,0,2,2,0],[0,0,0,1,3,2],[0,0,4,2,1,0],[4,2,1,4,2,0],[1,0,3,1,0,0]])
#list = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])
#list = np.array(range(20)).reshape(4,5) 生成顺序矩阵
#list = np.random.randint(0,4,(6,6))
print('Oringial List:','\n',list)

matrix = np.array(list)
#a=np.argmax(np.bincount(matrix.flat))
#np.bincount返回list里0-9的出现次数
#np.argmax 返回最大数的索引 两函数联合使用找到list里出现最多次数的数字

list_1=[]
list_2=[]
for i in range(0,list.shape[0]):#黑白格分开
    for j in range(0,list.shape[1]):
        if(((i%2==0)&(j%2==0))+((i%2!=0)&(j%2!=0))):
            list_1.append(matrix[i][j])
        else:
            list_2.append(matrix[i][j])

print('list1:',list_1)
print('list2:',list_2)
list1_max = np.argmax(np.bincount(list_1))
list2_max = np.argmax(np.bincount(list_2))
print('What factor list1 has most:',list1_max)
print('What factor list2 has most:',list2_max)

list1_index = np.bincount(list_1)
list2_index = np.bincount(list_2)
print('list1 index:',list1_index)
print('list2 index:',list2_index)

if (list1_max==list2_max): #如果相等

    list1_index[np.argmax(list1_index)] = np.min(list1_index)#[0,0,5,4,1,2] --> [0,0,0,4,1,2] 最大的变最小，第二变最大

    list2_index[np.argmax(list2_index)] = np.min(list2_index)

    if(np.max(list1_index)>>np.max(list2_index)): #第二大的数出现的次数（出现第二多的数的次数
        list1_max = np.argmax(list1_index)
    else:
        list2_max = np.argmax(list2_index)


print('New max factor for list1:',list1_max)
print('New max fatcor for list2:',list2_max)
print('Updated index for list1:',list1_index)
print('Updated index for list2:',list2_index)


list_coret = np.zeros((list.shape[0],list.shape[1]),npq.int)
for i in range(0,list.shape[0]):
    for j in range(0,list.shape[1]):
        if (((i % 2 == 0) & (j % 2 == 0)) + ((i % 2 != 0) & (j % 2 != 0))):
            list_coret[i][j] = list1_max
        else:
            list_coret[i][j] = list2_max


step = list.shape[0]*list.shape[1] - np.max(list1_index) - np.max(list2_index)
print('Corrected matrix:','\n',list_coret)
print("It takes %s step(s) to convert"%step)





