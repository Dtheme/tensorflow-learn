"""
Numpy的一些基本使用
"""

import numpy as np

# 定义一个数组,一维的矩阵就是一个基本的数组
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr)

# 数组转成矩阵使用
matrixA = np.array([[1,2,3],[2,3,4]])
print(matrixA)

# shape 是指矩阵的形式 例如我这里定义的是一个2行3列的矩阵。dtype获取数据类型
print('shape:', matrixA.shape, '数据类型：', matrixA.dtype)

# 创建一个全是0的单位矩阵 需要shape属性确定 这个0矩阵的形式 例如：定义一个3行5列的单位0矩阵
zeroMatrix = np.zeros(shape=(3,5))
print(zeroMatrix)

# 同上创建单位1矩阵和空矩阵 分别是4行4列和3行4列的矩阵
onesMatrix = np.ones(shape=(4,4))  
emptyMatrix = np.empty(shape=(3,4))  # 注意空矩阵不是全为nil值 而是 自己填充了一些随机的小值进去，防止异常

print(onesMatrix)
print(emptyMatrix)

# arange(1，10，2）表示起始位置，终点位置，步长，我使用的方式是从0开始到10步长1
matrixB = np.arange(10)  
matrixC = np.array([[1,2,3],[4,5,6],[7,8,9],[2,34,69]])
print("切片：",matrixB[0:5],"---",matrixC[1:3])

# 矩阵的切片 等量切片
matrixA_Equ = np.split(matrixC,4,axis=0)

# 不等切片
matrixA_unEqu = np.array_split(matrixC, 5, axis=1)  # 按照指定方向将matrixC切5片 如果后面是空就返回空数组
print("等分切片", matrixA_Equ,"不等分切片", matrixA_unEqu)

rangeArr = np.arange(2,19,3)
print("rangeArr:", rangeArr)

# 根据比较条件 将元素逐个比较返回ture或者false
compareArr = np.arange(9)  # [0,1,2,3,4,5,6,7,8]
print(compareArr < 4,"\n","最大值:", max(compareArr),"最小值:", min(compareArr))
# 也可以重新对compareArr进行序列 例：序列成3行3列的矩阵,序列化以后是tuple 不能使用serialCompareArr<3去比较了
serialCompareArr = compareArr.reshape(3,3)  # [[0,1,2],[3,4,5],[6,7,8]]
print("序列化：", serialCompareArr)
# 在把数组序列化成矩阵以后，看看矩阵中的max和min操作??? 注意：axis = 0表示行，1表示列
print("最大值:", np.max(serialCompareArr),"最小值:",np.min(serialCompareArr),"行最小值:", np.min(serialCompareArr,axis=0),"列最大值:",np.max(serialCompareArr,axis=1))
print("第一列最大值的位置：",np.argmax(serialCompareArr, axis=1))
print("平均值：", np.mean(serialCompareArr))
print("累加值逐个输出和累加值",np.cumsum(serialCompareArr),np.sum(serialCompareArr))
print("逐差值：", np.diff(serialCompareArr))

# 切片 例如matixB[0:5] 切片取0，1，2，3，4
# 对切片进行赋值 或者matixB[0:5] = 8888会将切片部分的所有值都赋值为8888
matrixB[0:5] = np.array([8888, 0000, 1111, 2222, 6666])  
print("切片后：",matrixB)

# operator :运算 数组之间可以直接运算
opArray1 = np.array([1,2,3,4])
opArray2 = np.array([5,6,7,8])
addRes = opArray1 + opArray2
subRes = opArray1 - opArray2
mulRes = opArray1 * 10
print("源数组:", opArray1,opArray2,"加", addRes, "减", subRes,"乘以标量", mulRes)

# 一些其他的基本运算 如：对opArray1的每一个值求sin值然后乘以2
opVal3 = 2 * np.sin(opArray1) 
print(opVal3)
# ** 是次方 这里的意思是对opArray1中的元素 分别求opArray2对应的次方数 即1^5,2^6,3^7,4^8
opVal4 = opArray1**opArray2
print(opVal4)

# 数组或者矩阵间的乘法比较特殊: 点乘和叉乘 还有数组元素逐个相乘
mulArr1 = np.array([1,2,3,4])
mulArr2 = np.array([5,6,7,8])

# 点乘 2种语法糖 结果是一样的
# A=[a1,a2,a3],B=[b1,b2,b3] 
# A·B=a1b1+a2b2+a3b3
dotRes1 = np.dot(mulArr1, mulArr2)
dotRes2 = mulArr1.dot(mulArr2)
print(dotRes1,dotRes2)

# 叉乘
# A×B=[a2b3-a3b2,a3b1-a1b3,a1b2-a2b1] 
mulRes = np.multiply(mulArr1, mulArr2)
print(mulRes)

# 迭代 
forArr = np.arange(3,15).reshape((3,4))
print(forArr)
# 逆矩阵 inverse of a matrix.
inv_forArr1 = forArr.T
# inv_forArr2 = np.linalg.inv(forArr) 这种方式只能被用于square类型的矩阵。
print(inv_forArr1)
print(forArr[1][1])  # 取某个元素值

# 迭代 注意迭代列实质是先转置矩阵求逆矩阵 然后再迭代出行
print("先打印一下源矩阵看看：\n", forArr)
for row in forArr:
    print("迭代行：", row)

for column in forArr.T:
    print("迭代列：", column)

# 矩阵合并
combineMatrixA = np.array([1, 1, 1])
combineMatrixB = np.array([2, 2, 2])
# 上下合并 vertical combine
combineMatrixV = np.vstack((combineMatrixA,combineMatrixB))
print("A:\n",combineMatrixA,"\nB:\n", combineMatrixB,"\nC:\n", combineMatrixV)
print(combineMatrixA.shape,combineMatrixV.shape)  # (3,) (2,3) 第一个是一维的3个元素，第二个是二维的2行3列

# 横向合并horizontal combine
combineMatrixH = np.hstack((combineMatrixA,combineMatrixB))
print("A:\n",combineMatrixA,"\nB:\n", combineMatrixB,"\nC:\n", combineMatrixH)
print(combineMatrixA.shape,combineMatrixH.shape)  # (3,) (6,) 第一个是一维的3个元素，第二个也是一维的6个元素 它们是数组

print(combineMatrixH[:,np.newaxis])  # 将一个横向的数列转换成一个一行一列的矩阵

# 多个数组合并的时候使用下面的方式（也可用于合并2个） axis = 0 横向合并 1 纵向合并
# [[1 1 1]
# [2 2 2]] 和[[5], [6]] 合并
# 合并结果：
# [[1 1 1 5]
# [2 2 2 6]]
combineMatrixC = np.array([[5], [6]])
print("...",combineMatrixV)
combineMore = np.concatenate((combineMatrixV,combineMatrixC), axis=1)
print("concatenate合并的方式：", combineMore)



