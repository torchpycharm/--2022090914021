# python学习

## 基础入门

#### 1.#是注释；

#### 2. print函数 

数字直接打，字符串加单引号，表达式直接打输出结果，自动换行，不换的话；

id（）函数显示标识

type()显示类型

字符串类型：str

#### 3.

数据输出到文件中：fp(一个变量？）= open（‘D：/文件名','a+')

 #file=fp

  fp.close()

#### 4.

最后一个字符不能是\  最前面加r或R可使转义字符失效(注意python严格区分大小写；

#### 5.

写二八十六进制数加“0b”，0o，0x，命名标识符不能以数字开头；

#### 6. python关键字

'False', 'None', 'True', '__peg_parser__', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']

#### 7.浮点小数保留问题：

导入模块decimal

from decimal import Decimal

print(Decimal('1.1')+Decimal('2.2'))

#### 8.布尔变量

可以通过加减法直接转化成True=1，False=0；

布尔运算符

and两项都成立

 or 有一项成立

not对运算数（布尔类型）取反

 in 判断元素是否在集合中

not in

![image-20221009215329478](C:\Users\29021\AppData\Roaming\Typora\typora-user-images\image-20221009215329478.png)

#### 9.

用str() int() float()可以实现类型转换 #直接写在括号里就是了

str转int 字符串必须是数字串，float转成int类型会舍去小数部分

#### 10.运算符

// 整除符号（注意一正一负时向下取整-2.3—— -3）

**幂运算

系列赋值：a,b,c=1,2,3  a,b=b,a

##### 位运算符：

将数据转换成二进制进行计算

位与&：对应位数都是1，结果数位才是1，否则为0

