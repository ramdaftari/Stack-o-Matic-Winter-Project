# Part 1
import pandas as pd

# Load the dataset
df = pd.read_csv('/content/sample_data/california_housing_train.csv',header = 0)
print(df.head(5))
print(df.tail(10))

# Data Summary
print(df.describe())
unique_vals = df['total_bedrooms'].unique()
print(len(unique_vals))

# Data Transformation
df['total_bedrooms_per_total_rooms'] = df['total_bedrooms']/df['total_rooms']
mean = '%.3f'%df['total_bedrooms_per_total_rooms'].mean()
stan_dev = '%.3f'%df['total_bedrooms_per_total_rooms'].std()
print("The mean of the total bedrooms per total rooms is ", mean)
print("The standard deviation of the total bedrooms per total rooms is ", stan_dev)

# Data Filtering
income_index = df['median_income']>5
print(df[income_index])
room_num_index = (df['total_rooms']>10000) & (df['median_house_value']<150000)
print(df[room_num_index])

# Export Modified Data
df.to_csv('/content/sample_data/my_file.csv')

# Part 2
import numpy as np

# Array Creation and Indexing
arr1 = np.arange(1,21,1)
arr2 = arr1[0::2]
print(arr2)
arr3 = np.random.randint(10,100,(5,4))
print(arr3)
arr4 = arr3[0:3,0:2]
print(arr4)

# Array Manipulation
arr5 = arr1.reshape((4,5))
print(arr5)
arr6 =arr1.ravel(order='C')
print(arr6)
arr7 = np.random.randint(1,5,(5,4))
print(arr7)
arr8 = np.dot(arr5,arr7)
print(arr8)

# Boolean and  Fancy Indexing
arr9 = np.random.randint(0,100,50)
print(arr9)
bool_mask = arr9>50
print(arr9[bool_mask])
arr10 = arr9[[2,4,7,9]]
print(arr10)

# Statistical Operations
print(np.mean(arr3, axis=0))
print(np.mean(arr3, axis=1))
print(np.max(arr3, axis=0))
print(np.max(arr3, axis=1))
print(np.sum(arr3, axis=0))
print(np.sum(arr3, axis=1))

# Part 3

# Loops and Conditional Statements
num1 =0
num2 = 1
while(num1<=100):
    print(num1)
    temp = num2
    num2 = num1+num2
    num1 = temp

for i in range(1,51):
    if(i%3==0 & i%5!=0):
        print("Fizz")
    if(i%3!=0 & i%5==0):
        print("Buzz")
    if(i%3==0 & i%5==0):
        print("FizzBuzz")

# Variable Length Arguments
#def find_median(numbers):

def accept_nums():
    nums = []
    n = int(input("How many numbers do you want to enter?"))
    for i in range(n):
        nums.append(int(input("Enter a number: ")))
    return nums

def calculate_average(nums):
    average = 0
    n = 0
    for i in nums:
        n+=1
        average = (average*(n-1) + i)/n
    average = '%.3f'%average
    print(average)


# Error Handling
def div_nums(a,b):
    try:
        return a/b
    except ZeroDivisionError:
        return "Cannot divide by zero!"