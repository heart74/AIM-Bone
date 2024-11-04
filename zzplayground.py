def check(num):
    s = str(num)
    return s==s[::-1]

def check2(num,n):
    arr = []
    while num:
        arr.append(num%n)
        num = num//n
    return arr[::-1]==arr

dp = [[0]*31 for _ in range(10)]
for k in range(2,10):
    count = 0
    summ = 0
    now = 0
    now2 = 0
    for i in range(10):
        if check2(i,k):
            summ+=i
            count+=1
        dp[k][count]=summ
    while count<31:
        now2+=1
        numm = int(str(now2)+str(now2)[::-1])
        if check2(numm,k):
            summ+=numm
            count+=1
        dp[k][count]=summ
        # print(now)
        print(dp[k])
print(dp)