# LeetCode

## Algorithms

### 2. Add Two Numbers 

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        res = ListNode(0)
        head = res
        x = 0
        while l1 and l2:
            res.next = ListNode(0)
            res = res.next
            x = l1.val + l2.val + x // 10
            res.val = x % 10
            l1 = l1.next
            l2 = l2.next   
        while l1:
            res.next = ListNode(0)
            res = res.next
            x = l1.val + x // 10
            res.val = x % 10
            l1 = l1.next
        while l2:
            res.next = ListNode(0)
            res = res.next
            x = l2.val + x // 10
            res.val = x % 10
            l2 = l2.next
        if x // 10:
            res.next = ListNode(1)
        return head.next
```

Runtime: 100 ms, faster than 12.91% of Python3 online submissions for Add Two Numbers.

Memory Usage: 14.3 MB, less than 72.40% of Python3 online submissions for Add Two Numbers.

### 3. Longest Substring Without Repeating Characters 

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic = {}  # char index dictionary
        res = 0  # output
        now = 0  # longest length till the i-th char
        for i in range(len(s)):
            last = i + 1  # distance to the last same char
            if s[i] in dic:
                last -= dic[s[i]]
            dic[s[i]] = i + 1  # update char index dictionary
            if now + 1 <= last:
                now += 1
                if now > res:
                    res = now
            else:
                now = last
        return res
```

Runtime: 66 ms, faster than 58.74% of Python3 online submissions for Longest Substring Without Repeating Characters.

Memory Usage: 14.2 MB, less than 80.00% of Python3 online submissions for Longest Substring Without Repeating Characters.

### 5. Longest Palindromic Substring

**Method 1: let the center of substring take char from head to tail.**

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = s[0]
        n = len(s)
        for i in range(n):
            j = 1
            while i - j >= 0 and i + j < n and s[i-j] == s[i+j]:
                j += 1
            if 2*j - 1 > len(res):
                res = s[i-j+1:i+j]
            j = 0
            while i - j >= 0 and i + j + 1 < n and s[i-j] == s[i+j+1]:
                j += 1
            if 2*j > len(res):
                res = s[i-j+1:i+j+1]
        return res
```

Runtime: 1508 ms, faster than 40.74% of Python3 online submissions for Longest Palindromic Substring.

Memory Usage: 14.3 MB, less than 60.97% of Python3 online submissions for Longest Palindromic Substring.

**Method 2: let the center of substring take char from center to sides of the original string.**

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = s[0]
        n = len(s)
        i = (n - 1) // 2
        while 2 * i + 2 > len(res):
            j = 1
            while i - j >= 0 and i + j < n and s[i-j] == s[i+j]:
                j += 1
            if 2*j - 1 > len(res):
                res = s[i-j+1:i+j]
            j = 0
            while i - j >= 0 and i + j + 1 < n and s[i-j] == s[i+j+1]:
                j += 1
            if 2*j > len(res):
                res = s[i-j+1:i+j+1]
            i = n - 1 - i
            j = 1
            while i - j >= 0 and i + j < n and s[i-j] == s[i+j]:
                j += 1
            if 2*j - 1 > len(res):
                res = s[i-j+1:i+j]
            j = 0
            while i - j >= 0 and i + j + 1 < n and s[i-j] == s[i+j+1]:
                j += 1
            if 2*j > len(res):
                res = s[i-j+1:i+j+1]
            i = n - i - 2
        return res
```

Runtime: 160 ms, faster than 95.96% of Python3 online submissions for Longest Palindromic Substring.

Memory Usage: 14.4 MB, less than 60.97% of Python3 online submissions for Longest Palindromic Substring.

### 6. ZigZag Conversion 

**Method 1: create `numRows` lists, distribute each char to the corresponding list, then combine.**

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        xlist = []
        for i in range(numRows):
            xlist.append([])
        i = 0
        step = 1
        for char in s:
            xlist[i].append(char)
            i += step
            if i == numRows - 1:
                step = -1
            elif i == 0:
                step = 1
        res = ''
        for i in range(numRows):
            res = res + ''.join(xlist[i])
        return res
```

Runtime: 97 ms, faster than 20.36% of Python3 online submissions for ZigZag Conversion.

Memory Usage: 14.3 MB, less than 87.98% of Python3 online submissions for ZigZag Conversion.

**Method 2: extract part of the string according to the remainder.**

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        else:
            p = 2 * numRows - 2
            res = s[::p]
            for i in range(1, numRows-1):
                res1 = list(s[i::p])
                res2 = list(s[p-i::p])
                temp = res1 + res2
                temp[::2] = res1
                temp[1::2] = res2
                res = res + ''.join(temp)
            res = res + s[numRows-1::p]
            return res
```

Runtime: 83 ms, faster than 27.94% of Python3 online submissions for ZigZag Conversion.

Memory Usage: 14.3 MB, less than 87.98% of Python3 online submissions for ZigZag Conversion.

### 8. String to Integer (atoi) 

```python
class Solution:
    def myAtoi(self, s: str) -> int:
        s = s.strip()
        if s == '':
            return 0
        sign = 1
        if s[0] == '-':
            sign = -1
            s = s[1:]
        elif s[0] == '+':
            s = s[1:]
        i = 0
        while i < len(s) and s[i].isdigit():
            i += 1
        if i == 0:
            return 0
        res = int(s[:i]) * sign
        if res < -2**31:
            res = -2**31
        elif res > 2**31 - 1:
            res = 2**31 - 1
        return res
```

Runtime: 32 ms, faster than 83.47% of Python3 online submissions for String to Integer (atoi).

Memory Usage: 14.4 MB, less than 25.41% of Python3 online submissions for String to Integer (atoi).

### 11. Container With Most Water 

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        i = 0
        j = len(height) - 1
        res = min(height[i], height[j]) * j
        while j > i:
            if height[i] > height[j]:
                j -= 1
            else:
                i += 1
            now = min(height[i], height[j]) * (j - i)
            if now > res:
                res = now
        return res
```

Runtime: 716 ms, faster than 79.05% of Python3 online submissions for Container With Most Water.

Memory Usage: 27.6 MB, less than 22.83% of Python3 online submissions for Container With Most Water.



## Database

### 175. Combine Two Tables 

```mssql
/* Write your T-SQL query statement below */
SELECT FirstName, LastName, City, State 
FROM (Person LEFT JOIN Address 
ON Person.PersonId = Address.PersonId)
```

### 176. Second Highest Salary 

```mysql
# Write your MySQL query statement below
SELECT (
    SELECT DISTINCT 
        Salary
    FROM 
        Employee 
    ORDER BY Salary DESC
    LIMIT 1 OFFSET 1) AS SecondHighestSalary
```

