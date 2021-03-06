# LeetCode

[TOC]

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

### 12. Integer to Roman

**Method 1**

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        res = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX'][num % 10]
        num = num // 10
        if num:
            add = ['', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC'][num % 10]
            res = add + res
            num = num // 10
        if num:
            add = ['', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM'][num % 10]
            res = add + res
            num = num // 10
        if num:
            add = ['', 'M', 'MM', 'MMM'][num]
            res = add + res
        return res
```

Runtime: 70 ms, faster than 18.91% of Python3 online submissions for Integer to Roman.

Memory Usage: 14.1 MB, less than 84.38% of Python3 online submissions for Integer to Roman.

**Method 2**

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        ones = ['I', 'X', 'C', 'M']
        fives = ['V', 'L', 'D']
        res = ''
        i = 0
        while num:
            r = num % 10
            add = ''
            if r == 4:
                add = ones[i] + fives[i]
                r -= 4
            elif r == 9:
                add = ones[i] + ones[i+1]
                r -= 9
            elif r >= 5:
                add = fives[i]
                r -= 5
            add = add + ones[i] * r
            res = add + res
            num = num // 10
            i += 1
        return res
```

Runtime: 40 ms, faster than 95.60% of Python3 online submissions for Integer to Roman.

Memory Usage: 14.4 MB, less than 5.13% of Python3 online submissions for Integer to Roman.

### 15. 3Sum 

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        n = len(nums)
        if n < 3:
            return res
        i = 0
        sub = []
        while nums[i] <= 0 and i < n-2:
            if nums[i] != nums[i-1]:
                sub = []
            j = i + 1
            k = n - 1
            while j < k:
                if nums[i] + nums[j] + nums[k] == 0:
                    now = [nums[i], nums[j], nums[k]]
                    if now not in sub:
                        sub.append(now)
                        res.append(now)
                    j += 1
                elif nums[i] + nums[j] + nums[k] < 0:
                    j += 1
                else:
                    k -= 1
            i += 1
        return res
```

Runtime: 4910 ms, faster than 14.37% of Python3 online submissions for 3Sum.

Memory Usage: 17.5 MB, less than 73.29% of Python3 online submissions for 3Sum.

### 16. 3Sum Closest 

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        res = sum(nums[0:3])
        nums.sort()
        n = len(nums)
        i = 0
        while i < n-2:
            j = i + 1
            k = n - 1
            while j < k:
                now = nums[i] + nums[j] + nums[k]
                if now == target:
                    return target
                elif now < target:
                    j += 1
                else:
                    k -= 1
                if abs(now - target) < abs(res - target):
                    res = now
            i += 1
        return res
```

Runtime: 219 ms, faster than 35.61% of Python3 online submissions for 3Sum Closest.

Memory Usage: 14.2 MB, less than 90.16% of Python3 online submissions for 3Sum Closest.

### 17. Letter Combinations of a Phone Number 

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        dic = {'2': ['a', 'b', 'c'], '3': ['d', 'e', 'f'], '4': ['g', 'h', 'i'],
              '5': ['j', 'k', 'l'], '6': ['m', 'n', 'o'], '7': ['p', 'q', 'r', 's'],
              '8': ['t', 'u', 'v'], '9': ['w', 'x', 'y', 'z']}
        res = []
        if digits == "":
            return res
        res.append('')
        for d in digits:
            new = []
            for i in dic[d]:
                new = new + [j+i for j in res]
            res = new
        return res
```

Runtime: 28 ms, faster than 86.81% of Python3 online submissions for Letter Combinations of a Phone Number.

Memory Usage: 14.3 MB, less than 62.18% of Python3 online submissions for Letter Combinations of a Phone Number.

### 18. 4Sum 

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        res = []
        nums.sort()
        n = len(nums)
        if n < 4:
            return res
        i = 0
        sub = []
        while nums[i] <= target / 4 and i < n-3:
            if nums[i] != nums[i-1]:
                sub = []
            j = i + 1
            jtop = (target - nums[i]) / 3
            while nums[j] <= jtop and j < n-2:
                k = j + 1
                l = n - 1
                while k < l:
                    if nums[i] + nums[j] + nums[k] + nums[l] == target:
                        now = [nums[i], nums[j], nums[k], nums[l]]
                        if now not in sub:
                            sub.append(now)
                            res.append(now)
                        k += 1
                    elif nums[i] + nums[j] + nums[k] + nums[l] < target:
                        k += 1
                    else:
                        l -= 1
                j += 1
            i += 1
        return res
```

Runtime: 1488 ms, faster than 31.69% of Python3 online submissions for 4Sum.

Memory Usage: 14.5 MB, less than 27.31% of Python3 online submissions for 4Sum.

### 19. Remove Nth Node From End of List

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        i = 0
        temp = head
        while temp:
            i += 1
            temp = temp.next
        if i == n:
            return head.next
        temp = head
        for j in range(i-n-1):
            temp = temp.next
        temp.next = temp.next.next
        return head
```

Runtime: 51 ms, faster than 14.41% of Python3 online submissions for Remove Nth Node From End of List.

Memory Usage: 14.3 MB, less than 47.38% of Python3 online submissions for Remove Nth Node From End of List.

### 22. Generate Parentheses 

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        dic = {}
        res = [""]
        for k in range(1, n+1):
            dic[k-1] = res
            res = ["(" + i + ")" for i in dic[k-1]]
            for x in range(1, k):
                res = res + ["(" + i + ")" + j for i in dic[x-1] for j in dic[k-x]]
        return res
```

Runtime: 36 ms, faster than 68.09% of Python3 online submissions for Generate Parentheses.

Memory Usage: 14.6 MB, less than 67.55% of Python3 online submissions for Generate Parentheses.

### 24. Swap Nodes in Pairs 

```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        elif not head.next:
            return head
        temp = head
        prev = ListNode()
        prev.next = head.next
        res = head.next
        while temp and temp.next:
            left = temp
            right = temp.next
            prev.next = right
            prev = left
            temp = right.next
            left.next = temp
            right.next = left
        return res
```

Runtime: 44 ms, faster than 15.73% of Python3 online submissions for Swap Nodes in Pairs.

Memory Usage: 14.2 MB, less than 48.53% of Python3 online submissions for Swap Nodes in Pairs.

### 29. Divide Two Integers 

```python
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        sign = 1
        if dividend < 0:
            dividend = -dividend
            sign = -sign
        if divisor < 0:
            divisor = -divisor
            sign = -sign
        i = 0
        temp = divisor
        power = 1
        divlist = []
        reslist = []
        while temp <= dividend:
            divlist.append(temp)
            reslist.append(power)
            temp += temp
            power += power
            i += 1
        res = 0
        for j in range(i):
            if divlist[i-1-j] <= dividend:
                dividend -= divlist[i-1-j]
                res += reslist[i-1-j]
        if sign == -1:
            res = -res
        if res > 2147483647:
            return 2147483647
        return res
```

Runtime: 46 ms, faster than 21.25% of Python3 online submissions for Divide Two Integers.

Memory Usage: 14.3 MB, less than 53.26% of Python3 online submissions for Divide Two Integers.

### 31. Next Permutation 

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        i = n-1
        while i > 0 and nums[i-1] >= nums[i]:
            i -= 1
        if i == 0:
            nums.reverse()
            return
        j = i-1
        while i < n and nums[j] < nums[i]:
            i += 1
        nums[i-1], nums[j] = nums[j], nums[i-1]
        temp = nums[j+1:]
        temp.reverse()
        nums[j+1:] = temp
        return
```

Runtime: 63 ms, faster than 15.72% of Python3 online submissions for Next Permutation.

Memory Usage: 14.3 MB, less than 52.35% of Python3 online submissions for Next Permutation.

### 33. Search in Rotated Sorted Array 

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left < right:
            idx = (left + right) // 2
            if nums[idx] == target:
                return idx
            elif nums[idx] > target:
                if nums[idx] >= nums[left] and target < nums[left]:
                    left = idx + 1
                else:
                    right = idx - 1
            else:
                if nums[idx] <= nums[right] and target > nums[right]:
                    right = idx - 1
                else:
                    left = idx + 1
        if left == right and nums[left] == target:
            return left
        return -1
```

Runtime: 60 ms, faster than 17.29% of Python3 online submissions for Search in Rotated Sorted Array.

Memory Usage: 14.8 MB, less than 22.43% of Python3 online submissions for Search in Rotated Sorted Array.

### 34. Find First and Last Position of Element in Sorted Array 

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        left = 0
        right = len(nums) - 1
        while left <= right:
            idx = (left + right) // 2
            if nums[idx] < target:
                left = idx + 1
                continue
            elif nums[idx] > target:
                right = idx - 1
                continue
            else:
                left1 = left
                right1 = idx
                while left1 < right1:
                    idx1 = (left1 + right1) // 2  # floor
                    if nums[idx1] < target:
                        left1 = idx1 + 1
                    else:
                        right1 = idx1
                left2 = idx
                right2 = right
                while left2 < right2:
                    idx2 = (left2 + right2 + 1) // 2  #ceiling
                    if nums[idx2] > target:
                        right2 = idx2 - 1
                    else:
                        left2 = idx2
                return [right1, left2]
        return [-1, -1]
```

Runtime: 84 ms, faster than 76.54% of Python3 online submissions for Find First and Last Position of Element in Sorted Array.

Memory Usage: 15.5 MB, less than 50.89% of Python3 online submissions for Find First and Last Position of Element in Sorted Array.

### 36. Valid Sudoku 

**Method 1: build a list for each row, column, box, and find if there are repeated values.** 

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        for i in range(9):
            lst = board[i]
            for k in range(1,9):
                if lst[k] != "." and lst[k] in lst[:k]:
                    return False
            lst = [board[j][i] for j in range(9)]
            for k in range(1,9):
                if lst[k] != "." and lst[k] in lst[:k]:
                    return False
        for i in [0, 3, 6]:
            for j in [0, 3, 6]:
                lst = board[i][j: j+3] + board[i+1][j: j+3] + board[i+2][j: j+3]
                for k in range(1,9):
                    if lst[k] != "." and lst[k] in lst[:k]:
                        return False
        return True
```

Runtime: 109 ms, faster than 26.60% of Python3 online submissions for Valid Sudoku.

Memory Usage: 14.2 MB, less than 69.81% of Python3 online submissions for Valid Sudoku.

**Method 2: bitmask (inspired by others).**

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [0] * 9
        cols = [0] * 9
        boxes = [0] * 9
        for r in range(9):
            for c in range(9):
                p = board[r][c]
                if p == '.':
                    continue
                mask = 1 << (int(p) - 1)
                i = 3 * (r // 3)  + c // 3
                if rows[r] & mask or cols[c] & mask or boxes[i] & mask:
                    return False
                rows[r] |= mask
                cols[c] |= mask
                boxes[i] |= mask
        return True
```

Runtime: 96 ms, faster than 72.09% of Python3 online submissions for Valid Sudoku.

Memory Usage: 14.2 MB, less than 69.81% of Python3 online submissions for Valid Sudoku.

### 38. Count and Say 

```python
class Solution:
    def countAndSay(self, n: int) -> str:
        x = "1"
        for i in range(n-1):
            y = ""
            last = x[0]
            cnt = 0
            for now in x:
                if now != last:
                    y += str(cnt) + last
                    last = now
                    cnt = 1
                else:
                    cnt += 1
            x = y + str(cnt) + last
        return x    
```

Runtime: 36 ms, faster than 95.89% of Python3 online submissions for Count and Say.

Memory Usage: 14.4 MB, less than 22.54% of Python3 online submissions for Count and Say.

### 39. Combination Sum

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort(reverse = True)
        for i in range(len(candidates)):
            cnt = target // candidates[i]
            if target == candidates[i] * cnt:
                res.append([candidates[i]] * cnt)
            cnt = (target - 1) // candidates[i]  # minus 1 when divisible
            for j in range(1, cnt+1):
                res += [[candidates[i]] * j + k 
                        for k in Solution().combinationSum(
                            candidates[i+1:], target - candidates[i] * j)]
        return res
```

Runtime: 98 ms, faster than 45.50% of Python3 online submissions for Combination Sum.

Memory Usage: 14.4 MB, less than 51.63% of Python3 online submissions for Combination Sum.

### 40. Combination Sum II 

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        if sum(candidates) < target:
            return res
        if len(set(candidates)) == 1 and target % candidates[0] == 0:  # handle all-1 condition
            return [[candidates[0]] * (target // candidates[0])]
        candidates.sort(reverse = True)
        for i in range(len(candidates)):
            if target < candidates[i]:
                continue
            elif target == candidates[i]:
                res.append([candidates[i]])
            else:
                res += [[candidates[i]] + k for k in Solution().combinationSum2(candidates[i+1:], target - candidates[i])]
        uni = []  # unique items in res
        for i in res:
            if i not in uni:
                uni.append(i)
        return uni
```

Runtime: 80 ms, faster than 56.40% of Python3 online submissions for Combination Sum II.

Memory Usage: 14.4 MB, less than 49.73% of Python3 online submissions for Combination Sum II.

### 43. Multiply Strings 

```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        res = [0]
        n1 = len(num1)
        n2 = len(num2)
        lst1 = list(num1)
        lst2 = list(num2)
        lst1.reverse()
        lst2.reverse()
        for k in range(n1+n2-1):
            i = max(0, k-n2+1)
            digit = 0
            while i < n1 and k - i >= 0:
                digit += int(lst1[i]) * int(lst2[k-i])
                i += 1
            digit += res[-1]
            res[-1] = digit % 10
            res.append(digit // 10)
        res.reverse()
        output = ''.join([str(i) for i in res]).lstrip('0')
        return output if output else '0'
```

Runtime: 124 ms, faster than 43.89% of Python3 online submissions for Multiply Strings.

Memory Usage: 14.4 MB, less than 24.99% of Python3 online submissions for Multiply Strings.

### 45. Jump Game II 

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        res = 0
        now = 0
        n = len(nums)
        if n == 1:
            return 0
        elif nums[0] >= n-1:
            return 1
        while now < n-1:
            far = 0
            farid = 0
            for i in range(1, nums[now]+1):
                if i + nums[now+i] > far:
                    far = i + nums[now+i]
                    farid = i
            if now + far >= n-1:
                return res + 2
            else:
                res += 1
                now += farid
```

Runtime: 112 ms, faster than 98.55% of Python3 online submissions for Jump Game II.

Memory Usage: 15.1 MB, less than 72.39% of Python3 online submissions for Jump Game II.

### 46. Permutations 

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = [[nums[0]]]
        n = len(nums)
        i = 1
        while i < n:
            now = nums[i]
            for j in range(len(res)):
                temp = res.pop(0)
                for k in range(len(temp)+1):
                    res.append(temp[:k] + [now] + temp[k:])
            i += 1
        return res
```

Runtime: 40 ms, faster than 74.21% of Python3 online submissions for Permutations.

Memory Usage: 14.2 MB, less than 89.06% of Python3 online submissions for Permutations.

### 47. Permutations II 

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = [[]]
        while nums:
            now = nums.pop(0)
            i = 1  # find how many now's are there in list nums
            while nums and nums[0] == now:
                nums.pop(0)
                i += 1
            for j in range(len(res)):
                temp = res.pop(0)  # for each list in res
                tempres = []
                for k in range(len(temp)+1):
                    tempres.append(temp[:k] + [now] + temp[k:])  # insert the 1st now
                p = i - 1
                while p:  # insert other now's
                    p -= 1
                    for l in range(len(tempres)):
                        y = tempres.pop(0)
                        for m in range(y.index(now) + 1):
                            # insert prior to all inserted now's
                            tempres.append(y[:m] + [now] + y[m:])
                res += tempres
        return res
```

Runtime: 44 ms, faster than 99.77% of Python3 online submissions for Permutations II.

Memory Usage: 14.5 MB, less than 77.74% of Python3 online submissions for Permutations II.

### 48. Rotate Image 

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n//2):
            for j in range(i, n-i-1):
                temp, matrix[i][j] = matrix[i][j], matrix[n-j-1][i]
                temp, matrix[j][n-i-1] = matrix[j][n-i-1], temp
                temp, matrix[n-i-1][n-j-1] = matrix[n-i-1][n-j-1], temp
                matrix[n-j-1][i] = temp
```

Runtime: 36 ms, faster than 68.69% of Python3 online submissions for Rotate Image.

Memory Usage: 14.3 MB, less than 59.63% of Python3 online submissions for Rotate Image.

### 49. Group Anagrams 

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = {}
        for s in strs:
            slst = [0] * 26
            for i in list(s):
                n = ord(i) - 97
                slst[n] += 1
            slst = tuple(slst)
            res[slst] = res.get(slst, []) + [s]
        return list(res.values())
```

Runtime: 120 ms, faster than 43.58% of Python3 online submissions for Group Anagrams.

Memory Usage: 18.9 MB, less than 21.38% of Python3 online submissions for Group Anagrams.

### 50. Pow(x, n) 

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n < 0:
            return 1 / Solution().myPow(x, -n)
        res = 1
        now = x
        while n > 0:
            if n % 2:
                res = res * now
            n = n // 2
            now = now * now
        return res
```

Runtime: 32 ms, faster than 66.42% of Python3 online submissions for Pow(x, n).

Memory Usage: 14.2 MB, less than 50.06% of Python3 online submissions for Pow(x, n).

### 54. Spiral Matrix

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m = len(matrix)
        if m == 0:
            return []
        elif m == 1:
            return matrix[0]
        res = matrix.pop(0)
        n = len(res)
        if n == 0:
            return res
        elif n == 1:
            return res + [lst[0] for lst in matrix]
        for i in range(m-2):
            res.append(matrix[i].pop(n-1))
        res.extend(matrix.pop(m-2)[::-1])
        for i in range(m-2):
            res.append(matrix[m-3-i].pop(0))
        return res + self.spiralOrder(matrix)
```

Runtime: 70 ms, faster than 5.60% of Python3 online submissions for Spiral Matrix.

Memory Usage: 13.8 MB, less than 89.19% of Python3 online submissions for Spiral Matrix.

**Improvement in Code Simplicity: **https://leetcode.com/problems/spiral-matrix/discuss/20571/1-liner-in-Python-%2B-Ruby

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        return matrix and [*matrix.pop(0)] + self.spiralOrder([*zip(*matrix)][::-1])
```

### 55. Jump Game

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        # find 0 in reverse order
        target = len(nums) - 1
        res = True
        for i in range(target)[::-1]:
            if i + nums[i] >= target:
                target = i
                res = True
                continue
            res = False
        return res
```

Runtime: 480 ms, faster than 93.45% of Python3 online submissions for Jump Game.

Memory Usage: 15.3 MB, less than 25.85% of Python3 online submissions for Jump Game.

### 56. Merge Intervals

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals = sorted(intervals, key=lambda x:x[0])
        i = 1
        while i < len(intervals):
            if intervals[i-1][1] >= intervals[i][0]:
                temp = intervals.pop(i)
                intervals[i-1][1] = max(intervals[i-1][1], temp[1])
            else:
                i += 1
        return intervals
```

Runtime: 177 ms, faster than 66.52% of Python3 online submissions for Merge Intervals.

Memory Usage: 18.1 MB, less than 86.08% of Python3 online submissions for Merge Intervals.

### 57. Insert Interval

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        i = 0
        while i < len(intervals) and intervals[i][1] < newInterval[0]:
            i += 1
        if i == len(intervals):
            return intervals + [newInterval]
        if intervals[i][0] > newInterval[1]:
            return intervals[:i] + [newInterval] + intervals[i:]
        intervals[i][0] = min(intervals[i][0], newInterval[0])
        intervals[i][1] = max(intervals[i][1], newInterval[1])
        i += 1
        while i < len(intervals) and intervals[i-1][1] >= intervals[i][0]:
                temp = intervals.pop(i)
                intervals[i-1][1] = max(intervals[i-1][1], temp[1])
        return intervals
```

Runtime: 92 ms, faster than 74.75% of Python3 online submissions for Insert Interval.

Memory Usage: 17.3 MB, less than 65.75% of Python3 online submissions for Insert Interval.

### 59. Spiral Matrix II

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        return [[*range(1, n+1)]] + [*zip(*self.generateTwoEdges(n-1, n+1)[::-1])]
    def generateTwoEdges(self, length, start):
        if length == 0:
            return []
        res = self.generateTwoEdges(length-1, start+2*length)
        res = [[*range(start+length, start+2*length)]] + [*zip(*res[::-1])]
        return [[*range(start, start+length)]] + [*zip(*res[::-1])]
```

Runtime: 66 ms, faster than 9.72% of Python3 online submissions for Spiral Matrix II.

Memory Usage: 13.9 MB, less than 89.61% of Python3 online submissions for Spiral Matrix II.

### 60. Permutation Sequence

```python
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        fac = [1]
        nlist = ['1']
        for i in range(1, n):
            fac.append(fac[-1] * i)
            nlist.append(str(i + 1))
        return ''.join(self.getNext(n, k-1, nlist, fac))
    
    def getNext(self, n, k, nlist, fac):
        if n == 0:
            return []
        now = nlist.pop(k // fac[n - 1])
        return [now] + self.getNext(n - 1, k % fac[n - 1], nlist, fac)
```

Runtime: 32 ms, faster than 90.83% of Python3 online submissions for Permutation Sequence.

Memory Usage: 13.8 MB, less than 86.19% of Python3 online submissions for Permutation Sequence.

### 61. Rotate List

```python
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head:
            return head
        now = head
        length = 1
        while now.next:
            now = now.next
            length += 1
        cnt = k % length
        if cnt == 0:
            return head
        now.next = head
        now = head
        cnt = length - 1 - cnt
        while cnt > 0:
            now = now.next
            cnt -= 1
        head = now.next
        now.next = None
        return head
```

Runtime: 63 ms, faster than 29.24% of Python3 online submissions for Rotate List.

Memory Usage: 13.9 MB, less than 85.83% of Python3 online submissions for Rotate List.

### 62. Unique Paths

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        if min(m, n) == 1:
            return 1
        lst = [1] * m
        for i in range(n-1):
            for j in range(1, m):
                lst[j] = lst[j] + lst[j-1]
        return lst[-1]
```

Runtime: 56 ms, faster than 23.46% of Python3 online submissions for Unique Paths.

Memory Usage: 13.8 MB, less than 83.86% of Python3 online submissions for Unique Paths.

### 63. Unique Paths II

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        n = len(obstacleGrid)
        m = len(obstacleGrid[0])
        if min(m, n) == 1:
            if sum([sum(i) for i in obstacleGrid]) > 0:
                return 0
            else:
                return 1
        lst = [0] * (m+1)
        if obstacleGrid[0][0] == 1:
            return 0
        lst[1] = 1
        for i in range(n):
            for j in range(m):
                lst[j+1] = (lst[j+1] + lst[j]) * (1 - obstacleGrid[i][j])
        return lst[-1]
```

Runtime: 69 ms, faster than 36.41% of Python3 online submissions for Unique Paths II.

Memory Usage: 13.8 MB, less than 98.18% of Python3 online submissions for Unique Paths II.

### 64. Minimum Path Sum

```
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        lst = [sum(grid[0][:j]) for j in range(1, m+1)]
        for i in range(1, n):
            lst[0] += grid[i][0]
            for j in range(1, m):
                lst[j] = min(lst[j-1], lst[j]) + grid[i][j]
        return lst[-1]
```

Runtime: 84 ms, faster than 99.92% of Python3 online submissions for Minimum Path Sum.

Memory Usage: 14.7 MB, less than 93.54% of Python3 online submissions for Minimum Path Sum.

### 71. Simplify Path

```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        res = []
        for i in path.split('/'):
            if i == '..':
                res = res[:-1]
            elif i and i != '.':
                res.append(i)
        return '/' + '/'.join(res)
```

Runtime: 50 ms, faster than 44.80% of Python3 online submissions for Simplify Path.

Memory Usage: 13.9 MB, less than 87.69% of Python3 online submissions for Simplify Path.

### 73. Set Matrix Zeroes

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        i = j = 0
        m, n = len(matrix), len(matrix[0])
        row_with_0 = []
        col_with_0 = []
        while i < m:
            j = 0
            while j < n:
                if matrix[i][j]: 
                    j += 1
                    continue
                row_with_0.append(i)
                break
            i += 1
        i = j = 0
        while j < n:
            i = 0
            while i < m:
                if matrix[i][j]: 
                    i += 1
                    continue
                col_with_0.append(j)
                break
            j += 1
        for i in row_with_0:
            for j in range(n):
                matrix[i][j] = 0
        for j in col_with_0:
            for i in range(m):
                matrix[i][j] = 0
```

Runtime: 279 ms, faster than 6.67% of Python3 online submissions for Set Matrix Zeroes.

Memory Usage: 14.7 MB, less than 69.47% of Python3 online submissions for Set Matrix Zeroes.



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

### 177. Nth Highest Salary

```mysql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  DECLARE M INT;
  SET M = N-1;
  RETURN (
      # Write your MySQL query statement below.
      SELECT DISTINCT salary 
      FROM Employee 
      ORDER BY salary DESC
      LIMIT 1 OFFSET M
  );
END
```

### 178. Rank Scores

```mysql
SELECT 
    score, 
    DENSE_RANK() OVER (
        ORDER BY score DESC
    ) AS 'rank'
FROM 
    Scores;
```

### 180. Consecutive Numbers

```mysql
# Write your MySQL query statement below
SELECT DISTINCT num AS ConsecutiveNums FROM Logs
WHERE (id-1, num) IN (SELECT * FROM Logs) AND (id+1, num) IN (SELECT * FROM Logs) 
```

### 181. Employees Earning More Than Their Managers

```mysql
# Write your MySQL query statement below
SELECT a.name AS Employee FROM Employee a
INNER JOIN Employee b ON a.managerId = b.id
WHERE a.salary > b.salary
```

### 182. Duplicate Emails

```mysql
# Write your MySQL query statement below
SELECT email AS Email FROM Person
GROUP BY email
HAVING COUNT(Email) > 1
```

### 183. Customers Who Never Order

```mysql
# Write your MySQL query statement below
SELECT name AS Customers FROM Customers
WHERE id NOT IN (SELECT DISTINCT customerId FROM Orders)
```

### 184. Department Highest Salary

```mysql
# Write your MySQL query statement below
SELECT a.name AS Department, b.name AS Employee, b.salary AS Salary FROM Department a
INNER JOIN 
(SELECT 
    name, 
    salary, 
    MAX(salary) OVER (PARTITION BY departmentId) AS max_salary, 
    departmentId 
FROM Employee) b
ON a.id = b.departmentId
WHERE salary = max_salary
```

### 185. Department Top Three Salaries

```mysql
# Write your MySQL query statement below
SELECT a.name AS Department, b.name AS Employee, b.salary AS Salary FROM Department a
INNER JOIN 
(SELECT 
    name, 
    salary, 
    DENSE_RANK() OVER (PARTITION BY departmentId ORDER BY salary DESC) AS drank, 
    departmentId 
FROM Employee) b
ON a.id = b.departmentId
WHERE drank < 4
```

### 196. Delete Duplicate Emails

```mysql
# Write your MySQL query statement below
DELETE a FROM Person a, Person b
WHERE a.email = b.email AND a.id > b.id
```

### 197. Rising Temperature

```mysql
# Write your MySQL query statement below
SELECT w1.id FROM Weather w1, Weather w2
WHERE DATEDIFF(w1.recordDate, w2.recordDate) = 1 AND w1.temperature > w2.temperature
```

### 262. Trips and Users

```mysql
# Write your MySQL query statement below
SELECT 
    request_at AS Day, 
    ROUND(COUNT(CASE WHEN status <> 'completed' THEN 1 ELSE NULL END) / COUNT(*), 2) AS 'Cancellation Rate'
FROM Trips
WHERE 
    request_at BETWEEN '2013-10-01' AND '2013-10-03'
AND
    client_id IN (SELECT users_id FROM Users WHERE role = 'client' AND banned = 'No')
AND
    driver_id IN (SELECT users_id FROM Users WHERE role = 'driver' AND banned = 'No')
GROUP BY request_at
```

### 511. Game Play Analysis I

```mysql
# Write your MySQL query statement below
SELECT player_id, MIN(event_date) AS first_login
FROM Activity
GROUP BY player_id
```

### 512. Game Play Analysis II

```mysql
# Write your MySQL query statement below
SELECT a.player_id, b.device_id
FROM (
    SELECT player_id, MIN(event_date) AS first_login
    FROM Activity
    GROUP BY player_id
) a LEFT JOIN Activity b
ON a.player_id = b.player_id AND a.first_login = b.event_date
```

### 534. Game Play Analysis III

```mysql
# Write your MySQL query statement below
SELECT 
    player_id, 
    event_date, 
    SUM(games_played) OVER (
        PARTITION BY player_id 
        ORDER BY event_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS games_played_so_far
FROM Activity
```

### 550. Game Play Analysis IV

```mysql
# Write your MySQL query statement below
SELECT ROUND((
    SELECT COUNT(DISTINCT player_id) FROM Activity
    WHERE (player_id, DATE_SUB(event_date, INTERVAL 1 DAY)) IN (
        SELECT player_id, MIN(event_date)
        FROM Activity
        GROUP BY player_id
    )
) / (
    SELECT COUNT(DISTINCT player_id) FROM Activity
), 2) AS fraction
```

### 569. Median Employee Salary

```mysql
# Write your MySQL query statement below
WITH temp AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (PARTITION BY company ORDER BY salary) AS ranking,
        COUNT(*) OVER (PARTITION BY company) AS counts
    FROM Employee
)
SELECT e.id, company, salary FROM Employee e
INNER JOIN temp t
ON e.id = t.id
WHERE ranking >= counts / 2 AND ranking <= counts / 2 + 1
```

### 570. Managers with at Least 5 Direct Reports

```mysql
# Write your MySQL query statement below
SELECT name FROM (
    SELECT managerId FROM Employee GROUP BY managerId HAVING COUNT(*) >= 5
) a INNER JOIN Employee b
ON a.managerId = b.id
```

### 571. Find Median Given Frequency of Numbers

```mysql
# Write your MySQL query statement below
WITH temp AS (
    SELECT num, SUM(frequency) OVER (ORDER BY num) AS freq_up_to FROM Numbers
)
SELECT ROUND(
    ((SELECT num FROM temp 
     WHERE freq_up_to >= (SELECT SUM(frequency) FROM Numbers) / 2
     LIMIT 1) + 
    (SELECT num FROM temp 
     WHERE freq_up_to >= ((SELECT SUM(frequency) FROM Numbers) + 1) / 2 
     LIMIT 1)) / 2, 1
) AS median
```

### 574. Winning Candidate

```mysql
SELECT name FROM Candidate
WHERE id = (
    SELECT candidateId FROM Vote
    GROUP BY candidateId
    ORDER BY COUNT(*) DESC
    LIMIT 1
)
```

### 577. Employee Bonus

```mysql
SELECT name, bonus FROM Employee e LEFT JOIN Bonus b
ON e.empId = b.empId
WHERE IFNULL(bonus, 0) < 1000
```

### 578. Get Highest Answer Rate Question

```mysql
SELECT question_id AS survey_log FROM (
    SELECT 
        question_id, 
        SUM(IF(action = 'answer', 1, 0)) / SUM(IF(action = 'show', 1, 0)) as ans_rate
    FROM SurveyLog
    GROUP BY question_id
) temp
ORDER BY ans_rate DESC, question_id ASC
LIMIT 1
```

### 579. Find Cumulative Salary of an Employee

```mysql
SELECT 
    E1.id, 
    E1.month, 
    E1.salary + IFNULL(E2.salary, 0) + IFNULL(E3.salary, 0) AS Salary
FROM Employee E1
LEFT JOIN Employee E2
ON E1.id = E2.id AND E1.month = E2.month + 1
LEFT JOIN Employee E3
ON E1.id = E3.id AND E1.month = E3.month + 2
INNER JOIN (
    SELECT id, month, MAX(month) OVER (PARTITION BY id) AS max_month
    FROM Employee
) E4
ON E1.id = E4.id AND E1.month = E4.month
WHERE E1.month < E4.max_month
ORDER BY E1.id ASC, E1.month DESC
```

### 580. Count Student Number in Departments

```mysql
SELECT dept_name, IFNULL(student_number, 0) AS student_number
FROM Department D
LEFT JOIN (
    SELECT dept_id, COUNT(*) AS student_number 
    FROM Student
    GROUP BY dept_id
) S
ON D.dept_id = S.dept_id
ORDER BY student_number DESC, dept_name
```

### 584. Find Customer Referee

```mysql
SELECT name FROM Customer WHERE IFNULL(referee_id, 0) <> 2
```

### 585. Investments in 2016

```mysql
SELECT ROUND(SUM(tiv_2016), 2) AS tiv_2016 FROM Insurance I
INNER JOIN (
    SELECT tiv_2015, COUNT(*) AS cnt2015 FROM Insurance GROUP BY tiv_2015
) C1
ON I.tiv_2015 = C1.tiv_2015
INNER JOIN (
    SELECT lat, lon, COUNT(*) AS cntcity FROM Insurance GROUP BY lat, lon
) C2
ON I.lat = C2.lat AND I.lon = C2.lon
WHERE cnt2015 > 1 AND cntcity = 1
```

### 586. Customer Placing the Largest Number of Orders

```mysql
SELECT customer_number FROM Orders
GROUP BY customer_number
ORDER BY COUNT(*) DESC
LIMIT 1
```

### 595. Big Countries

```mysql
SELECT name, population, area FROM World
WHERE area >= 3000000 OR population >= 25000000
```

### 596. Classes More Than 5 Students

```mysql
SELECT class FROM Courses GROUP BY class HAVING COUNT(*) >= 5 
```

### 597. Friend Requests I: Overall Acceptance Rate

```mysql
SELECT ROUND(IFNULL(
    (SELECT COUNT(DISTINCT requester_id, accepter_id) FROM RequestAccepted AS A) /
    (SELECT COUNT(DISTINCT sender_id, send_to_id) FROM FriendRequest AS B), 0), 2) 
    AS accept_rate
```

### 601. Human Traffic of Stadium

```mysql
WITH temp AS (
    SELECT S1.id AS id1, S2.id AS id2, S3.id AS id3
    FROM Stadium S1, Stadium S2, Stadium S3
    WHERE S1.id = S2.id - 1 AND S1.id = S3.id - 2 AND 
        S1.people >= 100 AND S2.people >= 100 AND S3.people >= 100
)
SELECT * FROM Stadium WHERE id IN (
    SELECT id1 FROM temp
    UNION
    SELECT id2 FROM temp
    UNION
    SELECT id3 FROM temp
)
```

### 1126. Active Businesses

```mysql
WITH Avg_act AS (
    SELECT event_type, AVG(occurences) AS avg_act
    FROM Events GROUP BY event_type
)

SELECT business_id FROM Events e
LEFT JOIN Avg_act a ON e.event_type = a.event_type
WHERE occurences > avg_act
GROUP BY business_id HAVING COUNT(*) > 1
```

