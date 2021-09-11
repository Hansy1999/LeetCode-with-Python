# Leetcode

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

