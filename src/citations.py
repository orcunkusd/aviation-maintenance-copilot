"""Parse citation markers such as [2], [1][3] and [2, 4] out of an answer."""
import re


def cited_numbers(text):
    nums = set()
    for group in re.findall(r"\[([\d,\s]+)\]", text or ""):
        nums.update(int(x) for x in re.findall(r"\d+", group))
    return sorted(nums)
