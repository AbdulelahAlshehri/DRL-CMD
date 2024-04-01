from dataclasses import dataclass, field, InitVar
from typing import List


@dataclass
class GroupSMARTS:
    """Wrapper for a SMARTS representation of a group.

    Attributes:
        raw_smarts: Unmodified SMARTS string.
        smart_str: SMART string modified with exclusions.
        group_num: The group number.
        weight: The molecular weight of group.
        values: List of all SMARTS strings corresponding to the group.
    """
    raw_smarts: List[List[str]]
    smart_str: List[List[str]]
    group_num: int
    weight: float

    @property
    def values(self) -> List[str]:
        results = []
        for smart in self.no_excl:
            smart_w_excl = smart[:-1]  # remove last bracket
            for e in self.excl:
                smart_w_excl = smart + ";!$([" + e + "])"
            smart_w_excl += "]"  # add in last bracket"
            results.append(smart)
