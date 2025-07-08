import random

from typing import Optional

from vllm.config import VllmConfig


def _adjust_random_nums_by_max(nums: list[int], max_num: int):
    exceed_len = sum([max(seq_len - max_num, 0) for seq_len in nums])
    for i in range(len(nums)):
        if nums[i] > max_num:
            nums[i] = max_num

        if exceed_len > 0 and nums[i] < max_num:
            delta = min(max_num - nums[i], exceed_len)
            nums[i] += delta
            exceed_len -= delta

    return nums


def _create_random_intergers_by_sum(
    sum_value: int, num: int, start: int = 1
) -> list[int]:
    """
    generate (num - 1) points from range [0, sum_value - start * num],
    so the mean value of intervals between adjacent points is (sum_value / num - start).
    And we add start later to get list with mean value of avg_value.
    """
    if num == 0:
        return []
    partition_length = sum_value - start * num
    partitions = [random.randint(0, partition_length) for _ in range(num - 1)]
    partitions.sort()
    partitions = [0] + partitions + [partition_length]
    samples = [partitions[i + 1] - partitions[i] for i in range(len(partitions) - 1)]
    nums = [sample + start for sample in samples]
    return nums

def _create_random_integers_by_avg(
    avg_seq_len: int, batch: int, max_seq_len: Optional[int] = None
) -> list[int]:
    seq_lens = _create_random_intergers_by_sum(avg_seq_len * batch, batch)
    # generate random intergers by average only ensure the average value,
    # i.e. the sum of intergers is (avg_seq_len x batch), however,
    # there may be some intergers larger then max_seq_len, there we need to
    # adjust seq_lens by max_seq_len
    if max_seq_len is not None:
        assert avg_seq_len <= max_seq_len
        seq_lens = _adjust_random_nums_by_max(seq_lens, max_seq_len)

    return seq_lens


def create_seq_lens(
    batch: int,
    avg_seq_len: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    config: Optional[VllmConfig] = None,
) -> list[int]:
    if max_seq_len is None:
        assert config is not None
        max_seq_len = config.model_config.max_model_len
    seq_lens = _create_random_integers_by_avg(avg_seq_len, batch, max_seq_len)
    return seq_lens
