import asyncio
import itertools
from types import SimpleNamespace

import torch

from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
from verl.experimental.reward_loop.reward_manager.remote import RemoteRewardManager


def test_remote_reward_manager_reuses_dapo_run_single():
    assert issubclass(RemoteRewardManager, DAPORewardManager)
    assert RemoteRewardManager.run_single is DAPORewardManager.run_single


def test_dapo_reward_manager_uses_remote_scorer_and_keeps_overlong_penalty():
    class Tokenizer:
        def decode(self, _token_ids, skip_special_tokens):
            assert skip_special_tokens is True
            return "Reasoning\nAnswer: 2"

    class RemoteMethod:
        def remote(self, **kwargs):
            async def result():
                assert kwargs["data_source"] == "rlvr"
                assert kwargs["ground_truth"] == "2"
                return {"score": 1.0, "acc": True}

            return result()

    class RewardWorker:
        compute_score = RemoteMethod()

    class ImmediateLoop:
        def run_in_executor(self, _executor, function):
            async def result():
                return function()

            return result()

    data_item = SimpleNamespace(
        batch={
            "responses": torch.tensor([1, 2, 3, 4, 5, 6]),
            "attention_mask": torch.ones(6, dtype=torch.long),
        },
        non_tensor_batch={
            "data_source": "rlvr",
            "reward_model": {"ground_truth": "2"},
        },
    )

    class SingleItemData:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return data_item

    manager = DAPORewardManager.__new__(DAPORewardManager)
    manager.tokenizer = Tokenizer()
    manager.reward_router_address = None
    manager.reward_model_tokenizer = None
    manager.is_async_reward_score = False
    manager.loop = ImmediateLoop()
    manager.reward_worker_pool = itertools.cycle([RewardWorker()])
    manager.overlong_buffer_cfg = SimpleNamespace(enable=True, len=2, penalty_factor=1.0, log=True)
    manager.max_resp_len = 6

    result = asyncio.run(manager.run_single(SingleItemData()))

    assert result["reward_score"] == 0.0
    assert result["reward_extra_info"] == {
        "score": 1.0,
        "acc": True,
        "overlong_reward": -1.0,
        "overlong": True,
    }
