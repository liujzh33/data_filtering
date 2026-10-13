"""机器人数据集七阶段质检流水线。

只检测和记录, 不删任何数据。总览见 pipeline.md。

    from check_pipeline.pipeline import run_episode, DropBudget
    from check_pipeline.manifest import EpisodeAssets
"""

__all__ = ["errors", "intervals", "manifest", "pipeline", "probe", "signals"]
