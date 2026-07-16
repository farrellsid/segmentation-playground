from pipeline import PipelineConfig


def test_multimask_generous_defaults_false():
    assert PipelineConfig().multimask_generous is False
