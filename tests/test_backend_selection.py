import batch
from pipeline import PipelineConfig


def test_build_predictors_sam2_uses_setup(monkeypatch):
    seen = {}
    def fake_build(**kw):
        seen[kw["kind"]] = kw
        return (f"SAM2_{kw['kind']}", None)
    monkeypatch.setattr(batch.setup, "build_predictor", fake_build)
    img, vid = batch.build_predictors(PipelineConfig(backend="sam2"))
    assert img == "SAM2_image" and vid == "SAM2_video"
    assert set(seen) == {"image", "video"}


def test_build_predictors_sam3_uses_adapters(monkeypatch):
    import sam2_utils.sam3_backend as sb
    monkeypatch.setattr(sb, "Sam3ImagePredictor", lambda ckpt: ("SAM3_image", ckpt))
    monkeypatch.setattr(sb, "Sam3VideoPredictor", lambda ckpt: ("SAM3_video", ckpt))
    img, vid = batch.build_predictors(PipelineConfig(backend="sam3", sam3_checkpoint="/x"))
    assert img == ("SAM3_image", "/x") and vid == ("SAM3_video", "/x")


def test_build_predictors_sam3_defaults_checkpoint(monkeypatch):
    import sam2_utils.sam3_backend as sb
    monkeypatch.setattr(sb, "Sam3ImagePredictor", lambda ckpt: ("i", ckpt))
    monkeypatch.setattr(sb, "Sam3VideoPredictor", lambda ckpt: ("v", ckpt))
    img, _ = batch.build_predictors(PipelineConfig(backend="sam3"))
    assert img[1] == sb.DEFAULT_CHECKPOINT_DIR
