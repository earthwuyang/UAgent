from app.optim.meta_optimizer import MetaOptimizer, MetaOptConfig

def test_meta_optimizer_skip_when_disabled():
    meta = MetaOptimizer(gepa=None, cfg=MetaOptConfig(enabled=False))
    program, info = meta.maybe_improve_program(0.1, None, [], [], lambda *_: 0.0, {})
    assert program is None and info is None


def test_meta_optimizer_threshold_gate():
    meta = MetaOptimizer(gepa=None, cfg=MetaOptConfig(trigger_threshold=0.5, enabled=True))
    program, info = meta.maybe_improve_program(0.8, None, [], [], lambda *_: 0.0, {})
    assert program is None and info is None
