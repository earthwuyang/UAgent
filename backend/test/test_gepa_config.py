from app.optim.gepa_optimizer import GEPAConfig

def test_gepa_config_defaults():
    cfg = GEPAConfig()
    assert cfg.max_metric_calls == 150
    assert cfg.min_delta == 0.02
