from safety_audit.cli import _evaluate


def test_audit_evaluates_registered_environment():
    from ccpl import make_ccpl

    agent = make_ccpl(6, 5, seed=3, pretrain_steps=0, batch_size=4)
    rows = _evaluate(agent, "standard", 2, episodes=1, seeds=[3],
                     max_steps=4, threshold=3.0)
    assert len(rows) == 1
    assert 0.0 <= rows[0]["csr"] <= 1.0
