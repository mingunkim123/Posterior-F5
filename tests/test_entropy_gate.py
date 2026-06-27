from f5_tts.posterior.gating import blank_to_text_weight, combine_gate, entropy_to_text_weight


def test_entropy_gate_prefers_text_for_low_entropy():
    low = entropy_to_text_weight(0.1, threshold=1.0)
    high = entropy_to_text_weight(2.0, threshold=1.0)

    assert low > high
    assert low > 0.5
    assert high < 0.5


def test_blank_gate_prefers_ssl_for_high_blank_mass():
    low_blank = blank_to_text_weight(0.1, threshold=0.5)
    high_blank = blank_to_text_weight(0.9, threshold=0.5)

    assert low_blank > high_blank


def test_combined_gate_multiplies_signals():
    combined = combine_gate(entropy=0.1, blank_prob=0.1)
    entropy_only = entropy_to_text_weight(0.1)

    assert 0.0 < combined < entropy_only


def test_combined_gate_broadcasts_scalar_to_sequence():
    combined = combine_gate(entropy=[0.1, 2.0], blank_prob=0.1)

    assert len(combined) == 2
    assert combined[0] > combined[1]
