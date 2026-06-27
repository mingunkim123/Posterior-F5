import json

import pytest

from f5_tts.posterior.schema import Hypothesis, PosteriorTokenMap, PosteriorUtterance, TopKPosterior


def test_hypothesis_round_trip_through_json():
    hypothesis = Hypothesis(
        text="hello world",
        posterior=0.75,
        score=-1.25,
        tokens=["hello", "world"],
        token_ids=[4, 8],
        metadata={"source": "unit-test"},
    )

    payload = json.loads(json.dumps(hypothesis.to_dict()))
    restored = Hypothesis.from_dict(payload)

    assert restored == hypothesis


def test_posterior_utterance_round_trip_with_nested_fields():
    utterance = PosteriorUtterance(
        utterance_id="utt-001",
        audio_path="audio/ref.wav",
        sample_rate=24000,
        asr_source="ctc-test",
        one_best="hello world",
        nbest=[
            Hypothesis(text="hello world", posterior=0.7),
            Hypothesis(text="yellow world", posterior=0.3),
        ],
        frame_posteriors=TopKPosterior(
            token_ids=[[1, 0], [2, 0]],
            probs=[[0.8, 0.2], [0.6, 0.4]],
            num_frames=2,
            top_k=2,
            frame_rate=50.0,
            blank_id=0,
        ),
        token_map=PosteriorTokenMap(
            source="toy-ctc",
            tokens=["<blank>", "h", "e"],
            token_to_id={"<blank>": 0, "h": 1, "e": 2},
            blank_id=0,
            filler_id=0,
            unk_id=3,
        ),
        expected_ref_len=10.5,
        mean_entropy=0.42,
        duration_sec=1.2,
        language="en",
        metadata={"split": "dev"},
    )

    payload = json.loads(json.dumps(utterance.to_dict()))
    restored = PosteriorUtterance.from_dict(payload)

    assert restored == utterance
    assert restored.nbest[1].text == "yellow world"
    assert restored.frame_posteriors.blank_id == 0
    assert restored.token_map.f5_vocab_offset == 1


def test_optional_nested_fields_can_be_absent():
    utterance = PosteriorUtterance.from_dict(
        {
            "utterance_id": "utt-002",
            "audio_path": "audio/ref.wav",
        }
    )

    assert utterance.frame_posteriors is None
    assert utterance.token_map is None
    assert utterance.nbest == []


def test_missing_required_fields_raise_key_error():
    with pytest.raises(KeyError):
        Hypothesis.from_dict({})

    with pytest.raises(KeyError):
        PosteriorUtterance.from_dict({"utterance_id": "utt-003"})

