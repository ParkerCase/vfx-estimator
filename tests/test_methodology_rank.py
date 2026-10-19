"""Methodology suggestions are ordered by fit, with one best match."""

from vfx_estimator.estimate.methodology_rank import rank_methodology_suggestions


SIMPLE = [
    {
        "label": "Hero Full CG",
        "methodology": "Full CG hero creature with complex simulation and lighting.",
        "departments": ["lighting"],
    },
    {
        "label": "2D Composite Only",
        "methodology": "Plate-based 2D composite using roto and paint.",
        "departments": ["compositing"],
    },
]

HERO = [
    {
        "label": "2D Composite Only",
        "methodology": "Plate-based 2D composite using roto and paint.",
    },
    {
        "label": "Hero Destruction",
        "methodology": "Full CG destruction simulation of the practical city.",
    },
]


def test_simple_shot_ranks_lighter_methodology_first():
    ranked = rank_methodology_suggestions("Paint out a wire.", SIMPLE)
    assert ranked[0]["label"] == "2D Composite Only"
    assert ranked[0]["is_best_match"] is True
    assert ranked[1]["is_best_match"] is False
    assert ranked[0]["score"] > ranked[1]["score"]
    assert all(0.0 <= item["score"] <= 1.0 for item in ranked)
    assert ranked[1]["departments"] == ["lighting"]


def test_demanding_shot_does_not_bury_hero_option():
    description = (
        "Hero destruction of the intricate city. Difficult integration, "
        "challenging camera, complex falling debris."
    )
    ranked = rank_methodology_suggestions(description, HERO)
    assert ranked[0]["label"] == "Hero Destruction"
    assert ranked[0]["is_best_match"] is True


def test_keyword_overlap_prefers_the_matching_approach():
    options = [
        {
            "label": "2D Paint",
            "methodology": "Paint out a small rig and grain match.",
        },
        {
            "label": "Crowd and Tree",
            "methodology": "CG crowd in the courtyard plus a replacement tree.",
        },
    ]
    ranked = rank_methodology_suggestions(
        "Courtyard wide shot with a CG crowd and a replacement tree.",
        options,
    )
    assert ranked[0]["label"] == "Crowd and Tree"
