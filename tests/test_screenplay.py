from vfx_estimator.screenplay.scene_match import index_screenplay_plaintext, match_scenes, parse_user_scene_number


def test_scene_number():
    assert parse_user_scene_number("Scene 2 INT church") == 2


def test_match():
    raw = "EXT. PARK - DAY\nWind.\nINT. CHURCH - NIGHT\nHal enters.\n"
    scenes = index_screenplay_plaintext(raw)
    hits = match_scenes("Scene 2 int church", scenes, top_k=1)
    assert hits[0][0].order_index == 2
