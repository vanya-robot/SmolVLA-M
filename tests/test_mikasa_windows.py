from smolvla_data.mikasa_windows import make_episode_split, make_windows


def test_episode_split_is_deterministic_and_disjoint():
    first = make_episode_split(250, seed=123, train_fraction=0.7, validation_fraction=0.15)
    second = make_episode_split(250, seed=123, train_fraction=0.7, validation_fraction=0.15)
    assert first == second
    groups = [set(first.train), set(first.validation), set(first.test)]
    assert sum(map(len, groups)) == 250
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_windows_never_cross_episode_and_mask_tail():
    windows = make_windows([4, 7], [1], horizon=5, pad_to_horizon=True)
    assert len(windows) == 7
    assert windows[0].global_start == 4
    assert windows[0].valid_length == 5
    assert windows[-1].start_in_episode == 6
    assert windows[-1].global_start == 10
    assert windows[-1].valid_length == 1


def test_full_window_mode_drops_short_tails():
    windows = make_windows([4, 7], [1], horizon=5, pad_to_horizon=False)
    assert [window.start_in_episode for window in windows] == [0, 1, 2]
    assert all(window.valid_length == 5 for window in windows)
