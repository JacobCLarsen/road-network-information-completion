# Shared fixtures for the ML tests

import pytest

from src.ml import save_run


# Replaces the functions in save_run with stand-ins that write down what they get instead of using the database, so
# the run scripts can be tested without a database. A test uses it by listing fake_database as a parameter.
# Returns a dict of lists that fill up while the script runs:
# - "splits":   the masks of every save_split call
# - "started":  (architecture, seed, split_id, hyperparameters) of every start_run call
# - "finished": (run_id, results, history, best_epoch) of every finish_run call
# - "failed":   the run_id of every fail_run call
@pytest.fixture
def fake_database(monkeypatch):
    recorded = {"splits": [], "started": [], "finished": [], "failed": []}

    def fake_save_split(graph, masks):
        recorded["splits"].append(masks)
        return 7

    def fake_start_run(architecture, seed, split_id, hyperparameters):
        recorded["started"].append((architecture, seed, split_id, hyperparameters))
        # A different run_id for every run: 100, 101, ...
        return 100 + len(recorded["started"]) - 1

    def fake_finish_run(run_id, graph, predictions, results, history=None, best_epoch=None):
        recorded["finished"].append((run_id, results, history, best_epoch))

    def fake_fail_run(run_id):
        recorded["failed"].append(run_id)

    monkeypatch.setattr(save_run, "save_split", fake_save_split)
    monkeypatch.setattr(save_run, "start_run", fake_start_run)
    monkeypatch.setattr(save_run, "finish_run", fake_finish_run)
    monkeypatch.setattr(save_run, "fail_run", fake_fail_run)
    return recorded
