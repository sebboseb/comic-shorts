import anthropic

from pipeline.stage2_understand import BATCH_STATE_FILE, _get_or_create_batch


def _not_found():
    return anthropic.NotFoundError.__new__(anthropic.NotFoundError)


class _FakeBatches:
    def __init__(self, known=None):
        self.known = known or {}
        self.created = []

    def retrieve(self, batch_id):
        if batch_id not in self.known:
            raise _not_found()
        return self.known[batch_id]

    def create(self, requests):
        self.created.append(requests)
        return type("B", (), {"id": "batch_new",
                              "processing_status": "in_progress"})()


class _FakeClient:
    def __init__(self, batches):
        self.messages = type("M", (), {"batches": batches})()


def test_fresh_run_submits_and_persists_id(tmp_path):
    batches = _FakeBatches()
    batch = _get_or_create_batch(_FakeClient(batches), ["req"], tmp_path)
    assert batch.id == "batch_new"
    assert len(batches.created) == 1
    assert (tmp_path / BATCH_STATE_FILE).read_text() == "batch_new"


def test_known_id_resumes_without_submitting(tmp_path):
    existing = type("B", (), {"id": "batch_old",
                              "processing_status": "ended"})()
    batches = _FakeBatches(known={"batch_old": existing})
    (tmp_path / BATCH_STATE_FILE).write_text("batch_old")
    batch = _get_or_create_batch(_FakeClient(batches), ["req"], tmp_path)
    assert batch is existing
    assert batches.created == []
    assert (tmp_path / BATCH_STATE_FILE).read_text() == "batch_old"


def test_stale_id_falls_through_to_fresh_submit(tmp_path):
    batches = _FakeBatches()
    (tmp_path / BATCH_STATE_FILE).write_text("batch_gone")
    batch = _get_or_create_batch(_FakeClient(batches), ["req"], tmp_path)
    assert batch.id == "batch_new"
    assert len(batches.created) == 1
    assert (tmp_path / BATCH_STATE_FILE).read_text() == "batch_new"
