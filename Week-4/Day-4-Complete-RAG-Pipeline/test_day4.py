import unittest
from day4_demo import POLICIES, QUESTION, retrieve_offline, build_context, answer_offline
class Day4Tests(unittest.TestCase):
    def test_superseded_and_canada_excluded(self):
        ids={p.id for p in retrieve_offline(QUESTION)}
        self.assertEqual(ids,{'US-REMOTE-2026','US-BENEFITS-2026'})
    def test_context_has_sources(self):
        self.assertIn('[SOURCE US-REMOTE-2026]',build_context(retrieve_offline(QUESTION)))
    def test_grounded_answer(self):
        answer=answer_offline(retrieve_offline(QUESTION))
        self.assertIn('HR approval',answer)
        self.assertIn('[US-BENEFITS-2026]',answer)
    def test_insufficient_evidence(self):
        self.assertIn('Insufficient',answer_offline([POLICIES[0]]))
if __name__=='__main__': unittest.main()
