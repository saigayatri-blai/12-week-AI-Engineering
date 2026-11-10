import unittest
from day2_demo import load_text, POLICY, fixed_size_chunks, section_chunks, has_complete_evidence
class Day2Tests(unittest.TestCase):
    def test_loader(self): self.assertIn('12 months',load_text(POLICY))
    def test_bad_chunks_lose_condition(self):
        self.assertFalse(any(has_complete_evidence(c) for c in fixed_size_chunks(load_text(POLICY))))
    def test_good_chunk_preserves_condition(self):
        self.assertTrue(all(has_complete_evidence(c) for c in section_chunks(load_text(POLICY))))
if __name__=='__main__': unittest.main()
