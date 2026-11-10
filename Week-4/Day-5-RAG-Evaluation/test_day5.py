import unittest

from BuildLifeandAI_Week4_Day5_Evaluation_Demo import (
    GOLD_EVIDENCE_IDS,
    create_collection,
    get_encoder,
    grounded_answer,
    recall_at_k,
    retrieve,
)


class Day5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = get_encoder()
        cls.collection, _ = create_collection(cls.encoder)
        cls.retrieved = retrieve(cls.collection, cls.encoder, top_k=3)
        cls.ids = [item["id"] for item in cls.retrieved]

    def test_actual_recall_at_1_is_half(self):
        self.assertEqual(recall_at_k(self.ids, GOLD_EVIDENCE_IDS, 1), 0.5)

    def test_actual_recall_at_3_is_complete(self):
        self.assertEqual(recall_at_k(self.ids, GOLD_EVIDENCE_IDS, 3), 1.0)

    def test_top_1_abstains_when_evidence_is_incomplete(self):
        self.assertIn("enough retrieved evidence", grounded_answer(self.retrieved[:1]))

    def test_top_3_uses_twelve_month_requirement_and_two_month_tenure(self):
        answer = grounded_answer(self.retrieved)
        self.assertIn("12 months", answer)
        self.assertIn("2 months", answer)
        self.assertTrue(answer.startswith("No."))


if __name__ == "__main__":
    unittest.main()
