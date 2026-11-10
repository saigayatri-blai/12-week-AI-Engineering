import unittest

from day1_rag_demo import EMPLOYEE, HR_LEADER, Principal, grounded_answer, search


class FakeEmbedding:
    def tolist(self):
        return [[0.1, 0.2]]


class FakeModel:
    def encode(self, _values):
        return FakeEmbedding()


class FakeCollection:
    documents = {
        "hr_remote": "ABC Technologies employees may work remotely up to three days per week, subject to manager approval.",
        "hr_leave": "ABC Technologies employees receive 20 days of annual leave per year.",
        "hr_ceo": "ABC Technologies CEO's annual base salary is $534,000.",
    }

    def __init__(self):
        self.last_query = None

    def query(self, **kwargs):
        self.last_query = kwargs
        allowed = kwargs.get("where", {}).get("access", {}).get("$in")
        ids = ["hr_ceo", "hr_remote", "hr_leave"]
        if allowed == ["all"]:
            ids = ["hr_remote", "hr_leave"]
        return {
            "ids": [ids[:kwargs["n_results"]]],
            "documents": [[self.documents[item_id] for item_id in ids[:kwargs["n_results"]]]],
        }


class Day1Tests(unittest.TestCase):
    def setUp(self):
        self.collection = FakeCollection()
        self.model = FakeModel()

    def test_insecure_path_can_retrieve_salary(self):
        results = search(self.collection, self.model, "What is the CEO salary?")
        self.assertIn("hr_ceo", {item_id for item_id, _ in results})
        self.assertNotIn("where", self.collection.last_query)

    def test_employee_filter_excludes_salary(self):
        results = search(self.collection, self.model, "What is the CEO salary?", EMPLOYEE)
        self.assertNotIn("hr_ceo", {item_id for item_id, _ in results})
        self.assertEqual(self.collection.last_query["where"], {"access": {"$in": ["all"]}})
        self.assertIn("couldn't find", grounded_answer("What is the CEO salary?", results))

    def test_hr_leader_filter_allows_salary(self):
        results = search(self.collection, self.model, "What is the CEO salary?", HR_LEADER)
        self.assertIn("hr_ceo", {item_id for item_id, _ in results})
        self.assertEqual(
            self.collection.last_query["where"],
            {"access": {"$in": ["all", "hr_leadership"]}},
        )

    def test_unknown_role_denies_by_default(self):
        with self.assertRaises(ValueError):
            search(self.collection, self.model, "Anything", Principal("Unknown", "admin"))


if __name__ == "__main__":
    unittest.main()
