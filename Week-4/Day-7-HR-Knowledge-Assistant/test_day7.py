import os
from pathlib import Path
import tempfile
import unittest

from evaluate import run_evaluation
from ingest import ingest_documents, get_encoder
from rag import (
    ABSTENTION,
    HR_LEADERSHIP,
    REGULAR_EMPLOYEE,
    answer_question,
)


class Day7IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_dir = Path(cls.temp_dir.name) / "chroma"
        cls.encoder = get_encoder()
        cls.collection = ingest_documents(cls.db_dir, encoder=cls.encoder)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def ask(self, question, profile):
        return answer_question(
            question,
            profile,
            collection=self.collection,
            encoder=self.encoder,
        )

    def test_regular_employee_salary_is_denied_before_context(self):
        result = self.ask("What is the CEO's annual salary?", REGULAR_EMPLOYEE)
        self.assertEqual(result["answer"], ABSTENTION)
        self.assertNotIn("executive_compensation", {item["id"] for item in result["sources"]})

    def test_hr_leadership_salary_is_allowed(self):
        result = self.ask("What is the CEO's annual salary?", HR_LEADERSHIP)
        self.assertIn("$534,000", result["answer"])
        self.assertIn("executive_compensation", {item["id"] for item in result["sources"]})

    def test_remote_work_answer_matches_recorded_example(self):
        result = self.ask("How many days per week can employees work remotely?", REGULAR_EMPLOYEE)
        self.assertIn("up to three days per week, subject to manager approval", result["answer"])
        self.assertIn("remote_work", {item["id"] for item in result["sources"]})

    def test_recall_at_3_is_calculated_from_real_retrieval(self):
        rows = run_evaluation(self.collection, self.encoder)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertGreaterEqual(row["recall_at_3"], 0.0)
            self.assertLessEqual(row["recall_at_3"], 1.0)
            self.assertTrue(set(row["relevant_ids"]) & set(row["retrieved_ids"]))


class Day7StreamlitTests(unittest.TestCase):
    def test_required_ui_flows(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as temp_dir:
            db_dir = Path(temp_dir) / "chroma"
            encoder = get_encoder()
            ingest_documents(db_dir, encoder=encoder)
            previous = os.environ.get("DAY7_CHROMA_DIR")
            os.environ["DAY7_CHROMA_DIR"] = str(db_dir)
            try:
                app = AppTest.from_file("app.py", default_timeout=30).run()

                # Required demo 1: normal remote-work answer.
                app.button[0].click().run()
                self.assertFalse(app.exception)
                rendered_text = "\n".join(element.value for element in app.markdown)
                self.assertIn("up to three days per week, subject to manager approval", rendered_text)

                # Required demo 2: regular employee cannot retrieve salary evidence.
                app.selectbox[0].select("Regular Employee")
                app.text_input[0].set_value("What is the CEO's annual salary?")
                app.button[0].click().run()
                rendered_text = "\n".join(element.value for element in app.markdown)
                self.assertIn(ABSTENTION, rendered_text)
                self.assertNotIn("Executive Compensation", rendered_text)

                # Required demo 3: HR leadership can retrieve the salary source.
                app.selectbox[0].select("HR Leadership")
                app.button[0].click().run()
                rendered_text = "\n".join(element.value for element in app.markdown)
                self.assertIn("$534,000", rendered_text)
                self.assertIn("Executive Compensation", rendered_text)
            finally:
                if previous is None:
                    os.environ.pop("DAY7_CHROMA_DIR", None)
                else:
                    os.environ["DAY7_CHROMA_DIR"] = previous


if __name__ == "__main__":
    unittest.main()
