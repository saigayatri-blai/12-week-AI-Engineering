# Day 2 Python demo — teleprompter (approximately 2:15–2:45)

**[SCREEN: project open in VS Code]**

Now let's test keyword search against semantic search, using a real example.

Our customer searches for **Sakinalu**, but the product catalog contains **Traditional Andhra Chekkinalu**, plus four other foods.

**[SHOW PRODUCT LIST]**

First, basic keyword search. Our code checks whether the exact query text appears in each product listing.

**[RUN; SHOW SECTION 1]**

Zero results. That's expected: the word Sakinalu isn't in our catalog.

Now let's try semantic search.

We're using the same free embedding model from yesterday: all-MiniLM-L6-v2. It converts the product descriptions and the query into vectors, then compares them using cosine similarity.

**[SHOW SECTION 2, ACTUAL OUTPUT]**

Here are the actual rankings. Look at where Chekkinalu appears.

**[PAUSE]**

If it ranks highly, the model has captured a useful connection in this test. If it doesn't, that's equally important: embeddings don't automatically understand every regional name.

And a similarity score alone is not proof that a model knows two foods are equivalent.

But what if we already know these names are related?

**[SHOW SECTION 3]**

We add a tiny synonym mapping: Sakinalu maps to Chekkinalu.

Now run the keyword search again.

**[SHOW MATCH]**

We find the listing immediately. No embedding model needed for that rule.

Here's the engineering lesson: **use the simplest reliable approach, and evaluate the results instead of assuming AI gets them right.**

Tomorrow we'll unpack the math behind those similarity scores: cosine similarity and vector distance.
