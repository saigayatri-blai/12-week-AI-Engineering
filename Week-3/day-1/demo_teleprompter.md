# Day 1 Python Demo — Teleprompter (about 2:30)

**[SCREEN RECORDING — SHOW PROJECT]**

Now let's stop talking about embeddings and actually create one.

We're going to use Python and a free pretrained model called all-MiniLM-L6-v2. You don't need a GPU or a paid API.

**[SHOW INSTALL COMMAND — SKIP INSTALL WAIT]**

First, install Sentence Transformers. I've already installed it so we can focus on the experiment.

**[RUN PYTHON FILE — SHOW SHAPE]**

Our first experiment has three product descriptions: Sakinalu, Chekkinalu, and a smartphone charging cable.

We pass them through the same embedding model. Look at the output: three rows and 384 columns.

That means each description has become a vector of 384 numbers. These first five values are just a tiny part of the Sakinalu embedding.

**[HIGHLIGHT EXPERIMENT 2 SCORES]**

Now the interesting part. Let's compare Sakinalu and Chekkinalu using cosine similarity. We also compare Sakinalu with the smartphone cable.

Look at the actual scores. Which pair is more similar?

But wait. Both snack descriptions contain the same words: traditional crispy rice flour snack.

So does the model really understand the regional names? Or is it matching the shared description?

**[HIGHLIGHT EXPERIMENT 3]**

Let's remove the descriptions and test only two words: Sakinalu and Chekkinalu.

Now compare this score with the earlier one.

**[PAUSE — SHOW ACTUAL SCORES]**

This is an important AI engineering lesson. Similarity scores depend on the model and the context we provide. A high score alone doesn't prove the model knows these are regional names for related snacks.

If we were building a real shopping search engine, we'd test more regional names and perhaps add descriptive text or a synonym dictionary.

And that's it. You just generated embeddings and compared them using Python.

Tomorrow, we'll use this idea to understand semantic search versus keyword search.
