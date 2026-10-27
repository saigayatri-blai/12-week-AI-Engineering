# Week 3 Day 4 — Python demo teleprompter
**Target narration: ~2:15–2:45, excluding installation/model download**

### Clip A — Create the database
**[SCREEN: VS Code and terminal]**

Now let's build a tiny vector database ourselves. We're using Chroma, a local vector database, and the same free Sentence Transformers model from earlier this week.

**[SHOW COLLECTION CODE]**

First, we create a persistent database called `shopping_db`. Persistent means the data stays on disk after our script finishes. Our collection is called `products`, and we're using cosine distance for searching.

### Clip B — Store product embeddings
**[SHOW PRODUCT LIST AND UPSERT]**

Here's our catalog: Sakinalu, Chekkinalu, mango pickle, murukku, and rice crackers. Each product has a description, category, price, and stock status.

We convert the descriptions into embeddings and store those vectors alongside the product information.

**[RUN CODE; SHOW ACTUAL PRODUCT COUNT AND 384 DIMENSIONS]**

Look at that: five products, each represented by 384 numbers. And because we're using `upsert`, rerunning this script won't duplicate our product IDs.

### Clip C — Search with filters
**[SHOW QUERY AND FILTERS]**

Now a customer searches for **crispy rice snack**. We convert that query into an embedding and send it to Chroma.

But we don't just want related products. We want **snacks under ten dollars that are currently in stock**. So we add metadata filters.

**[SHOW ACTUAL RESULTS]**

Here are the returned products and their distances. These are cosine **distances**, not cosine similarity scores. Smaller values mean closer matches.

Notice that out-of-stock products are excluded, even if their descriptions are relevant.

### Clip D — Persistence and Day 5 teaser
**[SHOW PERSISTENCE CHECK]**

Finally, we reopen the collection and verify the products are still there. That's the difference between storing vectors in a temporary Python array and using a persistent database.

Now imagine doing this for millions of products. How can the database find useful results without checking every vector? Tomorrow we'll explore **vector indexing and Meta FAISS**.
