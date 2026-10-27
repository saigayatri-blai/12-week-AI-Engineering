# Day 5 — FAISS Python demo | ~2 minutes 30 seconds

**[SCREEN: Open day5_faiss.py in VS Code]**

Now let's build a real vector search index using Meta FAISS.

First, we have six product descriptions: running shoes, sneakers, walking shoes, office shoes, and headphones.

We use a free Sentence Transformers model to convert them into embeddings.

**[SHOW: model.encode and embedding dimension]**

Each product becomes a vector containing 384 numbers.

Next, we create an IndexFlatIP index. This performs exact nearest-neighbor search. Because we normalized the vectors, inner product gives us cosine similarity.

Now we search for *comfortable running shoes*.

**[RUN: SHOW ACTUAL TOP THREE PRODUCTS AND SCORES]**

These are the three closest products returned by the model. The scores come from the actual embeddings—not manually assigned values.

But six products is easy. What happens when the catalog gets much larger?

**[SHOW: synthetic vectors section]**

For the second experiment, we're generating 10,000 **synthetic** vectors. These are not real product embeddings. We're using them to demonstrate the indexing algorithm.

First, we build an exact index. Then we build an IVF index with 100 clusters.

Here's the important parameter: *nprobe*. It controls how many clusters IVF searches.

**[SHOW: nprobe 1, 5, and 20]**

With nprobe set to one, we search only one cluster. At five, we search five. At twenty, we search twenty.

**[RUN: SHOW EXACT IDS, IVF IDS, AND RECALL@5]**

Now compare the results. Recall at five tells us how many of the exact top-five neighbors IVF recovered for this test query.

Searching more clusters often improves recall, but it also increases the search work.

And that's the engineering tradeoff: speed versus retrieval quality. In a real system, we'd test many queries and measure latency too.

Tomorrow, we'll combine embeddings and FAISS into our first semantic search application.
