"""BuildLifeandAI Week 3 Day 4: local persistent vector database demo."""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = Path(__file__).resolve().parent / 'shopping_db'
MODEL_NAME = 'all-MiniLM-L6-v2'
PRODUCTS = [
    {'id': 'p1', 'name': 'Sakinalu', 'description': 'Traditional crispy rice flour snack', 'category': 'snacks', 'price': 7.50, 'in_stock': True},
    {'id': 'p2', 'name': 'Chekkinalu', 'description': 'Crunchy homemade rice crackers', 'category': 'snacks', 'price': 6.50, 'in_stock': True},
    {'id': 'p3', 'name': 'Mango Pickle', 'description': 'Sweet mango pickle', 'category': 'pickles', 'price': 5.00, 'in_stock': True},
    {'id': 'p4', 'name': 'Murukku', 'description': 'Spicy South Indian murukku', 'category': 'snacks', 'price': 8.00, 'in_stock': False},
    {'id': 'p5', 'name': 'Rice Crackers', 'description': 'Crispy savory rice crackers', 'category': 'snacks', 'price': 9.00, 'in_stock': True},
]


def main():
    print('\n=== WEEK 3 DAY 4: VECTOR DATABASES ===')
    print('Loading embedding model (first run downloads it)...')
    model = SentenceTransformer(MODEL_NAME)

    print('\n[1] CREATE PERSISTENT CHROMA COLLECTION')
    client = chromadb.PersistentClient(path=str(DB_PATH))
    # Modern Chroma collection configuration. Use `configuration` instead of legacy
    # metadata={"hnsw:space": "cosine"}, which may be deprecated in recent versions.
    collection = client.get_or_create_collection(
        name='products',
        configuration={'hnsw': {'space': 'cosine'}},
    )
    print('Local database:', DB_PATH)

    print('\n[2] GENERATE AND STORE EMBEDDINGS + METADATA')
    descriptions = [p['description'] for p in PRODUCTS]
    embeddings = model.encode(descriptions).tolist()
    collection.upsert(
        ids=[p['id'] for p in PRODUCTS],
        documents=descriptions,
        embeddings=embeddings,
        metadatas=[{
            'name': p['name'], 'category': p['category'],
            'price': p['price'], 'in_stock': p['in_stock'],
        } for p in PRODUCTS],
    )
    print('Products stored:', collection.count())
    print('Embedding dimensions:', len(embeddings[0]))
    print('Upsert is safe to rerun: existing product IDs are updated, not duplicated.')

    print('\n[3] SEMANTIC SEARCH WITH METADATA FILTERS')
    query = 'crispy rice snack'
    query_embedding = model.encode(query).tolist()
    filters = {'$and': [
        {'category': {'$eq': 'snacks'}},
        {'price': {'$lt': 10.0}},
        {'in_stock': {'$eq': True}},
    ]}
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where=filters,
        include=['documents', 'metadatas', 'distances'],
    )
    print('Query:', query)
    print('Filters: snacks, under $10, in stock')
    for product_id, document, metadata, distance in zip(
        results['ids'][0], results['documents'][0],
        results['metadatas'][0], results['distances'][0],
    ):
        print(f"{product_id} | {metadata['name']} | ${metadata['price']:.2f} | cosine distance={distance:.3f}")
        print(f'  Description: {document}')
    print('Smaller cosine distance means more aligned vectors.')
    print('Rankings reflect the model and descriptions; regional-name understanding is not guaranteed.')
    #
    print('\n[4] PERSISTENCE CHECK')
    reopened_client = chromadb.PersistentClient(path=str(DB_PATH))
    reopened_collection = reopened_client.get_collection('products')
    print('Reopened collection product count:', reopened_collection.count())
    print('Saved data directory:', DB_PATH)
    print('\nNEXT: Day 5 — Vector indexing and Meta FAISS')


if __name__ == '__main__':
    main()
