import requests
import json
from time import time

URL = 'http://34.44.77.84:8080'

# Queries that failed completely
failed_queries = [
    "Television invention broadcast media",
    "Impressionism Monet Renoir", 
    "Stonehenge prehistoric monument",
    "Photography invention Daguerre",
    "Ballet origins France Russia",
    "Fossil fuels climate change"
]

# Expected results for these queries
expected = {
    "Television invention broadcast media": [3636075, 29831, 527026, 14682695, 113604],
    "Impressionism Monet Renoir": [60214787, 15169, 21435370, 46351674, 57826068],
    "Stonehenge prehistoric monument": [27633, 5936517, 3151382, 3730333, 230566],
    "Photography invention Daguerre": [103177, 247934, 2435889, 61476134, 3032314],
    "Ballet origins France Russia": [15669381, 4802982, 49733, 1325838, 1161691],
    "Fossil fuels climate change": [48146, 5042951, 12686181, 3201, 46255716]
}

print("="*80)
print("ANALYZING FAILED QUERIES")
print("="*80)

for query in failed_queries:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    # Get current results
    try:
        res = requests.get(f"{URL}/search", params={'query': query}, timeout=35)
        if res.status_code == 200:
            data = res.json()
            current_ids = [item[0] for item in data[:20]]
            current_titles = [item[1] for item in data[:20]]
            
            print(f"\nCurrent Top 10:")
            for i, (doc_id, title) in enumerate(data[:10], 1):
                is_expected = doc_id in expected[query]
                mark = "✓" if is_expected else "✗"
                print(f"  {i:2d}. [{mark}] {doc_id}: {title[:55]}")
            
            print(f"\nExpected Top 5:")
            for doc_id in expected[query][:5]:
                if doc_id in current_ids:
                    rank = current_ids.index(doc_id) + 1
                    print(f"  {doc_id}: Found at rank {rank}")
                else:
                    print(f"  {doc_id}: NOT IN TOP 20")
                    
    except Exception as e:
        print(f"Error: {e}")

# Also test search_title and search_body separately
print("\n" + "="*80)
print("TESTING INDIVIDUAL ENDPOINTS")
print("="*80)

for query in ["Impressionism Monet Renoir", "Stonehenge prehistoric monument"]:
    print(f"\n--- {query} ---")
    
    # Test search_title
    try:
        res = requests.get(f"{URL}/search_title", params={'query': query}, timeout=35)
        if res.status_code == 200:
            data = res.json()[:5]
            print(f"search_title top 5:")
            for doc_id, title in data:
                print(f"  {doc_id}: {title[:50]}")
    except Exception as e:
        print(f"search_title error: {e}")
    
    # Test search_body
    try:
        res = requests.get(f"{URL}/search_body", params={'query': query}, timeout=35)
        if res.status_code == 200:
            data = res.json()[:5]
            print(f"search_body top 5:")
            for doc_id, title in data:
                is_expected = doc_id in expected.get(query, [])
                mark = "✓" if is_expected else "✗"
                print(f"  [{mark}] {doc_id}: {title[:50]}")
    except Exception as e:
        print(f"search_body error: {e}")
