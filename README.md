# Wikipedia Search Engine

A search engine for Wikipedia articles, built as part of the Information Retrieval course (Assignment 3).

## Project Overview

This search engine implements a ranking algorithm that combines:
- **Title matching** with IDF weighting
- **BM25** scoring on document body
- **PageRank** for authority signals
- **Multi-term coverage bonus** for documents matching multiple query terms

## Performance

| Metric | Score |
|--------|-------|
| Average Results Quality (RQ) | 0.151 |
| Average Precision@5 | 0.387 |
| Average Precision@10 | 0.260 |
| Average Response Time | 2.52s |

## Project Structure

```
├── search_frontend.py      # Main Flask application with search endpoints
├── inverted_index_gcp.py   # Inverted index implementation for GCP
├── create_index.ipynb      # Notebook for creating inverted indices
├── evaluation/
│   ├── run_evaluation.py   # Evaluation script for 30 training queries
│   ├── queries_train.json  # Training queries with relevance judgments
│   └── analyze_failures.py # Analysis of failed queries
├── deployment/
│   ├── startup_script_gcp.sh    # GCP instance startup script
│   └── run_frontend_in_gcp.sh   # Deployment instructions
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search?query=...` | GET | Main search - returns top 100 results |
| `/search_body?query=...` | GET | TF-IDF search on body text only |
| `/search_title?query=...` | GET | Search by title matching |
| `/search_anchor?query=...` | GET | Search by anchor text |
| `/get_pagerank` | POST | Get PageRank values for document IDs |
| `/get_pageview` | POST | Get page view counts for document IDs |

## Data Files (Google Cloud Storage)

**Bucket:** `gs://dotan-irassignment3-bucket/`

| File | Description | Size |
|------|-------------|------|
| `postings_gcp/index.pkl` | Body text inverted index | ~18 MB |
| `doc_titles.pkl` | Document ID to title mapping | ~169 MB |
| `doc_lengths.pkl` | Document lengths for BM25 | ~44 MB |
| `pagerank.pkl` | PageRank scores | ~85 MB |
| `title_postings.pkl` | Title inverted index | ~XX MB |

## Deployment

### Prerequisites
- Google Cloud Platform account
- Python 3.8+
- Flask, NLTK, google-cloud-storage

### Run Locally
```bash
pip install flask nltk google-cloud-storage
python search_frontend.py
```

### Deploy to GCP
```bash
# Create instance
gcloud compute instances create search-engine-instance \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --metadata-from-file startup-script=deployment/startup_script_gcp.sh

# Copy application
gcloud compute scp search_frontend.py USER@search-engine-instance:/home/USER

# SSH and run
gcloud compute ssh USER@search-engine-instance
~/venv/bin/python ~/search_frontend.py
```

## Algorithm Details

### Scoring Formula

```
score = title_score + bm25_score + coverage_bonus + pagerank_boost

where:
- title_score = IDF-weighted sum of matching terms in title (0-100)
- bm25_score = normalized BM25 score from body text (0-100)
- coverage_bonus = bonus based on fraction of query terms matched:
    - 100% terms: +300 points
    - 75%+ terms: +150 points
    - 50%+ terms: +75 points
- pagerank_boost = min(pagerank * 5, 10)
```

### BM25 Parameters
- k1 = 1.5
- b = 0.75

## Authors

- [ameer jamool] - [ameerja@post.bgu.ac.il]
- [dotan katz] - [dotankat@post.bgu.ac.il]
