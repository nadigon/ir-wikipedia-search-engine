inverted_index_gcp.py
This file contains the inverted index implementation provided by the course staff.
It implements an inverted index optimized for working with Google Cloud Storage (GCP) and is used without modifications.
The module supports efficient reading and writing of posting lists and index metadata to and from GCP buckets, including handling large posting lists using a multi-file storage mechanism.
It serves as the core indexing infrastructure used by our search engine for accessing precomputed indexes during query time.


preparation.ipynb
This notebook is responsible for all preparation steps of the search engine.
It builds all required inverted indexes (with and without stemming), computes document lengths (DL), and creates mappings such as doc2title.
These preprocessing steps are computationally expensive and therefore performed once in advance, allowing search_frontend.py to return results efficiently at query time.
Most of the code in this notebook is based on Assignment 3 of the Information Retrieval course, with adaptations for the final project.




search_frontend.py
his file contains the implementation of 6 main search-related methods(search, search_body , search_title , search_anchor , get_pagerank , get_pageview)
in this project, we focused primarily on the main search method.
For each query, the query is tokenized and removed the stopwords.
Then, a relevance score is calculated separately for the title and the body of each document using BM25 for each query token.
The title and body scores are merged using predefined weights, and documents are ranked according to their final combined score.
Although all six methods were implemented, the final system uses only the main search method