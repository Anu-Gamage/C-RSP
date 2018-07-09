-- 3Sources Dataset

We provide here a new multi-view text dataset, collected from three online news sources: BBC, Reuters, and The Guardian. This dataset exhibits a number of common aspects of multi-view problems highlighted previously -- notably that certain stories will not be reported by all three sources (i.e incomplete views), and the related issue that sources vary in their coverage of certain topics (i.e. partially missing patterns).
Dataset construction

In total we collected 948 news articles covering 416 distinct news stories from the period February–April 2009. Of these stories, 169 were reported in all three sources, 194 in two sources, and 53 appeared in a single news source. Each story was manually annotated with one or more of the six topical labels: business, entertainment, health, politics, sport, technology. These roughly correspond to the primary section headings used across the three news sources.

This archive contains data for 3 different views. The view data files have the following formats: 
    - *.mtx: Term frequencies stored in a sparse term-document matrix in Matrix Market format.
    - *.terms: List of content-bearing terms in the corpus, with each line corresponding to a row of the term-document matrix.
    - *.docs: List of story identifiers, with each line corresponding to a column of the term-document matrix. Note that the story identifiers correspond across views.
    
In addition annotations are provided for the news stories:
    - 3sources.overlap.clist: Overlapping (multi-label) annotated topic classes.            
    - 3sources.disjoint.clist: Non-overlapping (single label) annotated topic classes, based on dominant topic for each story.             

This dataset is made available for non-commercial and research purposes only, and is provided in pre-processed matrix format. Note that stemming (Porter algorithm), stop-word removal, and low term frequency filtering (count < 3) have already been applied to the data. 

All rights, including copyright, in the content of the original abstracts are owned by the original authors.  

For further details please contact derek.greene@ucd.ie

http://mlg.ucd.ie/datasets/3sources.html

