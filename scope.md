\# CSCI370 Project Scope



\## Topic

Analyze YouTube comments on Ryan Trahan videos to extract audience sentiment, key entities/topics, and enable source-grounded Q\&A/summarization using RAG.



\## Dataset Boundaries

\- Channel/theme: Ryan Trahan official videos

\- Number of videos: 30

\- Target comments: 10000 top-level comments (replies optional)

\- Language: English only

\- Period: recent + popular videos (mix for diversity)



\## Supported Query Types

1\. QA

&#x20;  - Example: "What do viewers say about Ryan's storytelling?"

2\. Summary

&#x20;  - Example: "Summarize audience opinion across these videos."

3\. Insight
&#x20;  - Example: "What are the most common negative topics and named entities?"



\## Success Criteria

\- NLP pipeline produces: sentiment, NER, keywords, topic modeling

\- RAG returns grounded answers with retrieved source comments

\- Agent/router correctly directs user query to QA / summary / insight flow

\- Basic dashboard supports interaction and displays outputs

\- MLflow logs experiments and evaluation metrics

