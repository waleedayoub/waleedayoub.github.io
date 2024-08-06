---
date: 2024-08-06T14:50:46-04:00
title: "LLM Zoomcamp Week 4 - Monitoring and Evaluation Notes"
author: waleed
description: Datatalksclub LLM Zoomcamp Week 4 Notes
tags: ["datatalksclub", "LLM","python"]
series: ["llm-zoomcamp-2024"]
hasMermaid: true
draft: false
---

# LLM Zoomcamp - Week 4 Notes

A quick recap of what the first three sections have been about:
- Let's chart with the visual of what we created and map it back to our `rag` function:
  ```mermaid
    graph TD
        A[User] -->|Q| B[Knowledge DB]
        B -->|Relevant Documents D1, D2, ..., DN| C[Context = Prompt + Q + Documents]
        A -->|Q| C
        C -->|Q| D[LLM]
        D -->|Answer| A
        subgraph Context
            direction LR
            D1
            D2
            D3
            D4
            ...
            DN
        end
        B -.-> D1
        B -.-> D2
        B -.-> D3
        B -.-> D4
        B -.-> ...
        B -.-> DN
        classDef entity fill:#f9f,stroke:#333,stroke-width:4px;
    ```
  - and the function itself was:
    ```python
    def rag(query):
      search_results = search(query)
      prompt = prompt_builder(query, search_results)
      answer = llm(prompt)
      return answer
    ```
- In section 1:
  - We built the scaffold for the function above
  - We learned all about what a RAG is, how to apply a common "search" problem using a source document as context, how to implement one using OpenAI's GPT models, and how to use Elasticsearch to do "semantic" or "keyword" search to simplify the size of the documents being passed to the LLM
- In section 2:
  - We implemented various versions of the `llm` function
  - We focused further on self-hosted LLMs and how to effectively replicate everything we did in section 1 but using `ollama` as a platform to access self-hosted models
  - I further set up my windows gaming PC to act as a server running 3 containers: `ollama`, `openwebui` and `elasticsearch` in order to have "always on" access to these services
- In section 3:
  - We experimented with various implementations of the `search` function
  - We switched from doing a straight "semantic" or "keyword" search using Elasticsearch to creating embeddings in order to do vector search. The main difference here is that instead of relying on Elasticsearch's Lucine engine to look up relevant documents based on a text query, we were using various `encoding` algorithms like cosine distance, SBERT models, etc.
  - We then built a ground-truth dataset using LLMs in order to evaluate the quality of our retrieval system and compared the performance of "semantic" search and "vector" search in retrieving the most relevant documents for a given query

In this section the focus is on the following:
- Extending the evaluation work we did in section 3 to monitor answer quality over time
- How to look at answer quality with user feedback and interaction
- How to store all this data and visualize it, etc.