# Intelligent Conversational Agent

This repository contains the implementation of a conversational agent developed as part of the **Advanced Topics in AI** course project. The agent is designed to answer natural language questions about movies and provide intelligent recommendations, combining multiple AI techniques learned during the course.

**Developed by**: *Christian Skorski & Konstantin Moser*
**Date**: Fall Semester 2023

---

## 🧠 Project Overview

The agent is capable of handling:

* **Closed questions**, e.g. *"Who directed Inception?"*
* **Open questions**, e.g. *"What should I watch if I liked Interstellar?"*
* **Multimodal queries**, e.g. *"Show me a picture of Leonardo DiCaprio."*

It integrates multiple techniques:

* Knowledge graph querying via SPARQL (Wikidata)
* Embedding-based reasoning for incomplete or fuzzy queries
* Crowdsourced data correction and fallback
* Personalized movie recommendations using SVD
* Image retrieval from an IMDb-based dataset

---

## 🛠 Features

### 1. Natural Language Understanding

* Zero-shot classification to detect query type and intent
* Fuzzy matching for movie/actor names (using RapidFuzz)

### 2. Movie Knowledge Retrieval

* SPARQL queries on a movie-centric RDF knowledge graph
* Embedding fallback for unknown entries

### 3. Recommender System

* Uses MovieLens data and the **Surprise** library
* Recommends movies based on user-provided favorites
* Filters by genre and release year

### 4. Multimedia Integration

* Fetches up to 3 relevant images for actors, movie scenes, posters, or behind-the-scenes

---

## 📥 Example Queries

#### ➤ Closed Question

**Input**: Who is the screenwriter of *The Shawshank Redemption*?
**Output**: Frank Darabont and Stephen King.

#### ➤ Open Recommendation

**Input**: I liked *Inception* and *Interstellar*, what should I watch?
**Output**: Sherlock: The Abominable Bride, Source Code, The Double, etc.

#### ➤ Multimedia Query

**Input**: What does Leonardo DiCaprio look like?
**Output**: \[Displays image]

---

## ⚙️ Architecture & Components

* `GraphAgent`: Handles SPARQL queries and IMDb ID lookups
* `NLPProcessor`: Classifies and extracts intent + entities
* `EmbeddingsAgent`: Predicts missing triplets using vector similarity
* `Recommender`: Personalized suggestions via SVD
* `CrowdsourcedDataAgent`: Validates and supplements graph answers
* `ResponseGenerator`: Orchestrates everything into a human-readable reply

![image](https://github.com/user-attachments/assets/bec31cdc-8e98-4850-88ab-e34a57eb923f)
![image](https://github.com/user-attachments/assets/f86e53e2-0f07-4e02-b359-7fd90eb1ae34)

---

## 📌 Limitations

* Limited to common movie attributes (e.g., director, genre, etc.)
* Small recommendation dataset (subset of MovieLens)
* Doesn’t support very specific image queries (e.g., *actor X in movie Y*)

---

## 📚 Libraries Used

* `rdflib` (graph queries)
* `transformers` (zero-shot classification)
* `Surprise` (SVD recommendation)
* `RapidFuzz` (fuzzy name matching)
* `Statsmodels`, `Pandas`, `Numpy`, `Sklearn`

---

## 📁 Disclaimer

This is a course project. The repository does not include all code due to internal or licensing constraints. For academic inquiries, feel free to reach out.
