from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from spacy import Language
from transformers import Pipeline
import numpy as np
from typing import List, Dict
from datetime import datetime
from rapidfuzz import process, utils
from rapidfuzz.fuzz import partial_ratio




class NLPProcessor:
    """
    Handles NLP tasks for identifying movie-related entities in text and classifying the type of information requested about movies. 
    Uses pre-trained transformers for classification and pre-computed embeddings for entity recognition.
    """
    def __init__(self, movie_titles: List[str], people_names: List[str],
                #  movie_embeddings: Dict[str, np.ndarray], 
                 bart_pipe: Pipeline, nlp: Language, candidate_labels: List[str], question_types: List[str], multimedia_types: List[str]):
        """
        Initialize the NLPProcessor with movie embeddings, a zero-shot classifier pipeline, a spaCy NLP model, and candidate labels for classification.
        
        Parameters:
        movie_embeddings: A dictionary mapping movie titles to their vector embeddings.
        bart_pipe: A pipeline object from the transformers library for zero-shot classification.
        nlp: A spaCy language model for natural language processing tasks.
        candidate_labels: A list of strings representing candidate labels for the zero-shot classification.
        question_types: A list of strings representing types of questions for the zero-shot classification.
        """
        # self.movie_embeddings = movie_embeddings
        self.movie_titles = movie_titles
        self.people_names = people_names
        self.bart_pipe = bart_pipe
        self.candidate_labels = candidate_labels
        self.question_types = question_types
        self.nlp = nlp
        self.multimedia_types = multimedia_types

    def classify_question_type(self, text: str) -> str:
        """
        Classify the type of a question using a zero-shot classification pipeline.
        
        Parameters:
        text: A string containing the text of the question to be classified.
        """
        output = self.bart_pipe(text, self.question_types)

        # Find the index of the maximum score
        max_score_index = output['scores'].index(max(output['scores']))

        # Get the label with the highest score
        highest_score_label = output['labels'][max_score_index]
        
        return highest_score_label

    def classify_question(self, text: str) -> str:
        """
        Classify the intent of a question using a zero-shot classification pipeline.
        
        Parameters:
        text: A string containing the text of the question to be classified.
        """
        output = self.bart_pipe(text, self.candidate_labels)

        # Find the index of the maximum score
        max_score_index = output['scores'].index(max(output['scores']))

        # Get the label with the highest score
        highest_score_label = output['labels'][max_score_index]
        
        return highest_score_label
    
    def classify_multimedia_question(self, text: str) -> str:
        """
        Classify the type of a image using a zero-shot classification pipeline.
        
        Parameters:
        text: A string containing the text of the question to be classified.
        """

        output = self.bart_pipe(text, self.multimedia_types)

        # Find the index of the maximum score
        max_score_index = output['scores'].index(max(output['scores']))

        # Get the label with the highest score
        highest_score_label = output['labels'][max_score_index]
        
        return highest_score_label


    def get_embedding(self, words: List[str]) -> np.ndarray:
        """
        Generate a vector embedding for a given string of words.
        
        Parameters:
        words: A list of strings representing the words to be embedded.
        """
        return self.nlp(" ".join(words)).vector


    def match_titles_in_sentence_embeddings(self, sentence: str, title_word_limit:int=10) -> List[str]:
        """
        (Deprecated) Identify movie titles within a given sentence by generating embeddings for all possible word subsets and matching against known 
        movie title embeddings.
        
        Parameters:
        sentence: A string representing the sentence in which to find movie titles.
        """
        # Generate all possible subsets of words in the input sentence
        input_words = [token.text for token in self.nlp(sentence)]
        subsets = [input_words[start:end] for start, end in combinations(range(title_word_limit+1), 2)]

        # Calculate cosine similarity between input subsets and movie titles
        similarities = [
            (subset, title, cosine_similarity([self.get_embedding(subset)], [self.movie_embeddings[title]])[0, 0])
            for subset in subsets
            for title in self.movie_embeddings.keys()
        ]

        # Set a similarity threshold (may need to adjust this based on data)
        threshold = 0.95

        # Extract matched titles based on the subset with the highest similarity
        matched_titles = {}
        for subset, title, similarity in similarities:
            if similarity > threshold:
                if title not in matched_titles or similarity > matched_titles[title][1]:
                    matched_titles[title] = (subset, similarity)

        recognized_titles = []
        # Print matched titles and corresponding subsets
        for title, (subset, similarity) in matched_titles.items():
            # print(f"Matched Title: {title}, Subset: {subset}, Similarity: {similarity}")
            recognized_titles.append(title)
            # Return only the first movie as a list at the moment
        return recognized_titles
    
    def match_titles_in_sentence(self, sentence: str, threshold:int=90, min_title_length=3) -> List[str]:
        """
        Identify movie titles within a given sentence by using RapidFuzz library
        
        Parameters:
        sentence: A string representing the sentence in which to find movie titles.
        """
        normalized_sentence = utils.default_process(sentence)
        
        # Map normalized titles to original titles
        normalized_to_original = {utils.default_process(title): title for title in self.movie_titles if len(title) >= min_title_length}

        matches = process.extract(normalized_sentence, normalized_to_original.keys(), scorer=partial_ratio, limit=None)
        high_confidence_matches = [(match[0], match[1]) for match in matches if match[1] > threshold]

        # Sort matches by confidence score in descending order
        high_confidence_matches.sort(key=lambda x: x[1], reverse=True)

        final_matches = []
        for match in high_confidence_matches:
            if not any(match[0] != other_match[0] and match[0] in other_match[0] for other_match in high_confidence_matches):
                final_matches.append(match[0])

        # Sort the final matches so that the longest string is first
        final_matches.sort(key=lambda x: len(x), reverse=True)

        # Return the original titles
        return [normalized_to_original[match] for match in final_matches]
    

    def extract_person_name(self, sentence, threshold=90):
        normalized_sentence = utils.default_process(sentence)
        # Map normalized names to original names
        normalized_to_original = {utils.default_process(name): name for name in self.people_names}

        # Find the highest scoring match directly
        best_match = process.extractOne(normalized_sentence, normalized_to_original.keys(), scorer=partial_ratio, score_cutoff=threshold)

        # Return the original name of the best match if there is any, otherwise return None
        return normalized_to_original.get(best_match[0]) if best_match else None
        

    def format_date(self, date_string: str) -> str:
        """
        Converts a date string from 'yyyy-mm-dd' format to a more readable format like 'the 15th of March 1972'.

        Parameters:
        date_string: A string representing a date in 'yyyy-mm-dd' format.

        Returns:
        A string representing the formatted date.
        """
        try:
            # Convert the string to a datetime object
            date_obj = datetime.strptime(date_string, "%Y-%m-%d")

            # Format the date to a more readable string
            formatted_date = date_obj.strftime("the %dth of %B %Y")
            
            # Handle ordinal suffixes (1st, 2nd, 3rd, etc.)
            day = date_obj.day
            if 4 <= day <= 20 or 24 <= day <= 30:
                suffix = "th"
            else:
                suffix = ["st", "nd", "rd"][day % 10 - 1]
            formatted_date = formatted_date.replace("th", suffix)

            return formatted_date
        except ValueError:
            return "Invalid date format"
