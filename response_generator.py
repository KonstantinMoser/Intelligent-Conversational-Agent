from typing import Dict, List
from graph_agent import GraphAgent
from nlp_processor import NLPProcessor
from embeddings_agent import EmbeddingsAgent
from recommender import Recommender
from crowdsourced_data_agent import CrowdsourcedDataAgent

class ResponseGenerator:
    """
    Generates natural language responses to user queries by integrating 
    outputs from NLP processing and graph querying.
    """

    def __init__(self, graph_agent: GraphAgent, nlp_processor: NLPProcessor, embeddings_agent: EmbeddingsAgent,
                 recommender: Recommender, crowd_data_agent: CrowdsourcedDataAgent, question_types, multimedia_types):
        """
        Initialize the ResponseGenerator with instances of GraphAgent and NLPProcessor.
        
        Parameters:
        graph_agent: An instance of GraphAgent used for executing SPARQL queries.
        nlp_processor: An instance of NLPProcessor used for natural language processing tasks.
        """
        self.graph_agent = graph_agent
        self.nlp_processor = nlp_processor
        self.embeddings_agent = embeddings_agent
        self.recommender = recommender
        self.question_types = question_types
        self.multimedia_types = multimedia_types
        self.crowd_data_agent = crowd_data_agent


    def generate_response(self, question: str, debug:bool=True) -> str:
        """
        Generate a natural language response to a user's question by identifying movie titles, 
        classifying the question intent, and attempting to find answers using both graph queries and embeddings.
        
        Parameters:
        question: A string representing the user's question.
        """
        # Classify the type of the question
        question_type = self.nlp_processor.classify_question_type(question)
        print("Question Type: " + question_type)
        
        # Case multimedia question: Give back a picture

        if question_type in self.question_types[3:7] or "poster" in question or "behind the scenes" in question or "behind-the-scenes" in question: # Zero Shot found multimedia question
            multimedia_type = self.nlp_processor.classify_multimedia_question(question)
            print("Multimedia Type: " + multimedia_type)


            # multimedia_types = [
            #     "demand to show a scene from a movie",                        0
            #     "demand to show the image of a movie poster",                 1
            #     "demand to show an behind the scenes image of a movie",       2
            #     "demand to show an image of an actor"                         3
            # ]


            if multimedia_type in self.multimedia_types[0:3]:
                movie_titles = self.nlp_processor.match_titles_in_sentence(question)
                print("Found Movie:")
                print(movie_titles[0])
                if not movie_titles:
                    return "I couldn't find the movie title(s) in your question."
                
                movie_imdbid = self.graph_agent.get_imdbid(movie_titles[0])
                if not movie_imdbid:
                    return f"I couldn't find the IMDb ID for the movie '{movie_titles[0]}'."

                if multimedia_type == self.multimedia_types[0]:
                    return self.graph_agent.find_movie_images(movie_imdbid, 3, ["still_frame"])

                elif multimedia_type == self.multimedia_types[1]:
                    return self.graph_agent.find_movie_images(movie_imdbid, 3, ["poster"])

                elif multimedia_type == self.multimedia_types[2]:
                    return self.graph_agent.find_movie_images(movie_imdbid, 3, ["behind_the_scenes"])

            else:
                # Extract the name of the person in the sentence
                person_name = self.nlp_processor.extract_person_name(question)
                person_imdbid = self.graph_agent.get_imdbid(person_name)
                if not person_imdbid:
                    return f"I couldn't find the IMDb ID for the person '{person_name}'."
                
                # return self.graph_agent.find_person_images(person_imdbid, 3, ["publicity"])
                return self.graph_agent.find_person_images(person_imdbid, 3)

        else:
    
        # Identify the movie title in the question
            movie_titles = self.nlp_processor.match_titles_in_sentence(question)
            if not movie_titles:
                return "I couldn't find the movie title(s) in your question."
            if debug and len(movie_titles) < 100:
                print(f"Movies found in the question: {movie_titles}")
                #         question_types = [
        #     "What could I watch based on my movie preferences",           0
        #     "Recommend similar movies",                                   1
        #     "its a specific question asking something about a movie",     2
        #     "show picture of an actor",                                   3
        #     "show image or scene from a movie",                           4
        #     "display a picture of a poster of behind the scenes"          5
        # ]

            intent = self.nlp_processor.classify_question(question)

            # Case closed question: Find answer to closed question
            if question_type == self.question_types[2] and intent != "recommendation request": # Zero Shot found specific question
                # Classify the intent of the question
                
                
                if intent == "producer" and "executive" in question:
                        intent = "executive producer"
                
                if not intent:
                    return "I couldn't determine what information you're asking for."
                if debug:
                    print(f"Intent of the question: {intent}")
                # Use best matched title
                movie_title = movie_titles[0]
                
                # Look for answers using queries
                kg_result = self.find_answer_with_queries(movie_title, intent, debug=debug)
                print("kg_result:")
                print(kg_result)
                query_reply = self.generate_answer_sentence(intent, movie_title, kg_result) if kg_result else ""
                
                # Look for answer in embeddings or fix KG one
                embeddings_result = self.find_answer_with_embeddings(movie_title, intent)
                
                ### Generate the response based on the results
                final_reply = ""
                
                # Append KG response if found
                final_reply += query_reply + "\n\n"
                    
                # Append crowdsourced answer if found
                crowd_reply = self.find_answer_in_crowdsourced_data(movie_title, intent, debug)
                final_reply += crowd_reply
                    
                # Otherwise, append embeddings answer if found
                if not query_reply and not crowd_reply and embeddings_result:
                    if len(embeddings_result) > 1:
                        embeddings_result = ', '.join(embeddings_result)
                    else:
                        embeddings_result = embeddings_result[0]
                    final_reply += f"Embeddings data suggest {embeddings_result}.\n\n"
            
                return final_reply if final_reply else "Sorry, I couldn't find any information to answer your question."

            # Case recommendation: Give recommendations stemming from given movies
            elif question_type in self.question_types[0:2] or intent == "recommendation request": # Zero Shot found recommendation question
                user_ratings = [(title, 5.0) for title in movie_titles]
                recommendations = self.recommender.get_recommendations_for_user(user_ratings, debug=debug)
                
                if not recommendations:
                    return "Sorry, I don't know the ratings for these movies. " +\
                        "Are there other movies you like?"
                
                return f"Based on your liked movies, I think you might also like {', '.join(recommendations[:-1])}," +\
                    f" and {recommendations[-1]}."
            


    def find_answer_with_queries(self, movie_title: str, intent: str, debug=False) -> List[str]:
        """
        Tries to find an answer using graph queries.
        
        Parameters:
        movie_title: The title of the movie extracted from the user's question.
        intent: The intent or type of information the user is asking for, related to the movie.
        
        Returns:
        A list of strings containing the answers found using queries.
        """
        # Query the graph to get the requested information
        info_uri = self.graph_agent.get_entity_info_uri(movie_title, intent)
        if not info_uri:
            return ""

        # Check if the result is a URI or a literal
        info_label = []
        for uri in info_uri:
            if "http" in uri:
                # It's a URI, get the human-readable label
                label = self.graph_agent.get_entity_names([uri])
                info_label.extend(label if label else [uri])
            else:
                # It's a literal, use it directly
                info_label.append(uri)

        return info_label
    
    
    def find_answer_in_crowdsourced_data(self, movie_title: str, intent: str, debug=False) -> tuple[bool, str, str]:
        """Look for an answer using the crowdsourced data agent.

        Args:
            movie_title (str): movie title
            intent (str): user question type
            debug (bool, optional): debugging flag. Defaults to False.

        Returns:
            str: Pre-composed partial reply for the user regarding crowd-sourced data
        """
        # Get KG Triplet for Crowdsourcing data comparison
        info_uri = self.graph_agent.get_entity_info_uri(movie_title, intent)     
        info_uri_single = info_uri[0].split('/')[-1] if info_uri else None
            
        movie_uri = self.graph_agent.get_film_uri(movie_title).split('/')[-1]
        intent_property = self.graph_agent.get_intent(intent).split('/')[-1]
        
        triple = (movie_uri, intent_property, info_uri_single)
        if debug:
            print("KG tuple/triple matched for crowdsourced agent: ", triple)
            
        reply = ""
        
        # If info_uri (answer) found in graph, check for correctness in crowd data
        if info_uri:
            is_kg_correct, fix_triple = self.crowd_data_agent.check_and_correct_graph_ans(triple)
            fix_label = fix_triple[2] if fix_triple else None
            
            # Convert fix label to label if it isn't a date or numerical value
            if fix_label and not fix_label[0].isdigit():
                fix_label = self.graph_agent.get_entity_names([fix_label])
            
            if is_kg_correct:
                reply = "The crowd-sourced data confirm the above answer."
                
            elif is_kg_correct is False:
                reply = "On the other hand, crowd-sourced data suggest that this is wrong"
                if fix_label:
                    reply += f" and it actually is {fix_label}.\n\n"
                else:
                    reply += f", but no correct answer has been given.\n\n"
                # TODO add agreement, distrib
                    
        # If info_uri (answer) not found in KG, look it up in crowd data
        else:
            crowd_ans_id = self.crowd_data_agent.find_crowd_ans(triple[:2])
            
            if crowd_ans_id:
                if crowd_ans_id[0].isdigit():
                    crowd_ans_label = crowd_ans_id
                else:
                    crowd_ans_label = self.graph_agent.get_entity_names([crowd_ans_id])
                    
                reply = f"The crowd-sourced data suggest {crowd_ans_label}."
                
            # TODO add agreement, distrib
        
        if debug:
            pass
                
        return reply


    def find_answer_with_embeddings(self, movie_name: str, intent: str) -> str:
        """
        Attempts to find an answer to a user's question by using embeddings to identify the most 
        relevant entity or relation in the knowledge graph.

        Parameters:
        movie_name: The name of the movie as extracted from the user's question.
        intent: The intent or type of information the user is asking for, related to the movie.

        Returns:
        A string containing the natural language answer to the user's question.
        """
        # Get the URI for the movie name
        movie_uris = self.graph_agent.get_entities_urls(movie_name)
        if not movie_uris:
            return None

        # Assuming the first URI is the correct one for simplicity
        movie_uri = movie_uris[0]

        # Get the embedding for the movie URI
        try:
            movie_emb = self.embeddings_agent.get_entity_embedding(movie_uri)
        except ValueError as e:
            return str(e)

        # Get the corresponding entity URI for the intent
        intent_uris = self.graph_agent.get_entities_urls(intent)
        if not intent_uris:
            return None

        # Use the first URI as the relation URI
        relation_uri = intent_uris[0]

        # Get the embedding for the relation
        try:
            relation_emb = self.embeddings_agent.get_relation_embedding(relation_uri)
        except ValueError as e:
            return None

        # Predict the tail entity using the embeddings
        tail_uris = self.embeddings_agent.predict_missing_component(head_emb=movie_emb, relation_emb=relation_emb)
        if not tail_uris:
            return None

        # Handle multiple tail URIs (if any)
        tail_info = []
        for uri in tail_uris:
            names = self.graph_agent.get_entity_names([uri])
            if names:
                tail_info.extend(names)
        print(tail_info)
        if not tail_info:
            return None
        
        return tail_info



    def generate_answer_sentence(self, asked_info: str, film_name: str, query_names: List[str]) -> str:
        """
        Create a response sentence based on the type of information requested, the movie name, and the query results.
        
        Parameters:
        asked_info: A string representing the type of information requested (e.g., "director").
        film_name: A string representing the name of the film.
        query_names: A list of strings representing the query results to be included in the answer.
        """

        if len(query_names) > 1:
            query_names_str = ', '.join(query_names[:-1]) + ' and ' + query_names[-1]
            plural = 's'
            is_are = 'are'
        else:
            query_names_str = query_names[0]
            plural = ''
            is_are = 'is'
        
        sentence_templates = {
            "director": f"The director{'' if is_are == 'is' else 's'} of {film_name} {is_are} {query_names_str}.",
            "screenwriter": f"The screenwriter{'' if is_are == 'is' else 's'} of {film_name} {is_are} {query_names_str}.",
            "producer": f"The producer{'' if is_are == 'is' else 's'} of {film_name} {is_are} {query_names_str}.",
            "executive producer": f"The executive producer{'' if is_are == 'is' else 's'} of {film_name} {is_are} {query_names_str}.",
            "director of photography": f"The director of photography for {film_name} {is_are} {query_names_str}.",
            "original language": f"{film_name} was originally filmed in {query_names_str}.",
            "genre": f"{film_name} falls into the {query_names_str} genre.",
            "actors": f"The actors in {film_name} include {query_names_str}.",
            "costume designer": f"The costume designer{'' if is_are == 'is' else 's'} for {film_name} {is_are} {query_names_str}.",
            "distributed by": f"{film_name} was distributed by {query_names_str}.",
            "film editor": f"The film editor{'' if is_are == 'is' else 's'} for {film_name} {is_are} {query_names_str}.",
            "nominated for": f"{film_name} was nominated for {query_names_str}.",
            "based on": f"{film_name} is based on {query_names_str}.",
            "country of origin": f"{film_name} was filmed in {query_names_str}.",
            "narrative location": f"The story of {film_name} is set in {query_names_str}.",
            "filming location" : f"{film_name} was filmed at {query_names_str}.",
            "award received": f"{film_name} has received the award{'' if is_are == 'is' else 's'}: {query_names_str}.",
            "release year": f"{film_name} was released on the {self.nlp_processor.format_date(query_names_str)}.",
            "box office": f"The Box Office of {film_name} is {query_names_str}.",
            "film rating": f"The film rating of {film_name} is {query_names_str}."
        }

        return sentence_templates.get(asked_info, f"Sorry, I don't have information on {asked_info} for {film_name}.")