from speakeasypy import Speakeasy, Chatroom
from rdflib.namespace import Namespace, RDF, RDFS, XSD
from rdflib.term import URIRef, Literal
from rdflib.graph import Graph
from typing import List
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import spacy
import re
import os
#from spacy_entity_linker import EntityLinker
from spacy.language import Language

from transformers import BertTokenizer, BertModel, pipeline
import torch
#from sparql import SPARQL


import spacy
import csv
import rdflib
import json

# Import Classes
from graph_agent import GraphAgent
from nlp_processor import NLPProcessor
from response_generator import ResponseGenerator
from embeddings_agent import EmbeddingsAgent
from recommender import Recommender
from crowdsourced_data_agent import CrowdsourcedDataAgent



DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

# kindle-animato-marinade_bot

class Agent:

    def __init__(self, username, password):
        print("Initialized")
        print(os.getcwd())
        self.username = username
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.


        self.property_nr = {
        "director": "P57",
        "screenwriter": "P58",
        "release year": "P577",
        "producer": "P272",
        "executive producer": "P1431",
        "director of photography": "P344",
        "original language": "P364",
        "genre": "P136",
        "actors": "P161",
        "costume designer": "P2515",
        # "distributed by": "P750",
        "film editor": "P1040",
        "nominated for": "P1411",
        "based on": "P144",
        "country of origin": "P495",
        "filming location": "P915",
        "narrative location": "P840",
        "award received": "P166",
        "box office": "P2142",
        "film rating": "P1657"
        }

        self.nlp = spacy.load("en_core_web_md")
        self.movie_titles = self.read_movie_titles('data/movie_names.txt')
        # self.movie_embeddings = {title: self.nlp(title).vector for title in movie_titles}
        self.people_names = self.read_people_names('data/people_names.txt')
    
        self.bart_pipe = pipeline(model="facebook/bart-large-mnli")

        self.candidate_labels=[
            "director",
            "screenwriter",
            "release year",
            "producer",
            "director of photography",
            "original language",
            "genre",
            "actors",
            "costume designer", 
            # "distributed by",
            "film editor",
            "nominated for",
            "based on",
            "country of origin", 
            "filming location", 
            # "narrative location", 
            "award received",
            "box office",
            "film rating",
            
            # # For recommender
            # "recommendation",
            "recommendation request"
            # "liked movies",

            # # For multimedia questions
            # "show image"
        ]

        self.question_types = [
    "liked movies",
    "recommend movies",
    "asking for specific information regarding a movie",
    "show picture of an actor",
    "show an image or scene",
    "show movie poster",
    "show behind the scenes"
]

        self.multimedia_types = [
        "demand to show a scene from a movie",
        "demand to show the image of a movie poster",
        "demand to show an behind the scenes image of a movie",
        "demand to show an image of an actor"
    ]


        self.entity_emb = np.load('data/ddis-graph-embeddings/entity_embeds.npy')
        self.relation_emb = np.load('data/ddis-graph-embeddings/relation_embeds.npy')

        # Load mapping dictionaries
        with open('data/ddis-graph-embeddings/entity_ids.del', 'r') as ifile:
            ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
            id2ent = {v: k for k, v in ent2id.items()}
        with open('data/ddis-graph-embeddings/relation_ids.del', 'r') as ifile:
            rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
            id2rel = {v: k for k, v in rel2id.items()} 

        print("Reading graph...")
        self.graph = Graph()
        self.graph.parse('./data/14_graph.nt', format='turtle')
        print("Graph read.")

        with open("data/movienet/images.json", 'r') as file:
            self.images_database = json.load(file)


        self.graph_agent = GraphAgent(self.graph, self.property_nr, self.images_database)
        self.nlp_processor = NLPProcessor(self.movie_titles, self.people_names, self.bart_pipe, self.nlp, self.candidate_labels, self.question_types, self.multimedia_types)
        self.embeddings_agent = EmbeddingsAgent(self.entity_emb, self.relation_emb, ent2id, id2ent, rel2id, id2rel)
        self.recommender = Recommender('data/ml-latest-small/ratings.csv', 'data/ml-latest-small/movies.csv')
        self.crowd_data_agent = CrowdsourcedDataAgent('./data/crowd_data')

        self.response_generator = ResponseGenerator(self.graph_agent, self.nlp_processor, self.embeddings_agent, 
                                                    self.recommender, self.crowd_data_agent, self.question_types, self.multimedia_types)


    def listen(self):
        while True:
            # only check active chatrooms (i.e., remaining_time > 0) if active=True.
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    # send a welcome message if room is not initiated
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True
                # Retrieve messages from this chat room.
                # If only_partner=True, it filters out messages sent by the current bot.
                # If only_new=True, it filters out messages that have already been marked as processed.
                for message in room.get_messages(only_partner=True, only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new message #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")

                    # room.post_messages("Working on your query...")                    
                    reply = None
                    try:
                        reply = self.response_generator.generate_response(message.message, debug=True)

                        print("Posted Answer: " + reply)
                    except Exception as e:
                        print("ERROR: ", e)
                    
                    if not reply:
                        reply = "Sorry, I'm not able to reply to this question. Can I help you with something else?"
                    
                    # Send a message to the corresponding chat room using the post_messages method of the room object.
                    room.post_messages(reply)
                    
                    # Mark the message as processed, so it will be filtered out when retrieving new messages.
                    room.mark_as_processed(message)

                # Retrieve reactions from this chat room.
                # If only_new=True, it filters out reactions that have already been marked as processed.
                for reaction in room.get_reactions(only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

                    room.post_messages(f"Thanks for your '{reaction.type}' :D")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())
    
    
    def read_movie_titles(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read all lines in the file and strip newline characters
            movie_titles = [line.strip() for line in file.readlines()]
        return movie_titles
    
    def read_people_names(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read all lines in the file and strip newline characters
            people_names = [line.strip() for line in file.readlines()]
        return people_names


if __name__ == '__main__':
    marinade_bot = Agent("kindle-animato-marinade_bot", "SIHszdFkyiFWTg")
    marinade_bot.listen()

# kindle-animato-marinade_bot