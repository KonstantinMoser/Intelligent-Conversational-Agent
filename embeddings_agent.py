import numpy as np
from typing import Optional, Dict, Tuple
from rdflib import URIRef
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

class EmbeddingsAgent:
    """
    Manages embeddings for entities and relations and provides functionality
    to retrieve embeddings and predict missing components in RDF triplets.
    """
    
    def __init__(self, entity_emb: np.ndarray, relation_emb: np.ndarray, 
                 ent2id: Dict[URIRef, int], id2ent: Dict[int, URIRef], 
                 rel2id: Dict[URIRef, int], id2rel: Dict[int, URIRef]):
        """
        Initializes the Embeddings class with pre-loaded entity and relation embeddings and their corresponding mappings.
        
        Parameters:
        entity_emb: NumPy array of entity embeddings.
        relation_emb: NumPy array of relation embeddings.
        ent2id: Dictionary mapping entity URIRefs to their corresponding index in the entity embeddings.
        id2ent: Dictionary mapping indices to their corresponding entity URIRefs.
        rel2id: Dictionary mapping relation URIRefs to their corresponding index in the relation embeddings.
        id2rel: Dictionary mapping indices to their corresponding relation URIRefs.
        """
        self.entity_emb = entity_emb
        self.relation_emb = relation_emb
        self.ent2id = ent2id
        self.id2ent = id2ent
        self.rel2id = rel2id
        self.id2rel = id2rel

    def get_entity_embedding(self, entity_uri: str) -> np.ndarray:
        """
        Retrieves the embedding for a given entity URI.
        
        Parameters:
        entity_uri: The uri of the entity for which to retrieve the embedding.
        
        Returns:
        The embedding of the entity as a NumPy array.
        """
        
        entity_id = self.ent2id.get(URIRef(entity_uri))
        if entity_id is not None:
            return self.entity_emb[entity_id]
        else:
            raise ValueError(f"Entity '{entity_uri}' not found in the mapping.")

    def get_relation_embedding(self, relation_uri: str) -> np.ndarray:
        """
        Retrieves the embedding for a given relation URI.
        
        Parameters:
        relation_uri: The uri of the relation for which to retrieve the embedding.
        
        Returns:
        The embedding of the relation as a NumPy array.
        """
        relation_id = self.rel2id.get(URIRef(relation_uri))
        if relation_id is not None:
            return self.relation_emb[relation_id]
        else:
            raise ValueError(f"Relation '{relation_uri}' not found in the mapping.")

    def predict_missing_component(self, head_emb: Optional[np.ndarray] = None, 
                                  tail_emb: Optional[np.ndarray] = None, 
                                  relation_emb: Optional[np.ndarray] = None,
                                  similarity_threshold: float = 0.86) -> URIRef:
        """
        Predicts the missing component (head, relation, or tail) of an RDF triplet given the other two components.
        
        Parameters:
        head_emb: The embedding of the head entity if available; otherwise, None.
        tail_emb: The embedding of the tail entity if available; otherwise, None.
        relation_emb: The embedding of the relation if available; otherwise, None.
        
        Returns:
        The URIRef of the predicted missing component.
        """
        if head_emb is None:
            query_vector = tail_emb - relation_emb
            similarities = cosine_similarity(np.atleast_2d(query_vector), self.entity_emb)[0]
            # print(similarities)
            close_indices = np.where(similarities >= similarity_threshold)[0]
            if close_indices.any:
                return [self.id2ent[index] for index in close_indices]
            else:
                distances = pairwise_distances(np.atleast_2d(query_vector), self.entity_emb)
                predicted_index = np.argmin(distances)
                return self.id2ent[predicted_index]
        
        if tail_emb is None:
            query_vector = head_emb + relation_emb
            similarities = cosine_similarity(np.atleast_2d(query_vector), self.entity_emb)[0]
            # print(similarities)
            close_indices = np.where(similarities >= similarity_threshold)[0]
            if close_indices.any:
                return [self.id2ent[index] for index in close_indices]
            else:
                distances = pairwise_distances(np.atleast_2d(query_vector), self.entity_emb)
                predicted_index = np.argmin(distances)
                return self.id2ent[predicted_index]

        elif relation_emb is None:
            query_vector = tail_emb - head_emb
            similarities = cosine_similarity(np.atleast_2d(query_vector), self.entity_emb)[0]
            # print(similarities)
            close_indices = np.where(similarities >= similarity_threshold)[0]
            if close_indices.any:
                return [self.id2ent[index] for index in close_indices]
            else:
                distances = pairwise_distances(np.atleast_2d(query_vector), self.entity_emb)
                predicted_index = np.argmin(distances)
                return self.id2ent[predicted_index]


        else:
            raise ValueError("Exactly one of head_emb, tail_emb, or relation_emb must be None.")
