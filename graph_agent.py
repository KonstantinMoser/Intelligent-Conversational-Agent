from rdflib import Graph
from typing import List, Dict

class GraphAgent:
    """Provides SPARQL query functionalities for an RDF graph, focusing on movie data retrieval from Wikidata. 
    It supports fetching entity URLs, details by label, and names from IDs, using a predefined property mapping."""

    
    SPARQL_PREFIXES = '''
    PREFIX wd: <http://www.wikidata.org/entity/>   
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>   
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    '''
    
    def __init__(self, graph: Graph, property_nr: Dict[str, str], images_database):
        """
        Initialize the GraphAgent with an RDF graph and a mapping of properties to Wikidata property numbers.
        
        Parameters:
        graph: An rdflib.Graph instance, representing the RDF graph to be queried.
        property_nr: A dictionary mapping film-related properties to Wikidata property numbers.
        images_database: Database with information for movienet images
        """
        self.graph = graph
        self.property_nr = property_nr
        self.images_database = images_database

    def run_sparql_query(self, query_body: str) -> List[str]:
        """
        Execute a SPARQL query on the RDF graph and return the results as a list of strings.
        
        Parameters:
        query_body: A string containing the SPARQL query to be executed.
        """
        sparql_query = self.SPARQL_PREFIXES + query_body
        try:
            result = self.graph.query(sparql_query)
            return [str(row[0]) for row in result]
        except Exception as e:
            print(f"An error occurred during SPARQL query: {e}")
            return []

    def get_entities_urls(self, entity_label: str) -> List[str]:
        """
        Retrieve the URLs of entities with a specific label from the RDF graph.
        
        Parameters:
        entity_label: A string representing the label of the entities to find.
        """
        query_body = f'''
        SELECT ?entity WHERE {{
            ?entity rdfs:label "{entity_label}"@en .
        }}
        '''
        return self.run_sparql_query(query_body)

    def get_entity_info_uri(self, entity_label: str, info_type: str) -> List[str]:
        """
        Get specific information about a film entity from the RDF graph based on the entity's label and type.
        This method checks for various types of film entities including traditional films, anime films, documentaries, etc.
        
        Parameters:
        entity_label: A string representing the label of the entity.
        info_type: A string representing the type of information to retrieve, corresponding to a key in property_nr.
        """
        if info_type not in self.property_nr:
            raise ValueError(f"Info type '{info_type}' is not recognized.")

        film_types = [
            "Q11424",  # Film
            "Q20650540", # Anime Film
            "Q202866",  # Animated Film
            "Q93204",   # Documentary Film
            "Q24862",   # Short Film
            "Q506240",  # Television Film
            "Q226730",  # Silent Film
            "Q29168811", # Animated Feature Film
            "Q229390"   # 3D Film

        ]
        film_types_values = ' '.join([f"wd:{film_type}" for film_type in film_types])

        query_body = f'''
        SELECT ?x WHERE {{
            ?movie rdfs:label "{entity_label}"@en ;
                wdt:P31 ?filmType ;
                wdt:{self.property_nr[info_type]} ?x .
            VALUES ?filmType {{ {film_types_values} }}
        }}
        '''
        return self.run_sparql_query(query_body)
    

    def get_film_uri(self, film_name: str) -> str:
        """
        Get the URI of a film.

        Parameters:
        film_name: Film Name.
        """
        film_types = [
            "Q11424",    # Film
            "Q20650540", # Anime Film
            "Q202866",   # Animated Film
            "Q93204",    # Documentary Film
            "Q24862",    # Short Film
            "Q506240",   # Television Film
            "Q226730",   # Silent Film
            "Q29168811", # Animated Feature Film
            "Q229390"    # 3D Film
        ]
        film_types_values = ' '.join([f"wd:{film_type}" for film_type in film_types])

        query_body = f'''
        SELECT ?movie WHERE {{
            ?movie rdfs:label "{film_name}"@en ;
                wdt:P31 ?filmType .
            VALUES ?filmType {{ {film_types_values} }}
        }}
        '''
        results = self.run_sparql_query(query_body)
        return results[0] if results else None


    def get_entity_names(self, entity_uris: List[str]) -> List[str]:
        """
        Convert a list of Wikidata entity URIs to their labels.
        
        Parameters:
        entity_uris: A list of strings where each string is a URI representing a Wikidata entity.
        """
        names = []
        for id in entity_uris:
            id_str = id.split('/')[-1]  # Assumes 'id' is already a string
            query_body = f'''
            SELECT ?name WHERE {{
                wd:{id_str} rdfs:label ?name .
                FILTER(LANG(?name) = "en")
            }}
            '''
            names.extend(self.run_sparql_query(query_body))
        return names
    
    def get_imdbid(self, person_name: str) -> List[str]:
        """
        Retrieve the IMDbID (property P345) of an entity from the RDF graph based on their name.
        
        Parameters:
        person_name: A string representing the name of the entity.
        
        Returns:
        A list with the string of the IMDbID of the entity
        """
        query_body = f'''
        SELECT ?imdbid WHERE {{
            ?person rdfs:label "{person_name}"@en ;
                    wdt:P345 ?imdbid .
        }}
        '''
        return self.run_sparql_query(query_body)[0]
    
    def find_person_images(self, imdb_id, amount, preferred_types=[]):
        # Initialize a list to store the filtered images based on preferred types
        filtered_images = []

        # Look for images of the specified types first
        for image in self.images_database:
            if 'cast' in image and len(image['cast']) == 1 and imdb_id in image['cast']:
                if image['type'] in preferred_types:
                    filtered_images.append(image['img'])

        # If no images of preferred types are found, look for any type
        if not filtered_images:
            filtered_images = [
                image['img'] for image in self.images_database
                if 'cast' in image and len(image['cast']) == 1 and imdb_id in image['cast']
            ]

        # Generate the image chat codes without the .jpg extension
        image_codes = [f"image:{img_path.rsplit('.', 1)[0]}" for img_path in filtered_images[:amount]]

        return " ".join(image_codes)
        
    def find_movie_images(self, movie_imdb_id, amount, preferred_types=[]):
        # Initialize a list to store the filtered images based on preferred types
        filtered_images = []

        # Look for images of the specified types first
        for image in self.images_database:
            if movie_imdb_id in image['movie']:
                if image['type'] in preferred_types:
                    filtered_images.append(image['img'])

        # If no images of preferred types are found, look for any type
        if not filtered_images:
            filtered_images = [
                image['img'] for image in self.images_database
                if movie_imdb_id in image['movie']
            ]

        # Generate the image chat codes without the .jpg extension
        image_codes = [f"image:{img_path.rsplit('.', 1)[0]}" for img_path in filtered_images[:amount]]

        return " ".join(image_codes)
    
    def get_intent(self, intent):
        return self.property_nr[intent]