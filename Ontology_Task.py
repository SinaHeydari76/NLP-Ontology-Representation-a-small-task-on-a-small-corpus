import spacy
import glob
from nltk.corpus import wordnet as wn
from spacy_cleaner import Cleaner
from spacy_cleaner.processing import mutators, removers
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Ontology:
    def __init__(self, path):
        
        self.path = path
        self.language_model = spacy.load("en_core_web_sm")
        
        # self.graph = nx.MultiDiGraph()
        self.graph = nx.DiGraph()
        # self.graph = nx.Graph()
        
        self.corpus_uniques = {}
        self.corpus_dict = {}
        self.entity_list = []
        self.corpus_list = []
        
        self.lemma = Cleaner(self.language_model,
                             removers.remove_number_token,
                             removers.remove_punctuation_token,
                             mutators.mutate_lemma_token)

        # Read and clean the text file content
        with open(path, 'r', encoding="latin-1") as file:
            content = file.read()
            # Remove unwanted characters and normalize whitespace
            content = content.replace('\r', '') \
                             .replace('\n', ' ') \
                             .replace('\x92', '') \
                             .replace('\x93', '') \
                             .replace('\x94', '') \
                             .replace('\x95', '') \
                             .replace('\x96', '') \
                             .replace('\x97', '') \
                             .replace('\x98', '') \
                             .replace('?', ' ') \
                             .replace('!', ' ') \
                             .replace('.', ' ') \
                             .replace(':', ' ') \
                             .replace(';', ' ')

            self.corpus_dict[0] = content

    def preprocess(self):
        # Convert text to lowercase
        for key in self.corpus_dict:
            self.corpus_dict[key] = self.corpus_dict[key].lower()

        # Clean text using spacy_cleaner
        for key in self.corpus_dict:
            self.corpus_dict[key] = self.lemma.clean([str(self.corpus_dict[key])])

        # Store cleaned text in a list
        for key in self.corpus_dict:
            self.corpus_list.append(self.corpus_dict[key])

        # Process text with spaCy language model
        for i in range(len(self.corpus_list)):
            self.corpus_list[i] = self.language_model(str(self.corpus_list[i]))

        # Extract unique non-stop tokens and find their WordNet synsets
        for i in range(len(self.corpus_list)):
            tokens = self.corpus_list[i]
            for token in tokens:
                if not token.is_stop:
                    if str(token) not in self.corpus_uniques:
                        self.corpus_uniques[str(token)] = set(wn.synsets(str(token)))

    def build_graph(self):
        # Add unique tokens as nodes in the graph
        for node in self.corpus_uniques.keys():
            self.graph.add_node(node)
    
        nodes = list(self.corpus_uniques.keys())
        # Add edges between nodes based on relationships found in WordNet
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1 = nodes[i]
                node2 = nodes[j]
                synsets1 = self.corpus_uniques[node1]
                synsets2 = self.corpus_uniques[node2]
    
                relationship_found = False
                edge_attributes = {}
    
                # Check for direct synset overlap
                if synsets1 & synsets2:
                    relationship_found = True
                    edge_attributes['type'] = 'synset_overlap'
                else:
                    # Check for other meaningful relationships
                    for syn1 in synsets1:
                        for syn2 in synsets2:
                            if syn1 in syn2.hypernyms() or syn2 in syn1.hypernyms():
                                relationship_found = True
                                edge_attributes['type'] = 'hypernym'
                            elif syn1 in syn2.hyponyms() or syn2 in syn1.hyponyms():
                                relationship_found = True
                                edge_attributes['type'] = 'hyponym'
                            elif any(lemma in syn2.lemmas() for lemma in syn1.lemmas()):
                                relationship_found = True
                                edge_attributes['type'] = 'synonym'
                            elif any(antonym for lemma in syn1.lemmas()\
                                     for antonym in lemma.antonyms()\
                                         if antonym in syn2.lemmas()):
                                relationship_found = True
                                edge_attributes['type'] = 'antonym'
                            elif syn1 in syn2.part_meronyms() or syn2 in syn1.part_meronyms():
                                relationship_found = True
                                edge_attributes['type'] = 'meronym'
                            elif syn1 in syn2.part_holonyms() or syn2 in syn1.part_holonyms():
                                relationship_found = True
                                edge_attributes['type'] = 'holonym'
    
                            if relationship_found:
                                self.graph.add_edge(node1, node2, **edge_attributes)
                                break
                        if relationship_found:
                            break

    def visualize_graph(self):
        # Remove nodes without any edges
        isolated_nodes = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated_nodes)
    
        # Define layout for graph visualization
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        plt.figure(figsize=(15, 15))
    
        # Define edge colors based on relationship types
        edge_colors = []
        edge_color_map = {
            'synset_overlap': 'black',
            'hypernym': 'blue',
            'hyponym': 'green',
            'synonym': 'purple',
            'antonym': 'red',
            'meronym': 'orange',
            'holonym': 'brown'
        }
        
        for _, _, data in self.graph.edges(data=True):
            edge_colors.append(edge_color_map.get(data['type'], 'gray'))
        
        # Draw the graph with labels, colors, and sizes
        nx.draw(self.graph, pos, with_labels=True, node_size=2500,
                node_color="yellow", font_size=12, font_weight="bold", edge_color=edge_colors)
    
        # Create legend for edge colors
        legend_elements = [Line2D([0], [0], color=color, linewidth=4 , label=label)
                           for label, color in edge_color_map.items()]
        plt.legend(handles=legend_elements, loc='best')
    
        # Set the title and display the graph
        plt.title(f'Document {self.path} Entity Graph')
        plt.show()

# %% representing one document
document_list = glob.glob('*.txt')
print(len(document_list))

# %%
# represent = Ontology(document_list[0])
# represent.preprocess()
# represent.build_graph()
# represent.visualize_graph()

# %% representing all the documents(in seperate figures)
for document in document_list:
    represent = Ontology(document)
    represent.preprocess()
    represent.build_graph()
    represent.visualize_graph()

