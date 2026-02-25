
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(description: str):
    """
    Extract entities from description using spaCy.
    Returns a dict of lists: PERSON, GPE, ORG, DATE, etc.
    """
    doc = nlp(description or "")
    entities = {"PERSON": [], "GPE": [], "ORG": [], "DATE": [], "MISC": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
        else:
            entities["MISC"].append(ent.text)
    return entities

def get_keywords(description: str):
    """
    Produce a simplified keywords string for DB search.
    """
    doc = nlp(description or "")
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)
