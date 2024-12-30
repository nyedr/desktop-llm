import spacy
from app.memory.lightrag.manager import LightRAGManager

nlp = spacy.load("en_core_web_trf")


def advanced_ner_and_relationship_inference(
    text: str,
    user_id: str,
    manager: LightRAGManager
):
    """Perform NER and relationship inference on text."""
    doc = nlp(text)
    for ent in doc.ents:
        ent_text = ent.text
        ent_label = ent.label_

        # Handle DOG/ANIMAL entities
        if ent_label.upper() in ["DOG", "ANIMAL"]:
            # Link entity
            e_id = manager.link_entity(ent_text, "ANIMAL")
            # Create relationship
            manager.create_relationship(user_id, e_id, "hasPet", 1.0)

        # Handle PERSON entities
        elif ent_label.upper() == "PERSON":
            e_id = manager.link_entity(ent_text, "PERSON")
            manager.create_relationship(user_id, e_id, "knows", 1.0)

        # Handle ORG entities
        elif ent_label.upper() == "ORG":
            e_id = manager.link_entity(ent_text, "ORG")
            manager.create_relationship(user_id, e_id, "worksAt", 1.0)

        # Add more entity types and relationship rules as needed
