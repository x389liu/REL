from REL.REL.mention_detection import MentionDetection
from REL.REL.utils import process_results
from REL.REL.entity_disambiguation import EntityDisambiguation
from REL.REL.ner import NERBase, Span
#import spacy

class MD_Module(NERBase):
    def __init__(self):
        pass

    def predict(self, sentence_nlp):
        mentions = []
        for ent in sentence_nlp.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE']:
                mentions.append(Span(ent.text, ent.start_char, ent.end_char, 0, ent.label_))
        return mentions


# input_text: {'docid':sentence_nlp}
def entity_linking(input_sentence_nlp):
    base_url = '/home/x389liu/projects/def-jimmylin/x389liu/totalRelationRecall/REL/data'
    wiki_version = "wiki_2019"
    mention_detection = MentionDetection(base_url, wiki_version)

    tagger_custom = MD_Module()
    mentions_dataset, n_mentions = mention_detection.find_mentions(input_sentence_nlp, tagger_custom)
    config = {
        "mode": "eval",
        "model_path": "/home/x389liu/projects/def-jimmylin/x389liu/totalRelationRecall/REL/data/ed-wiki-2019/model",
    }
    model = EntityDisambiguation(base_url, wiki_version, config)
    predictions, timing = model.predict(mentions_dataset)

    result = process_results(mentions_dataset, predictions, input_sentence_nlp)
    return result

#nlp = spacy.load('en_core_web_lg')
#text = nlp('File: Sean Eldridge, president of Hudson River Ventures, left, and Chris Hughes, editor-in-chief and publisher of The New Republic and a founder of Facebook Inc., stand for a photograph during the Paris Review Spring Revel gala in New York, U.S., on Tuesday, April 3, 2012. ')
#input_sentence_nlp = {'doc1':text}
#print(entity_linking(input_sentence_nlp))
