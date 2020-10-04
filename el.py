from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import NERBase, Span
import spacy


class MD_Module(NERBase):
    def __init__(self):
        pass

    def predict(self, sentence_nlp):
        mentions = []
        for ent in sentence_nlp.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE']:
                mentions.append(Span(ent.text, ent.start_char, ent.end_char, 0, ent.label_))
        print(mentions)
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

    result = process_results(mentions_dataset, predictions, input_text)
    return result

