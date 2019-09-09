import sys
from . import data_prep_utils
if sys.version < '3' :
    from backports import csv
else:
    import csv

def autoLabel(raw_strings, module, type):
    return set([tuple(module.parse(raw_sequence.strip(), type=type)) for i, raw_sequence in enumerate(set(raw_strings), 1)])

def label(module, infile, outfile, xml, type=None):
    training_data = data_prep_utils.TrainingData(xml, module)
    reader = csv.reader(infile)
    strings = set(row[0] for row in reader if len(row) > 0)
    if type is None:
        tagger = module.TAGGER
    else:
        tagger = module.TAGGERS[type] or module.TAGGER
    if tagger:
        labeled_list = autoLabel(strings, module, type)
    else:
        raise Exception("Tagger is not defined in %s" % module.__name__)
    training_data.extend(labeled_list)

    with open(outfile, 'wb'):
        training_data.write(outfile)

    print("Training data successfully created and stored in stored in %s" % outfile)
