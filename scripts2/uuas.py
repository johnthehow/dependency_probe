def report_uuas_and_tikz(self, prediction_batches, dataset, split_name):
    """Computes the UUAS score for a dataset and writes tikz dependency latex.

    From the true and predicted distances, computes a minimum spanning tree
    of each, and computes the percentage overlap between edges in all
    predicted and gold trees.

    For the first 20 examples (if not the test set) also writes LaTeX to disk
    for visualizing the gold and predicted minimum spanning trees.

    All tokens with punctuation part-of-speech are excluded from the minimum
    spanning trees.

    Args:
        prediction_batches: A sequence of batches of predictions for a data split
        dataset: A sequence of batches of Observations
        split_name the string naming the data split: {train,dev,test}
    """
uspan_total = 0
uspan_correct = 0
total_sents = 0
for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in tqdm(zip(prediction_batches, dataset), desc='[uuas,tikz]'):
    for prediction, label, length, (observation, _) in zip(
        prediction_batch, label_batch,
        length_batch, observation_batch):
    words = observation.sentence
    poses = observation.xpos_sentence
    length = int(length)
    assert length == len(observation.sentence)
    prediction = prediction[:length,:length]
    label = label[:length,:length].cpu()

    gold_edges = prims_matrix_to_edges(label, words, poses)
    pred_edges = prims_matrix_to_edges(prediction, words, poses)

    if split_name != 'test' and total_sents < 20:
        self.print_tikz(pred_edges, gold_edges, words, split_name)

    uspan_correct += len(set([tuple(sorted(x)) for x in gold_edges]).intersection(
        set([tuple(sorted(x)) for x in pred_edges])))
    uspan_total += len(gold_edges)
    total_sents += 1
uuas = uspan_correct / float(uspan_total)
with open(os.path.join(self.reporting_root, split_name + '.uuas'), 'w') as fout:
    fout.write(str(uuas) + '\n')