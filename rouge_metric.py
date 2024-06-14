from korouge_score import rouge_scorer


default_rouge_scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)


def compute_metrics(predictions, references):
    min_length = min(len((predictions)), len(references))
    predictions = predictions[:min_length]
    references = references[:min_length]

    rougeLsum = 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        rougeLsum_scores_for_ground_truths = []
        for ground_truth in gold:
            score = default_rouge_scorer.score(prediction=pred, target=ground_truth)['rougeLsum'].fmeasure
            rougeLsum_scores_for_ground_truths.append(score)
        rougeLsum += max(rougeLsum_scores_for_ground_truths)

    rougeLsum = 100.0 * rougeLsum / len(references)
    metrics = {'rougeLsum': rougeLsum}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics
