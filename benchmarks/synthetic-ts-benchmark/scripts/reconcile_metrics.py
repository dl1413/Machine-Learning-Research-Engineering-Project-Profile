"""Recompute displayed portfolio arithmetic; no unsupported experiment repair."""
import json
import statistics

tn,fp,fn,tp=1876,55,25,544
folds=[96.4,97.2,95.8,96.1,97.0,95.6,96.2,95.9,96.5,95.4]
result={
    'confusion_matrix':{'tn':tn,'fp':fp,'fn':fn,'tp':tp},
    'accuracy':(tn+tp)/(tn+fp+fn+tp),
    'precision_harmful':tp/(tp+fp),
    'recall_harmful':tp/(tp+fn),
    'f1_harmful':2*tp/(2*tp+fp+fn),
    'fold_accuracy_mean_percent':statistics.mean(folds),
    'fold_accuracy_sample_sd_percentage_points':statistics.stdev(folds),
    'rating_count_from_table':4500*3,
    'reported_rating_count':67500,
    'unverified_missing_multiplier':67500/(4500*3),
    'status':'Arithmetic check only; original predictions and rating records required to resolve provenance.'
}
print(json.dumps(result,indent=2))
