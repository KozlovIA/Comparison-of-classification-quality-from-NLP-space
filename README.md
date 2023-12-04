# Comparative Analysis of Classification Quality Depending on Feature Systems

## Abstract

This thesis presents a study in the field of text data classification, focusing on the possibility of reducing the feature space without sacrificing classification quality. Two datasets for binary and multiclass classification, composed of scientific documents, are used for investigation.

The study explores machine learning algorithms such as logistic regression, k-nearest neighbors, decision trees, and random forests. It provides detailed descriptions of working with text data, separately examining binary and multiclass classification cases. Quality metrics and the Python programming language are employed in the analysis.

## Conclusion

In this research, binary and multiclass datasets were studied, visualized, classification tasks were formulated, classifiers were selected, their parameters were tuned, and a review of classification quality metrics was conducted.

Results on the Russian binary dataset and English multiclass datasets indicated the necessity for an individualized approach to tasks, justified selection of significant features, and careful parameter tuning.

The conclusion suggests the feasibility of using a reduced feature space in binary classification and the inadvisability in multiclass classification. For binary classification, the random forest method proved most accurate (Accuracy=0.87, f1-score=0.87 for the title space, and Accuracy=0.99, f1-score=0.99 for the Bag of Words space). In multiclass classification, results were more contentious; the random forest method performed best on the Bag of Words space (Accuracy=0.91, f1-score=0.90). However, it performed poorly on the reduced space (Accuracy=0.49, f1-score=0.48), where logistic regression excelled (Accuracy=0.60, f1-score=0.53), and on the Bag of Words space, logistic regression also performed well (Accuracy=0.86, f1-score=0.85).

Thus, results on the multiclass dataset degraded by approximately 9-40% compared to the binary dataset.

