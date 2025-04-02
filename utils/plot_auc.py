import matplotlib.pyplot as plt
from sklearn import metrics
fig, ax = plt.subplots(figsize=(14, 8))
fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test).argmax(axis=-1))
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Trading model')
display.plot(ax=ax)
ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.show()