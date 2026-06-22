🫁 LungAllytics

Two Independent AI Tools for COPD Assessment

LungAllytics is a research-oriented machine learning project for COPD assessment,
combining clinical/spirometry data and chest X-ray imaging. It deliberately
keeps the two modalities as independent, separately validated tools rather
than a fused ensemble — see "Why not a fused ensemble?" below.

🔗 Live demo: https://huggingface.co/spaces/Aishwarya-2006/copd-insight


🚀 Key Features


Clinical severity staging from symptoms/vitals (voting ensemble)
Chest X-ray screening for COPD-related abnormality (ResNet50 CNN)
A second imaging pipeline exploring PCA-based dimensionality reduction on
ResNet50 features, classified with XGBoost
Explainable AI via Grad-CAM heatmaps on the imaging model
Deployed as a Flask web app, containerized with Docker, hosted on Hugging
Face Spaces



🧠 Models Used


Clinical severity model → Voting ensemble (Random Forest + KNN + XGBoost), SMOTE for class imbalance — 91% test accuracy on Mild/Moderate vs. Severe/Very Severe staging
Imaging model (fine-tuned) → ResNet50, end-to-end fine-tuning — 97%+ test accuracy on Normal vs. Affected X-ray classification
Imaging model (PCA pipeline) → ResNet50 (frozen feature extractor) → PCA (256 components) → XGBoost — 97.6% accuracy, 0.998 AUC-ROC; paired with Grad-CAM for interpretability



⚠️ Why not a fused ensemble?

An earlier version of this project attempted to fuse the clinical and imaging
model outputs into a single probability score. That fusion produced only 64%
accuracy — worse than either model alone. Investigation showed why: the
clinical model predicts COPD severity among already-diagnosed patients,
while the imaging model predicts disease presence from an X-ray — two
different questions, and the two source datasets (230 clinical records, ~1,500
X-ray images) have no shared patient IDs to begin with. Rather than force an
invalid fusion, the two models are presented as independent diagnostic tools.


⚙️ Core Technologies

Python · scikit-learn · XGBoost · TensorFlow/Keras · imbalanced-learn (SMOTE) ·
Flask · Docker · Hugging Face Spaces


🔬 Methodology


Clinical severity staging via a heterogeneous voting ensemble, with SMOTE
to address class imbalance
Chest X-ray feature extraction and classification via transfer learning
on ResNet50 (both fine-tuned and frozen-feature-extractor variants)
Dimensionality reduction via PCA on ResNet50 feature vectors
Explainable AI visualization via Grad-CAM
Deployment as a containerized Flask web app on Hugging Face Spaces
