# Deploying the Pneumonia demo on Google Cloud Run

Folder layout (all in one folder):

```
app.py
requirements.txt
Dockerfile
.dockerignore
pneumonia_densenet121.keras   <- downloaded from Colab after training
threshold.txt                 <- downloaded from Colab after training
```

## 0. Match the TensorFlow version (important)
In Colab, run `import tensorflow as tf; print(tf.__version__)` and pin that version in
`requirements.txt` (for example `tensorflow-cpu==2.19.0`). A model saved with one Keras
version can fail to load on another.

## 1. Test locally first
```bash
pip install -r requirements.txt
streamlit run app.py
```
Upload a chest X-ray and check that the prediction and Grad-CAM overlay appear.

## 2. One-time Google Cloud setup
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```
(Billing must be enabled on the project. Check current Cloud Run pricing and free tier on Google's site.)

## 3. Deploy (builds the container for you)
Run from the folder that contains the Dockerfile:
```bash
gcloud run deploy pneumonia-demo \
  --source . \
  --region asia-south1 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --allow-unauthenticated
```
The command prints a public URL like `https://pneumonia-demo-xxxxx.a.run.app`. Put it on your
resume and in your GitHub README.

- `--memory 2Gi`: TensorFlow plus DenseNet121 does not fit in the 512 MiB default.
- `--min-instances 0`: you only pay while it is serving; the first visit after idle has a cold start (about 10-30 s). Open the link yourself before an interview.

## 4. Update after changes
Re-run the same `gcloud run deploy ...` command.

## 5. Delete when you no longer need it
```bash
gcloud run services delete pneumonia-demo --region asia-south1
```

## Optional: load the model from Cloud Storage instead of baking it into the image
```bash
gsutil mb -l asia-south1 gs://YOUR_BUCKET
gsutil cp pneumonia_densenet121.keras gs://YOUR_BUCKET/
```
Then in `app.py` download it at startup with `google-cloud-storage` (add it to requirements).
Baking the model into the image (as the Dockerfile does) is simpler and fine for a ~30 MB model.

## README checklist (what recruiters look for)
- One-line description plus the "screening demo, not a diagnostic tool" note
- Live demo link and a screenshot of the app
- Test recall, precision, ROC-AUC and the confusion matrix
- The MLflow comparison table screenshot
- Dataset source and a short "limitations and next steps" section
