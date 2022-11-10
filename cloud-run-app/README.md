## Deployment instructions GCP
In order to deploy a Cloud Run, run the following command in the directory containing the ```Dockerfile```:
```shell
gcloud builds submit --tag gcr.io/nlp-demo-313618/nlp-demo --project=nlp-demo-313618 &&\
gcloud run deploy nlp-demo --image gcr.io/nlp-demo-313618/nlp-demo \
--project=nlp-demo-313618 \
--platform managed --region=us-central1 \
--update-env-vars PROJECT_ID=nlp-demo-313618 \
--region=us-central1 --max-instances=1 --memory=1G --allow-unauthenticated

```
