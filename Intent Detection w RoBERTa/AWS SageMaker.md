# How To Move a Solution from a Jupyter Notebook to AWS SageMaker

### 1. AWS CLI (Command Line Interface v2)
Download CLI https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

```cli
C:\Users\fam2064>aws --version
```
```
aws-cli/2.24.17 Python/3.12.9 Windows/11 exe/AMD64
```

Configure AWS CLI
```cli
C:\Users\fam2064>aws configure
```
```
AWS Access Key ID [None]: AKIAWCYYA----------
AWS Secret Access Key [None]: 4WOcDwzj49QbR18wVV8uYkiT4f----------
Default region name [None]: us-east-1
Default output format [None]: json
```

### 2. Organize Your Files Locally
Before uploading to AWS, structure your project like this:
```
Sentiment-Analysis/
│── scripts/
│   ├── train.py       # Training script
│   ├── inference.py   # Inference script
│   ├── model.py       # Model class
│── dataset/
│   ├── train.csv      # Training dataset
│   ├── test.csv       # Test dataset
│── requirements.txt   # Python dependencies
│── notebook.ipynb     # Your Jupyter Notebook (for reference)
│── model/             # Store best performance model for inference
```

Convert your notebook to scripts
- Extract the training logic from your notebook into `train.py`.
- Extract the inference logic into `inference.py`.
- Extract the model class to `model.py` (used by both `train` and `inference`).

### 3. S3 Bucket
Create an S3 bucket (`s3-sentiment` is bucket name)
```
>aws s3 mb s3://s3-sentiment
```

List all S3 buckets
```
>aws s3 ls
```
```
2025-03-03 23:16:20 s3-sentiment
```

Upload file to S3
```
>aws s3 cp dataset/test.csv s3://s3-sentiment/dataset/test.csv
```

Verify that files were uploaded
```
>aws s3 ls s3://s3-sentiment
```
```
2025-03-04 17:15:29   13424637 test.csv
2025-03-04 17:15:50   13359396 train.csv
```

Move the file to the data subfolder (also create subfolder/prefix)
```
>aws s3 mv s3://s3-sentiment/train.csv s3://s3-sentiment/dataset/train.csv
```

### 4. IAM Role
Create an IAM Role for SageMaker (for the current AWS Access Key ID ?)
- Create a trust-policy.json file with this content:
```json
        { "Version": "2012-10-17", 
            "Statement": [ 
                { 
                    "Effect": "Allow", 
                    "Principal": { "Service": "sagemaker.amazonaws.com" }, 
                    "Action": "sts:AssumeRole" 
                } 
            ] 
        }
```
```
aws iam create-role --role-name SageMakerExecutionRole --assume-role-policy-document file://trust-policy.json
```

Attach Necessary Permissions to the Role
- Attach the managed SageMaker policy
```cli
aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```
- Attach an S3 policy
```
aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```
- Attach logging policy
```
aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
```
- Retrieve the Role ARN
```
aws iam get-role --role-name SageMakerExecutionRole
```
```
"Arn": "arn:aws:iam::418272790285:role/SageMakerExecutionRole"
```
- Copy the Role ARN and use it in SageMaker when creating training jobs.

### 5. Train.py

- Ensure your AWS credentials are configured (with `aws configure`)
- Specify model and label encoder variables
  + local path: name to save model and encoder at local
  + key: prefix and file name at S3
    ```python
    # AWS S3 Config
    bucket_name = "s3-intent"
    model_key = "models/model.pth"
    encoder_key = "models/label_encoder.pkl"
    # Local paths for saving model and label encoder
    model_local_path = "model.pth"
    encoder_local_path = "label_encoder.pkl"

    joblib.dump(label_encoder, encoder_local_path)
    s3_client.upload_file(encoder_local_path, bucket_name, encoder_key)

    torch.save({
        'model_state_dict': best_model_state,
        'accuracy': best_accuracy
    }, model_local_path)
    # Upload the trained model to S3
    s3_client.upload_file(model_local_path, bucket_name, model_key)
    ```
- We save label encoder to S3 during training so in the reference, we can load it and use to invert a number into an intent.
- We can run `train.py` locally, which download datasets from S3 bucket and then upload the best model to S3 bucket.

**Should I Train the Model Using a SageMaker Training Job or a Notebook?**
  1. Launch a SageMaker Training Job (Recommended)
  - Use this when 
    + you need GPU acceleration (e.g., ml.g4dn.xlarge)
    + you want auto-scaling & distributed training
    + you want to save costs (SageMaker jobs shut down automatically after training).
  - Monitor the job in AWS Console → SageMaker → Training Jobs
  - **For production use**

  2. Run `train.py` directly in a SageMaker notebook
  - Use this when:
    + you want to debug your training script before launching a job.
    + your dataset is small, and you don’t need distributed training.
    + you don’t mind keeping the notebook running manually.
  - Open SageMaker Notebook → Upload train.py and requirements.txt. Install and run.
    ```
    pip install -r requirements.txt
    !python train.py
    ```
  - **For quick debugging**

### 6. Inference.py
- We can run inference without using SageMaker by manually downloading the model from the S3 bucket and predict the intent of an input text.
- Use the appropriate I/O stream (BytesIO or StringIO) to ensure that the data is handled correctly based on its type.
  + `BytesIO` for Binary data: When dealing with binary files like PyTorch models (.pth, .pkl), images, or other non-text data. Proceed with torch.load(), joblib.load()
  ```python
  response = s3_client.get_object(Bucket=bucket_name, Key=key)
  model_stream = BytesIO(response["Body"].read())  # Read binary data into BytesIO
  model = torch.load(model_stream)  # Load binary data (e.g., a model)
  ```
  + `StringIO` for Text data: When dealing with text-based files like CSV, JSON, or plain text. Proceed with pd.read_csv(), json.load()
  ```python
  response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
  csv_data = response["Body"].read().decode("utf-8")  # Decode binary data to string
  df = pd.read_csv(StringIO(csv_data))  # Load text data (e.g., a CSV) into a DataFrame
  ```
- We can run `inference.py` locally, which download the model and label encoder from S3 bucket and use them to predict the intent of an input text.
```python
> python inference.py --text "I want to order a pizza"
```

- **Why we should use SageMaker?** A better choice for production-grade machine learning workflows
  + Scalability: Automatically scales the endpoint based on traffic if we configure auto-scaling policies to handle spikes in demand without manual intervention.
  + Managed Infrastructure: Fully managed infrastructure. SageMaker handles Provisioning and managing compute resources, Installing dependencies (via pre-built containers or custom images), Security updates and patches. The dev team focus on building models rather than managing infrastructure.
  + Endpoint Deployment and Monitoring: Provides built-in endpoint deployment and monitoring: Easy deployment of models as RESTful endpoints, Built-in monitoring with Amazon CloudWatch (e.g., latency, errors, resource utilization), Automatic retries and failure handling.
  + Cost Efficiency: SageMaker endpoints can scale down to zero when not in use (using Serverless Inference or Asynchronous Inference), reducing costs.
  + Security and Compliance: Built-in security features: Automatic encryption of data at rest and in transit, Integration with AWS IAM for fine-grained access control, Compliance with industry standards.

- SageMaker uses these 4 functions to standardize the inference process:
  - model_fn(model_dir): Ensures the model is loaded correctly. SageMaker automatically sets `model_dir` to the path where the model artifacts are extracted.
  - input_fn(input_data, content_type): Handles different input formats (e.g., JSON, CSV).
  - predict_fn(input_data, model): Performs inference using the model.
  - output_fn(predictions, accept): Formats the output for the client.

- Somehow I cannot install dependencies in requirements.txt while running the SakeMaker endpoint. I had to workaround by installing by Python code in `inference.py`. It slows down inference because dependencies install on every request.
  ```Python
  def install_dependencies():
    # Install all required packages
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "numpy=2.0.2", "transformers", ... ])

  install_dependencies()
  ```

### 7. Copy Files to AWS
Upload scripts and requirements to S3 so SageMaker can access them during training.
```
aws s3 cp train.py s3://s3-intent/scripts/train.py
aws s3 cp inference.py s3://s3-intent/scripts/inference.py
aws s3 cp model.py s3://s3-intent/scripts/model.py
aws s3 cp utils.py s3://s3-intent/scripts/utils.py
aws s3 cp requirements.txt s3://s3-intent/scripts/requirements.txt
```

S3 bucket s3://s3-intent
```
s3://s3-intent/
│── scripts/
│   ├── train.py            # Training script
│   ├── inference.py        # Inference script
│   ├── model.py            # Model class
│   ├── util.py             # Utilities
│   ├── requirements.txt    #
│── dataset/
│   ├── train.csv           # Training dataset
│   ├── test.csv            # Test dataset
│── model/                  # Store best performance model for inference
│   ├── model.pth           # 
│   ├── label_encoder.pkl   #
```

### 8. Training with SageMaker
- Go to the [AWS SageMaker Console](https://console.aws.amazon.com/sagemaker/)
- In the left-hand menu, click **Notebook**
- Create notebook instance `sagemaker-intent-notebook` with `ml.t3.medium` for general use using Role ARN created above.
- Once the instance is InService, click "Open Jupyter".
- `New`, select `conda_pytorch_p3.10`
- In `train.py`, add these lines of code at the beginning of the file to install packages in `requirements.txt`.
    ```python
    # Install dependencies from requirements.txt
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        subprocess.call(["pip", "install", "-r", requirements_path])
    ```
- `sagemaker.pytorch.PyTorch` is a framework-specific estimator designed for PyTorch. It automatically handles:
  + Pre-built PyTorch Docker images (no need to specify a custom training image).
  + Framework-specific configurations (e.g., PyTorch version, distributed training).
- `sagemaker.estimator.Estimator` is a generic estimator and requires you to manually specify a custom training Docker image and framework-specific configurations.
- AWS free tier cannot run SageMaker training jobs due to service quotas (e.g., insufficient instance limits), you can request a service quota increase.
- Request increase quotas for SageMaker https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas. Search `ml.t3.xlarge` to find item with "training job usage". Requested quota = 5, not sure what it is.
- Some ML instance available for "training job" and "endpoint" like `ml.c5.xlarge`.
- Sagemaker Instances with Pricing https://aws.amazon.com/sagemaker-ai/pricing/
  + SageMaker notebook: standard EC2 like `t3.medium`
  + SageMaker Training job instance: ML instances like `ml.t3.xlarge` $0.2/h, `ml.g4dn.xlarge` with GPU T4: $0.74/h
  + SageMaker Endpoint instance: the same with training
- Run the training job in the Jupyter Notebook.
- Copy all needed files to the local environment of SageMaker notebook, so the notebook can install packages in `requirements.txt` and run `train.py` from that local folder.

    ```bash
    !aws s3 cp s3://s3-intent/scripts/ scripts/ --recursive
    ```

- `requirements.txt`: do not include `spacy` which has lots of dependencies
    ```
    2025-03-09 16:28:52 Starting - Starting the training job...
    ..25-03-09 16:29:25 Downloading - Downloading input data.
    ..25-03-09 16:29:51 Downloading - Downloading the training image.
    .025-03-09 16:30:26 Training - Training image download completed. Training in progress..
    /opt/conda/bin/python3.8 -m pip install -r requirements.txt
    Requirement already satisfied: numpy in /opt/conda/lib/python3.8/site-packages (1.22.2)
    Requirement already satisfied: pandas in /opt/conda/lib/python3.8/site-packages (1.4.3)
    Collecting nltk
    Downloading nltk-3.9.1-py3-none-any.whl (1.5 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 72.4 MB/s eta 0:00:00
    Collecting transformers
    Downloading transformers-4.46.3-py3-none-any.whl (10.0 MB)
    ...
    Invoking script with the following command:
    /opt/conda/bin/python3.8 train.py
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
    Device cpu
    Number of unique intents: 27
    ```
- When you finish the training job, go back to `Notebook instances`, pick the instance and choose Actions - Stop.

SageMaker output path
- Keep the default as "/opt/ml/model" in train.py. After training, SageMaker will automatically copy the contents of "/opt/ml/model" to `s3://your-sentiment-analysis-bucket/models/`.
- In the training job config, set the S3 output path, e.g.:
```
"OutputDataConfig": {
    "S3OutputPath": "s3://your-sentiment-analysis-bucket/models/"
}
```

### 9. Inference with SageMaker Endpoint
Once training is done, deploy the model using a SageMaker Endpoint.
deploy `inference.py` as a SageMaker Endpoint to serve real-time predictions via an API.

#### Bash commands to create model.tar.gz file manually
- Copy files into folder a local model_dir: model.pth, label_encoder.pkl, inference.py, model.py, utils.py, requirements.txt
- Run command outside the model_dir to create model.tar.gz file
  ```bsh
  tar -czvf model.tar.gz -C model_dir .
  ```
- Upload model.tar.gz to s3://s3-intent/models
  ```bsh
  aws s3 cp model.tar.gz s3://s3-intent/models/model.tar.gz
  ```
- Run deployment in SageMaker notebook

- List all endpoints
    ```
    aws sagemaker list-endpoints
    ```
- `ml.g4dn.xlarge for endpoint usage` is available for free tier, so we can use it with the pre-trained model
- Check [CloudWatch Logs](https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups), /aws/sagemaker/Endpoints/intent-detection-endpoint
    ```
    aws logs describe-log-groups --log-group-name /aws/sagemaker/Endpoints/intent-detection-endpoint
    ```


### 10. Expose Model via an API Gateway


### 1x. Setting AWS Budget
- Why? Forgot to turn on an SageMaker endpoint using GPU for over a week
- Go to Billing and Cost Management > [Budget](https://us-east-1.console.aws.amazon.com/costmanagement/home?region=us-east-1#/budgets) > Create a budget
- Budget setup > Customize (advanced) > Cost budget - Recommended
- Set budget amount > Period = Daily > Enter your budgeted amount ($) = $5
- Setup alert at the threshold of 80%
- Current vs. budgeted 417.07%