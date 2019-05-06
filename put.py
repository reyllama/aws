from io import StringIO
import pandas as pd
import boto3

bucket = 'chaerin-dataset'
file_name = "mnist_train.csv"
df = pd.read_csv(file_name)


csv_buffer = StringIO()
df.to_csv(csv_buffer)
s3 = boto3.resource('s3')
s3.Object(bucket, 'upload_sample.csv').put(Body=csv_buffer.getvalue())
print(file_name)
