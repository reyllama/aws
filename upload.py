import boto3

# Create an S3 client
s3 = boto3.client('s3')

# directory + filename 을 넣고 filename으로 저장
dir_name  = 'C:/Users/rin46/Documents/2019-1/Machine_Learning/assignments/'
filename = 'mnist_train.csv'
bucket_name = 'chaerin-dataset'

# Uploads the given file using a managed uploader, which will split up large
# files automatically and upload parts in parallel.
s3.upload_file(dir_name+filename, bucket_name, filename)

filename2 = 'mnist_test.csv'
s3.upload_file(dir_name+filename2, bucket_name, filename2)

print('Upload Complete! - ',filename, filename2)
