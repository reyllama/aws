import boto3
import json
import pandas as pd
from io import StringIO

bucketName = 'instagram-post'
s3 = boto3.resource('s3')
my_bucket = s3.Bucket(bucketName)

#csv = pd.read_csv('c:/users/home/desktop/instagram-crawler_/instagram.csv', index_col=0, header=None, names=['username'])
#csv = pd.read_csv('instagram.csv', index_col=0, header=None, names=['username'])
csv = pd.read_csv('new_instagram.csv', index_col=0, header=None, names=['username'])
instaList = csv.index.tolist()

result = pd.DataFrame()


for json_file in my_bucket.objects.all():
    if json_file.key[:-5] in instaList:
        i+=1
        content_object = s3.Object(bucketName, json_file.key)
        file_content = content_object.get()['Body'].read().decode('utf-8')
        data = json.loads(file_content)

        username = csv.loc[json_file.key[:-5]][0]

        commenters = []
        for post in data:
            if post.get('comments'):
                for comment in post.get('comments'):
                    commenters.append(comment.get('author'))

        influencer = {'name': username,
                'youtube': json_file.key[:-5],
                'commenters': set(commenters)}
                # 프로필 크롤링 결과에서 칼럼 추가...

        result = pd.concat([result, pd.DataFrame(influencer)])
        print(i,'out of',csv.shape[0])

csv_buffer = StringIO()
result.reset_index(drop=True).to_csv(csv_buffer)
s3.Object(bucketName, 'commenter_result.csv').put(Body=csv_buffer.getvalue())
