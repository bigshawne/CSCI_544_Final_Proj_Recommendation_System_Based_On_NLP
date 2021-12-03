import pyspark
from pyspark import SparkContext, SparkConf
import json
import sys

if __name__ == '__main__':
# You can configure the SparkContext
    _, business_path, user_path = sys.argv

    conf = SparkConf()
    sc = SparkContext(appName='mm_exp', conf=conf)
    
    BusinessRDD = sc.textFile(business_path).map(lambda x: json.loads(x))
    print(BusinessRDD.count())
    BusinessRDD = BusinessRDD \
                        .filter(lambda x: x['state'] == 'MA') \
                        .filter(lambda x: x['review_count'] >= 2) \
                        .map(lambda x: (x['business_id'], x['stars'], x['review_count']))
    print(BusinessRDD.count())
    BusinessStats = BusinessRDD.collect()
    BusinessOutput = {item[0]:{'stars':item[1], 'review_count':item[2]} for item in BusinessStats}
    with open('./revised_data/business_revised.json', 'w') as f:
        json.dump(BusinessOutput, f)
    
    UserRDD = sc.textFile(user_path).map(lambda x: json.loads(x))
    print(UserRDD.count())
    UserRDD = UserRDD \
                    .filter(lambda x: x['review_count'] >= 2) \
                    .map(lambda x: (x['user_id'], x['average_stars'], x['review_count']))
    UserStats = UserRDD.collect()
    UserOutput = {item[0]:{'average_stars':item[1], 'review_count':item[2]} for item in UserStats}

    for i in UserStats:
        print(i)
        break       
    with open('./revised_data/user_revised.json', 'w') as f:
        json.dump(UserOutput, f)

    sc.stop()