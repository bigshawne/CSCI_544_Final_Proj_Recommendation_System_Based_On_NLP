from pyspark import SparkContext, SparkConf
import time
import json
import csv
import random
import sys
from math import sqrt

def pearson(iterator):
    curr_user = iterator[0]
    curr_bus = iterator[1]

    if curr_bus not in business_lst:
        if curr_user in user_lst:
            curr_user_bus_dict = user_sim_bus[curr_user]
            user_row_vals = curr_user_bus_dict.values()
            row_avg = sum(user_row_vals) / len(user_row_vals)
            return (curr_user, curr_bus, row_avg)
        else:
            return (curr_user, curr_bus, overall_avg)

    bus_user_dict = business_val_dict[curr_bus]
    curr_vals = bus_user_dict.values()
    col_avg = sum(curr_vals) / len(curr_vals)

    #guess_avg = round((col_avg + row_avg) / 2)

    if curr_user not in user_lst:
        return (curr_user, curr_bus, col_avg)

    curr_user_bus_dict = user_sim_bus[curr_user]
    user_row_vals = curr_user_bus_dict.values() 
    row_avg = sum(user_row_vals) / len(user_row_vals)
    guess_avg = (col_avg + row_avg) / 2

    #if curr_user in bus_user_dict.keys():
    #    return (curr_user, curr_bus, bus_user_dict[curr_user])

    numerator = 0
    denominator = 0 
    for b_val in curr_user_bus_dict.keys():
        if curr_bus == b_val:
            continue
        b_val_user_dict = business_val_dict[b_val]
        curr_r = b_val_user_dict[curr_user]

        if curr_bus < b_val:
            pair = (curr_bus, b_val)
        else:
            pair = (b_val, curr_bus)

        b_common = []
        b_val_common = []
        for key in b_val_user_dict.keys():
            if key in bus_user_dict.keys():
                b_common.append(bus_user_dict[key])
                b_val_common.append(b_val_user_dict[key])

        if len(b_common) < 80:
            continue
        #if len(b_common) * len(b_val_common) == 0:
        #    continue

        b_avg = sum(b_common) / (len(b_common) if len(b_common) > 0 else len(b_common) + 1)
        b_val_avg = sum(b_val_common) / (len(b_val_common) if len(b_val_common) > 0 else len(b_val_common) + 1) 

        weight_numerator = 0
        for i in range(len(b_common)):
            weight_numerator += ((b_common[i] - b_avg) * (b_val_common[i] - b_val_avg))
            
        b_denominator = sum([(r-b_avg)**2 for r in b_common])
        b_val_denominator = sum([(r-b_val_avg)**2 for r in b_val_common])

        try:
            weight = weight_numerator / (sqrt(b_denominator) * sqrt(b_val_denominator))
        except ZeroDivisionError:
            weight = 0

        #if weight < 0:
        #   continue
        numerator += curr_r * weight
        denominator += abs(weight)

    
    if denominator == 0:
        return (curr_user, curr_bus, guess_avg)
        
    final_val = numerator / denominator
    
    if final_val < 1:
        return (curr_user, curr_bus, 1.0)
    elif final_val >= 5:
        return (curr_user, curr_bus, 5.0)

    return (curr_user, curr_bus, final_val)

def load_data(mode, model):
    if mode == 2:
        train_path = './embed_sim/' + model + '_no_lemma_item_train.csv'
        test_path = './embed_sim/' + model + '_no_lemma_item_test.csv'
        sim_path = './embed_sim/' + model + '_no_lemma_user_sim.json'
    else:
        train_path = './embed_sim/' + model + '_w_lemma_item_train.csv'
        test_path = './embed_sim/' + model + '_w_lemma_item_test.csv'
        sim_path = './embed_sim/' + model + '_w_lemma_user_sim.json'

    train_data = []
    with open(train_path) as f:
        for row in f:
            rs = row.strip('\n').split(',')
            train_data.append((rs[0], rs[1], float(rs[2])))

    test_data = []
    with open(test_path) as f:
        for row in f:
            rs = row.strip('\n').split(',')
            test_data.append((rs[0], rs[1], float(rs[2])))

    with open(sim_path) as f:
        user_sim_bus = json.load(f)

    return train_data, test_data, user_sim_bus

if __name__ == '__main__':
    conf = SparkConf().setAppName('item_CF_W_review')
    sc = SparkContext(conf=conf)

    mode = int(sys.argv[1])
    model = sys.argv[2]

    train_data, test_data, user_sim_bus = load_data(mode, model)

    trainRDD = sc.parallelize(train_data)

    businessRDD = trainRDD.map(lambda row: (row[1], [(row[0], row[2])])) \
                        .reduceByKey(lambda a,b: a+b) \
                        .sortByKey() \
                        .map(lambda row: (row[0], dict(row[1])))

    business_lst = businessRDD.map(lambda row: row[0]).collect()

    business_val_dict = dict(businessRDD.collect())

    userRDD = trainRDD.map(lambda row: (row[0], [(row[1], row[2])])) \
                    .reduceByKey(lambda a,b: a+b) \
                    .sortByKey()

    user_lst = userRDD.map(lambda row: row[0]).collect()

    overall_avg = trainRDD.map(lambda row: (1, [row[2]])) \
                        .reduceByKey(lambda a,b: a+b) \
                        .map(lambda x: sum(x[1]) / len(x[1])) \
                        .collect()[0]

    testRDD = sc.parallelize(test_data)

    result = testRDD.map(pearson).collect()

    sum_all = 0
    count = 0
    for i in range(len(test_data)):
        count += 1

        sum_all += (test_data[i][-1] - result[i][-1])**2

    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)

        for i in range(len(test_data)):
            writer.writerow((test_data[i][0], test_data[i][1], test_data[i][2], result[i][-1]))

    RMSE = sqrt(sum_all / count)

    print('RMSE:', RMSE)