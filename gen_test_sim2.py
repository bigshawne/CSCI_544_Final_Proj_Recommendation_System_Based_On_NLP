import json
import sys
import csv
from sklearn.metrics.pairwise import cosine_similarity

def embedding_similarity(train_embed, test_embed):
    return cosine_similarity([train_embed], [test_embed])[0][0]

def gen_sim_user_bus(item_CF_train, item_CF_test, train_bus_r_id_dict, test_bus_r_id_dict, user_rated_bus, train_data, test_data, N):
    Sim_test = {}

    for i in range(len(item_CF_test)):
        u = item_CF_test[i][0]
        b = item_CF_test[i][1]
        test_r_id = test_bus_r_id_dict[(u,b)]
        test_r_embed = test_data[test_r_id]

        if u not in user_rated_bus.keys():
            Sim_test[u] = {}
        else:
            u_sim = {}

            if len(user_rated_bus[u]) < N:
                Sim_test[u] = user_rated_bus[u]
                continue

            for train_b in user_rated_bus[u].keys():
                train_r_ids = train_bus_r_id_dict[train_b]
                
                sumall = 0
                suml = 0

                for train_r_id in train_r_ids[:5]:
                    train_r_embed =  train_data[train_r_id]

                    sim = embedding_similarity(train_r_embed, test_r_embed)

                    sumall += sim
                    suml += 1

                if suml == 0:
                    continue
                else:
                    u_sim[train_b] = sumall / suml
            sorted_u_sim = dict(sorted(u_sim.items(), key = lambda kv: (kv[1], kv[0]), reverse=True)[:N])

            Sim_test[u] = {}
            for k in sorted_u_sim.keys():
                Sim_test[u][k] = user_rated_bus[u][k]

    return Sim_test

def gen_user_rated_bus(train_data):
    user_business_dict = {p[0]: {} for p in train_data}

    for u,b,star in train_data:
        user_business_dict[u][b] = star

    return user_business_dict

def load_user_bus_star_pair(r_id_lst, review_info, mode):
    data = [tuple(review_info[r_id].values()) for r_id in r_id_lst]
    
    if mode == 'train':
        bus_r_id_dict = {}
        for r_id in r_id_lst:
            b = review_info[r_id]['business_id']

            if b in bus_r_id_dict.keys():
                br_id = bus_r_id_dict[b]
                br_id.append(r_id)
                bus_r_id_dict[b] = br_id
            else:
                bus_r_id_dict[b] = [r_id]
    else:
        bus_r_id_dict = {(review_info[r_id]['user_id'], review_info[r_id]['business_id']):r_id for r_id in r_id_lst}
    return data, bus_r_id_dict

def load_data(model, folder_path, mode, N):
    if mode == 1:
        embedding_train_file = folder_path + model.lower() + '_lemma_train.json'
        embedding_test_file = folder_path + model.lower() + '_lemma_test.json'
    else:
        embedding_train_file = folder_path + model.lower() + '_no_lemma_train.json'
        embedding_test_file = folder_path + model.lower() + '_no_lemma_test.json'

    print('Load embedding train')
    with open(embedding_train_file) as f:
        train_data = json.load(f)
    print('Load embedding test')
    with open(embedding_test_file) as f:
        test_data = json.load(f)
    
    print('Get train r_id')
    train_r_id = list(train_data.keys())
    print('Get test r_id')
    test_r_id = list(test_data.keys())

    print('Load review info')
    review_info_file = './revised_data/review_info.json'
    with open(review_info_file) as f:
        review_info = json.load(f)

    print('Generate train pair')
    item_CF_train, train_bus_r_id_dict = load_user_bus_star_pair(train_r_id, review_info, 'train')
    print('Generate test pair')
    item_CF_test, test_bus_r_id_dict = load_user_bus_star_pair(test_r_id, review_info, 'test')
    del train_r_id, test_r_id

    print('Generate user rated bus')
    user_rated_bus = gen_user_rated_bus(item_CF_train)
    print('Generate sim dict')
    user_sim_bus = gen_sim_user_bus(item_CF_train, item_CF_test, train_bus_r_id_dict, test_bus_r_id_dict, user_rated_bus, train_data, test_data, N)

    return item_CF_train, item_CF_test, user_sim_bus
if __name__ == '__main__':
    mode = int(sys.argv[1]) # 1 for with lemma 2 for without lemma
    model = sys.argv[2] # Bert, GloVe, Google, Own
    N = int(sys.argv[3]) # Max number of neighbors

    if model == 'Own':
        folder_path = './Own model/'
    else:
        folder_path = './' + model + '/'

    train_data, test_data, user_sim_bus = load_data(model, folder_path, mode, N)

    print('Save user sim')
    if mode == 1:
        path = './embed_sim/' + model + '_w_lemma_user_sim.json'
    else:
        path = './embed_sim/' + model + '_no_lemma_user_sim.json'
    with open(path, 'w') as f:
        json.dump(user_sim_bus, f)

    print('Save item train')
    if mode == 1:
        path = './embed_sim/' + model + '_w_lemma_item_train.csv'
    else:
        path = './embed_sim/' + model + '_no_lemma_item_train.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)

        for row in train_data:
            writer.writerow(row)

    print('Save item test')
    if mode == 1:
        path = './embed_sim/' + model + '_w_lemma_item_test.csv'
    else:
        path = './embed_sim/' + model + '_no_lemma_item_test.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)

        for row in test_data:
            writer.writerow(row)