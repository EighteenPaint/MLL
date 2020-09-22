# 从文件读取的数据要记得转成series
import pandas as pd

data = pd.read_json("small_data")
print(data.head(10))

org_layer_df = pd.read_csv("/Users/benny.chen/PycharmProjects/MLL/juypter/org_to_ids_new_list.csv")
org_layer_df.rename(columns={'Unnamed: 0': 'org_name', '0': 'article_id'}, inplace=True)
# org_layer_df['article_id'].head(10)
org_layer_series = pd.Series(data=org_layer_df['article_id'].values, index=org_layer_df['org_name'])
org_layer_series.head(10)

org_to_name = dict()
for org_name in org_layer_series.index:  # 遍历每一个机构簇
    same_name_cluster = dict()  # 对于每一个机构需要生成机构下的同名簇
    for article_id in org_layer_series[org_name].split(","):  # 每一个机构簇里所有论文
        authors = data[article_id]['author']  # 取出文章的作者数据
        for name_dict in authors:  # 遍历该篇文章的作者数据
            author_name = name_dict['name']  # 取出name字段
            if author_name not in same_name_cluster.keys():  # 检查是不是第一次
                same_name_cluster[author_name] = set().add(article_id)
            else:  # 如果已经存在，则直接增加即可
                same_name_cluster[author_name].add(article_id)
    org_to_name[org_name] = same_name_cluster  # 挂载在机构下，形成机构下同名簇
