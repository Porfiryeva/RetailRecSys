import pandas as pd
import numpy as np


def prefilter_items(data, item_features=None, take_n_popular=5000,):

#         Уберем самые популярные товары (их и так купят)
#     popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
#     popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
#     popularity['share_unique_users'] = popularity['share_unique_users'] / data['user_id'].nunique()
# #     > 0.2
#     top_popular = popularity[popularity['share_unique_users'] > 0.8].item_id.tolist()
#     data = data.loc[~data['item_id'].isin(top_popular)]

# #     Уберем самые НЕ популярные товары (их и так НЕ купят)  < 0.02
#     top_notpopular = popularity[popularity['share_unique_users'] < 0.001].item_id.tolist()
#     data = data.loc[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    # придётся убрать около млн строк - те чуть меньше половины

    # Уберем неинтересные для рекомендаций категории (department)
    # немного улучшает recall
    if item_features is not None:
        department_size = (item_features.groupby('department')['item_id']
                                        .nunique()
                                        .sort_values(ascending=False)
                                        .reset_index()
                                        .rename(columns={'item_id': 'n_items'})
                          )
        rare_departments = department_size[department_size['n_items'] < 10].department.tolist()
        items_in_rare_departments =  item_features.loc[item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data.loc[~data['item_id'].isin(items_in_rare_departments)]
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.

    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
#     data = data[data['price'] > 0.001]
# этот вариант чуть лучше
    data = data[data['sales_value'] > 0.1]


    # Уберем слишком дорогие товары
    data = data[data['price'] < 200]
    
    # ...

#     пользователи, сделавшие менее 5 покупок
    active = data.groupby('user_id')['item_id'].count().reset_index().rename(columns={'item_id': 'n_purchases'})
    inactive_id = active.loc[active['n_purchases'] < 5].user_id.tolist()
    data = data.loc[~data['item_id'].isin(inactive_id)]

    # оставляем только top-5000
    popularity = data.groupby('item_id')['quantity'].sum().reset_index().rename(columns={'quantity': 'n_sold'})
    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data
    

def postfilter_items(user_id, recommednations):
    pass
   
    
def get_weighted_comb_rec(row, cols, weights, k=100):
    """
    Принимает веса и возвращает соответствующие доли от указанных колонок
    """
    weights = (1 + k * np.array(weights)).astype('int')
    
    unique_recommendations = row[cols[0]][:weights[0]]
    
    for col, w in zip(cols[1:], weights[:1]):
        for i in range(w):  # число добавленных рек.
            for item in row[col]:
                if item in unique_recommendations:
                    continue
                unique_recommendations.append(item)
                break
    
    return unique_recommendations[:k]
    
