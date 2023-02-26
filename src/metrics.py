import numpy as np


def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    hit_rate = int(flags.sum() > 0)
    
    return hit_rate


def hit_rate_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list[:k], bought_list,)   
    hit_rate = int(flags.sum() > 0)
    
    return hit_rate


def precision(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    k_recommended_list = np.array(recommended_list[:k])

    flags = np.isin(k_recommended_list, bought_list)

    precision = flags.sum() / len(k_recommended_list)  # или обработать деление на 0?
    return precision


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
        
    bought_list = np.array(bought_list)
    k_recommended_list = np.array(recommended_list[:k])
    k_prices_recommended = np.array(prices_recommended[:k])
    
    flags = np.isin(k_recommended_list, bought_list)
    
    precision = (flags * k_prices_recommended).sum() / k_prices_recommended.sum()
    
    return precision


def recall(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    recall = flags.sum() / len(bought_list)
    
    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    k_recommended_list = np.array(recommended_list[:k])

    flags = np.isin(k_recommended_list, bought_list)

    recall = flags.sum() / len(bought_list)
    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    k_recommended_list = np.array(recommended_list[:k])
    k_prices_recommended = np.array(prices_recommended[:k])
    prices_bought = np.array(prices_bought)
    
    flags = np.isin(k_recommended_list, bought_list)
    precision = (flags * k_prices_recommended).sum() / prices_bought.sum()
    
    return precision


def ap_at_k(recommended_list, bought_list, k=5):
    """
    Принимает ранжированный по релевантности recommended_list
    """
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list,bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(k): 
        
        if flags[i]:
            p_k = precision_at_k(recommended_list, bought_list, k=i+1)
            sum_ += p_k 
            
    result = sum_ / k 
    
    return result


def map_k(recommended_list, bought_list, k=5):
    """
    Принимает ранжированный по релевантности recommended_list
    recommended_list и bought_list - pd.Series
    """
    recommended_lists = np.array(recommended_list)
    bought_lists = np.array(bought_list)
    
    result = np.array([ap_at_k(x, y, k) for x, y in zip(recommended_lists, bought_lists)]).mean()
    
    return result


# здесь - разночтения в том, как считать IDCG: число единичек в маске = range(k) или 
# range(bought_list) если число покупок < k 
def ndcg_at_k(recommended_list, bought_list, k=5):
    """
    Принимает ранжированный по релевантности recommended_list
    """
    bought_list = np.array(bought_list)
    k_recommended_list = np.array(recommended_list[:k])
    
    flags = np.isin(k_recommended_list, bought_list)
    ideal_dcg = np.array([1/np.log2(i + 2) for i in range(k)])  # +1 начинаем с 0, +1 - по формуле discount
    
    ndcg = (flags * ideal_dcg).sum() / ideal_dcg.sum()
    
    return ndcg


def reciprocal_rank_at_k(recommended_list, bought_list, k=5):
    """
    Принимает ранжированный по релевантности recommended_list
    """
    bought_list = np.array(bought_list)
    k_recommended_list = np.array(recommended_list[:k])

    flags = np.isin(k_recommended_list, bought_list)
    for i, flag in enumerate(flags, start=1):
        if flag:
            return (1/i)
    
    return 0
    
