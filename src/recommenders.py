import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from implicit.bpr import BayesianPersonalizedRanking


class MainRecommender:
    """Рекомендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    weiting: string
        Тип взвешивания, один из вариантов: None, 'bm25', 'tfidf'
    fake_id: int
        Идентификатор, которым заменялись режкие объекты, либо None
    """

    def __init__(self, data, weighting=None, fake_id=999999):

        assert weighting in ['tfidf', 'bm25', None], 'Неверно указан метод'

        # топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        if fake_id is not None:
            self.top_purchases = self.top_purchases.loc[self.top_purchases['item_id'] != fake_id]

        # топ покупок по всему датасету
        self.overall_top_purchases = data.groupby(['item_id'])['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        if fake_id is not None:
            self.overall_top_purchases = self.overall_top_purchases.loc[
                self.overall_top_purchases['item_id'] != fake_id]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        # дальнейшая предобработка
        self.fake_id = fake_id
        #         self.data = data
        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self._prepare_dicts(
            self.user_item_matrix)
        # эту матрицу можно взвесить
        self.user_item_matrix = csr_matrix(self.user_item_matrix).tocsr()
        # а эта останется для предсказаний
        self.user_item_matrix_for_pred = self.user_item_matrix

        if weighting == 'bm25':
            self.user_item_matrix = bm25_weight(self.user_item_matrix, K1=120, B=0.6).tocsr()
            # self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T.tocsr()
        elif weighting == 'tfidf':
            self.user_item_matrix = tfidf_weight(self.user_item_matrix).tocsr()
            # self.user_item_matrix = tfidf_weight(self.user_item_matrix.T).T.tocsr()

        # пользователи, которых можно включать в вывод для similar_users
        self.users_for_similar = [self.userid_to_id[x] for x in self.top_purchases['user_id'].unique()]

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        self.bpr_model = self.fit_bpr_recommender(self.user_item_matrix)

    @staticmethod
    def _prepare_matrix(data, values='quantity'):  # выбор столбца для значений

        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values=values,
                                          aggfunc='count',
                                          fill_value=0
                                          )

        return user_item_matrix.astype(float)

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(user_item_matrix)

        return own_recommender

    @staticmethod
#     def fit(user_item_matrix, n_factors=300, regularization=1, alpha=0.5, iterations=15):
    # эти параметры незначительно хуже, но быстрее!
    def fit(user_item_matrix, n_factors=200, regularization=0.05, alpha=0.8, iterations=15):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        alpha=alpha,
                                        random_state=42)
        model.fit(user_item_matrix)

        return model
    
    @staticmethod
    def fit_bpr_recommender(user_item_matrix, factors=300, learning_rate=0.002, regularization=0.04, 
                            iterations=100):
        """Обучает модель bpr"""

        bpr_model = BayesianPersonalizedRanking(factors=factors,
                                                learning_rate=learning_rate,
                                                regularization=regularization,
                                                iterations=iterations,
                                                random_state=42)
        bpr_model.fit(user_item_matrix)

        return bpr_model

    def _update_dict(self, user_id):
        """Обновляет словари, если появился новый user/item"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values())) + 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)
        # первый наиболее похожий - сам товар
        top_rec = recs[0][1]
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если число рекомендаций < N, дополняем из из top-popular"""

        if len(recommendations) < N:
            top_popular = [rec for rec in self.overall_top_purchases[:N] if rec not in recommendations]
            recommendations.extend(top_popular)
            recommendations = recommendations[:N]
            
        # own где-то цепляет 1 лишюю рек, те для него бы свою ф-ю
        if len(recommendations) > N:
            recommendations = recommendations[:N]
            
        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стандартные библиотеки implicit"""

        self._update_dict(user_id=user)
        filter_items = [] if self.fake_id is None else [self.itemid_to_id[self.fake_id]]
        res = model.recommend(userid=self.userid_to_id[user],
                              user_items=self.user_item_matrix_for_pred[self.userid_to_id[user]],
                              N=N,
                              filter_already_liked_items=False,
                              filter_items=filter_items,
                              recalculate_user=True
                              )
        # implicit версии 0.6.2 сортирует как для ALS, так и для ItemItemRecommender
        # mask = res[1].argsort()[::-1]
        # res = [self.id_to_itemid[rec] for rec in res[0][mask]]
        res = [self.id_to_itemid[rec] for rec in res[0]]
        res = self._extend_with_top_popular(res, N=N)
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стандартные библиотеки implicit"""

        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендации товары среди тех, которые юзер уже купил"""
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendations(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        self._update_dict(user_id=user)
        top5 = self.top_purchases.loc[self.top_purchases['user_id'] == user].head(5)['item_id'].to_list()
        # res = [self.id_to_itemid[self.model.similar_items(self.itemid_to_id[top_item], N=2)[0][1]]
        #            for top_item in top5]
        res = [self._get_similar_item(item) for item in top5]
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        # рекомендации отсортированы по quantity для исходного товара
        return res

        # топ похожих юзеров + по одному товару от каждого
        # если товар следующего пользователя уже есть - берём 2 товар этого пользователя
        # и это неоптимально
    def get_similar_users_recommendations(self, user, n_similar=3, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        self._update_dict(user_id=user)

        # проверить, что userid есть в item-item матрице
        if self.userid_to_id[user] < self.user_item_matrix.shape[0]:
            # users=self.users_for_similar оставляет в выводе только те userid, которые есть в
            # top_purchases
            similar_users = self.model.similar_users(self.userid_to_id[user],
                                                     N=n_similar + 1,
                                                     users=self.users_for_similar)[0][1:]
            similar_users = [self.id_to_userid[x] for x in similar_users]
            similar_users_items = (self.top_purchases.loc[self.top_purchases['user_id'].isin(similar_users)]
                .groupby('user_id')['item_id']
                .unique()[similar_users]
                )
            res = []
            # добавляем первую покупку юзера, если её ещё нет в рекомендациях, иначе - следующую...
            for u in similar_users_items.index:
                for items in similar_users_items[u]:
                    if items in res:
                        continue
                    res.append(items)
                    break

        else:
            res = []
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_bpr_recommendations(self, user, N=5):
        """Рекомендации bpr"""
        
        self._update_dict(user_id=user)
        filter_items = [] if self.fake_id is None else [self.itemid_to_id[self.fake_id]]
        res = self.bpr_model.recommend(userid=self.userid_to_id[user],
                                       user_items=self.user_item_matrix_for_pred[self.userid_to_id[user]],
                                       N=N,
                                       filter_already_liked_items=False,
                                       filter_items=filter_items,
                                      )

        res = [self.id_to_itemid[rec] for rec in res[0]]
        res = self._extend_with_top_popular(res, N=N)
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_top_purchases(self, user, N=5):
        """Возвращает топ собственных покупок пользователя"""
        
        res = self.top_purchases.loc[self.top_purchases.user_id==user].head(N).item_id.to_list()
        res = self._extend_with_top_popular(res, N=N)
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
