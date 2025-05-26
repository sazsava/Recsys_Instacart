#Для визуального отображения прогресса выполнения кода
from tqdm import tqdm
#Для взаимодействия с ОС (здесь нужна для работы с директориями файлов)
import os
#Для работы с датафреймами
import pandas as pd
#Для вычислений
import numpy as np
#Для нормализации данных (приведения к единому диапазону)
from sklearn.preprocessing import MinMaxScaler
#Для работы с моделью
import tensorflow as tf
#Для модели бустинга
import xgboost as xgb
#Для модели ALS библиотеки surprise
from surprise import Dataset, Reader, BaselineOnly
#Для оптимизации использования памяти
from functools import reduce
#Для загрузки моделей:
import pickle
#Для обработки user input:
import sys
import argparse
# Garbage Collector для периодической очистки памяти
import gc                         
gc.enable()

#Итоговый класс retrain model
#Сохраним здесь пути для предобученных моделей:
abs_path = os.path.dirname('__file__')
xgb_ranker_path = os.path.join(abs_path, "xgb_ranker.pkl")
als_model_path = os.path.join(abs_path, "als_model.pkl")
#Данные по транзакциям и продуктам:
transactions_csv_path = os.path.join(abs_path, './transactions.csv')
products_csv_path = os.path.join(abs_path, './products.csv')

class RecsysModel():

    
    def __init__(self, xgb_ranker = None, als_model = None, transactions = None, 
                 X_train = None,y_train = None, n = 10, r_xgb = 0.3, r_als =  0.7, user_treshold = 50):
        #Определим модели:
        self._xgb_ranker = xgb_ranker
        self._als_model = als_model
        
        #Также определим данные для подачи в модель (уже обработанные X и y):
        self._X_train = X_train
        self._y_train = y_train
        

        #Сет с транзакциями для нахождения наиболее популярных товаров (при необходимости):
        self._transactions = transactions
        
        #И значения по умолчанию для обучения моделей:
        self._n = n #к-во продуктов в предсказании, в условиях задания равно 10
        self._r_xgb =  r_xgb #xgb ratio - доля предсказания модели бустинга в итоговом предсказании, по умолчанию = 0.3
        self._r_als = r_als  #als ratio - доля предсказания модели ALS в итоговом предсказании, по умолчанию = 0.7
        self._user_treshold = user_treshold #минимальное к-во пользователей, заказавших товар, для предскзания по популярности
    
    
    #Методы для подгрузки данных:
    def load_xgb_model(self, new_model = None, new_model_path = None):
        #Если метод уже запускался, то ничего не вернем:
        if self._xgb_ranker is not None: 
            return self
        #Непосредственная загрузка из ноутбука:
        if new_model is not None:
            self._xgb_ranker = new_model
            return self
        #ЛИБО загрузка из файла в формате pkl:
        elif new_model_path is not None:
            with open(new_model_path, 'rb') as f:
                self._xgb_ranker = pickle.load(f)
            return self
        #По умолчанию загружается обученная предварительно модель:  
        if self._xgb_ranker is None: 
            with open(xgb_ranker_path, 'rb') as f:
                self._xgb_ranker = pickle.load(f)
     
    #Аналогично загрузим als модель:
    def load_als_model(self, new_model = None, new_model_path = None):
        if self._als_model is not None: 
            return self
        if new_model is not None:
            self._als_model = new_model
            return self
        elif new_model_path is not None:
            with open(new_model_path, 'rb') as f:
                    self._als_model = pickle.load(f)
            return self      
        if self._als_model is None:
            with open(als_model_path, 'rb') as f:
                self._als_model = pickle.load(f)
    
    #Для X и y оставим возможность загрузки непосредственно из ноутбука 
    #(так мне кажется логичнее из архитектуры этого и следующих классов):
    def load_X_train(self,  new_X = None):
        if self._X_train is not None: 
            return self
        if new_X is not None:
            self._X_train = new_X
            return self
    
    
    def load_y_train(self, new_y = None):
        if self._y_train is not None: 
            return self
        if new_y is not None:
            self._y_train = new_y
            return self

                
    # А для транзакций можно оставить 2 варианта с приоритетом подачи напрямую:            
    def load_transactions(self, new_set = None, new_path = None):
        if self._transactions is not None: 
            return self
        if new_set is not None:
            self._transactions = new_set
            return self
        elif new_path is not None:
            self._transactions = pd.read_csv(new_path)
            return self
        if self._transactions is None:    
            self._transactions = pd.read_csv(transactions_csv_path)
                
    #Теперь по порядку определимся с задачами
    #Предсказание по одной позиции:
    def predict_one(self, user_id):
        #Загрузка нужного:
        self.load_xgb_model()
        self.load_als_model()
        self.load_X_train()
        self.load_y_train()
        #Отделим части сета для подачи в модели:
        _tr_trim = self._X_train[self._X_train.user_id == user_id]
        _predset = self._y_train[self._y_train.user_id == user_id]
      
        #Получим итоговый массив для подачи в ALS:
        _reader = Reader(rating_scale=(1, 10)) # Зададим разброс оценок
        _als_test = Dataset.load_from_df(_predset, _reader)
        _als_test = _als_test.df.to_numpy().tolist()
        #Получаем предсказания по пользователю
        #Для XGBoost:
        _xgb_pred = self._xgb_ranker.predict(_tr_trim.set_index(['user_id', 'product_id']))
        #Для ALS:
        _predictions = self._als_model.test(_als_test)
        _als_pred = []
        for i in _predictions:
            _als_pred.append(i[3])
        _als_pred = np.array(_als_pred)
        #И сразу сохраним их в столбцы сета:
        _predset['predicted_rating'] = _xgb_pred * self._r_xgb + _als_pred * self._r_als
        _predset = _predset.merge(_predset
        #Сгруппируем по пользователям, найдем 10 макс значений предсказанного рейтинга
            .groupby('user_id')['predicted_rating'].nlargest(self._n)
        #Получим новый df с  MultiIndex, котрый сбросим через reset_index
            .reset_index('user_id'),
        # в новом df нет колонки "product_id" поэтому необходимо объединить изначальный сет с полученным
            how='right') # при этом все строки, которых нет в новом df удалятся
        
        #Получим наш список
        _model_preds = _predset.groupby('user_id')['product_id'].unique().reset_index().product_id.values[0]
        return _model_preds
    
    #На случай, если попадется новый пользователь, решим проблему холодного старта простейшей рекомендацией по популярности:
    def get_popularity_rec(self):
        self.load_transactions()
        #Сделаем сортировку по количеству перезаказов, при этом учитём также и тот факт, 
        #что товар должен быть покупаем немаленьким количеством пользователей (мы возьмем 50 пользователей минимум).
        _df = self._transactions[self._transactions.groupby('product_id')['user_id'].transform('nunique') > self._user_treshold]
        _popularity = _df.groupby('product_id')['reordered'].sum().reset_index()
        _popularity.sort_values('reordered', ascending=False, inplace=True)
        _res = np.array(_popularity.head(self._n).product_id, dtype=np.int32)
        return _res
 
    #Непосредственный метод для предсказаний:
    def predict(self, ids):
        self.load_X_train()
        usrs = self._X_train.user_id.unique().tolist()
        #Проверка на тип входных значений для их обработки
        #Если на вход получен массив значений, то
        if hasattr(ids, "__len__"):
            #Зададим пустой массив, который мы будем подавать на выход
            _preds_array = np.zeros((len(ids), self._n), dtype=np.int32)
            for row, user_id in enumerate(ids):
                print('Calculating recommendation for user ', user_id)
                if user_id in usrs:
                    _preds_array[row,:] = self.predict_one(user_id)
                else:
                    _preds_array[row,:] = np.array(self.get_popularity_rec())
            return _preds_array
        #Если был подан один пользователь
        else:
            print('Calculating recommendation for user ', ids)
            if ids in usrs:
                return self.predict_one(ids)
            else:
                return np.array(self.get_popularity_rec())
    
    
    #Обучение моделей на новых дынных:
    #Подадим в качестве значений по умолчанию параметры, которые дали в нашем случае наилучший результат:
    def refit_model(self, eval_metric = ['ndcg@10'],
                   lambdarank_num_pair_per_sample = 15, n_estimators =75, min_child_weight = 12, max_depth = 11,
                   learning_rate = 0.1, bsl_method = 'als', bsl_epochs = 20, reg_u = 18, reg_i = 6):
        self.load_X_train()
        self.load_y_train()
        X_xgb = self._X_train.set_index(['user_id', 'product_id'])
        y_xgb = self._y_train.set_index(['user_id', 'product_id'])
        #Определим XGBRanker:
        self._xgb_ranker = xgb.XGBRanker(objective ='rank:ndcg',
            lambdarank_pair_method ='topk',
            lambdarank_num_pair_per_sample = lambdarank_num_pair_per_sample,
            n_estimators = n_estimators,
            min_child_weight = min_child_weight,
            max_depth = max_depth,
            learning_rate = learning_rate,
            eval_metric= eval_metric,
            tree_method ='hist',
            device = 'cuda',
            random_state=42)
        
        #Обучение
        self._xgb_ranker.fit(
                X_xgb,
                y_xgb,
#                 eval_set= None,
                verbose = 0
            )
        
        print('XGBRanker has been successfully fit!')
        
        #Настроим данные для ALS:
        _train_als = self._y_train
        _train_als['rating'] =_train_als['rating'].astype(np.int32)
        
        _reader = Reader(rating_scale=(1, 10)) # Зададим разброс оценок

        _trainset = Dataset.load_from_df(_train_als, _reader)
        _trainset = _trainset.build_full_trainset()

        #Установим модель:
        _bsl_options =  {'method': bsl_method, 'n_epochs': bsl_epochs, 'reg_u': reg_u, 'reg_i': reg_i}
        self._als_model = BaselineOnly(bsl_options = _bsl_options)
        #Обучение
        self._als_model.fit(_trainset)
        print('ALS model has been successfully fit!')

#Для работы с данными создадим отдельный класс:
class FormatDataset():
    def __init__(self,* , transactions = None, products = None, mainset = None, fill_empty = None, 
                 cols_to_leave = ['order_id', 'user_id', 'order_number','product_id', 'add_to_cart_order',
                                  'reordered',"order_dow","order_hour_of_day",'days_since_prior_order'], #что нужно для вычислений
                 feature_list = ['rating','pu_orders','p_reordered_times','pu_median_dspo',
                                 'pu_order_ratio','pu_median_cart_pos']):#что нужно для подачи в модель
        
        self._transactions = transactions
        self._products = products
        self._mainset = mainset
        self._fill_empty = fill_empty
        self._cols_to_leave = cols_to_leave
        self._feature_list = feature_list
        
    #Напишем пару классов для загрузки данных  
    # Проверка на ввод:  
    def process_input_data(self, new_data):
        #Dataframe возвращаем как есть:
        if isinstance(new_data, pd.DataFrame):
            return new_data
        #А путь передаем в pd.read_csv:
        elif isinstance(new_data, str):
            new_data = pd.read_csv(new_data)
            return new_data
        #Если что-то не так:
        else:
            raise TypeError('Wrong input type passed. Either pd.Dataframe or path to such is expected.')
            
    #Информация о продуктах:
    def load_products(self, new_set = None):
        if self._products is not None: 
            return self
        if new_set is not None:
            self._products = self.process_input_data(new_set)
        if self._products is None:
            self._products = pd.read_csv(products_csv_path)
            
    #Информация о транзакциях:
    def load_transactions(self, new_set = None):
        if self._transactions is not None: 
            return self
        if new_set is not None:
            self._products = self.process_input_data(new_set)
        if self._transactions is None:    
            self._transactions = pd.read_csv(transactions_csv_path)
            
    #И их объединенный сет:
    def load_mainset(self, new_set = None):
        if self._mainset is not None: 
            return self
        if new_set is not None:
            self._products = self.process_input_data(new_set)
    #Если mainset не определён, то запуститься метод merge_to_mainset:
        if self._mainset is None:
            self.load_products()
            self.load_transactions()
            self._mainset = self._merge_to_mainset()
            self._mainset = self._reduce_mem_usage(self._mainset)
    


    #Сюда вставим метод для уменьшения размерности датасета  c целью снижения нагрузки на память    
    def _reduce_mem_usage(self, df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2    
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose: 
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df
 
    #Общий метод обработки изменений:
    def _renew_data(self, init_df, new_data):
        new_cols = None
    
        """
        new_data - датафрейм c новыми данными о транзакциях/продуктах
        """
        #Проверка на наличие исходных столбцов:
        diff1 = set(init_df.columns) - set(new_data.columns)
        if len(diff1)>0:
            raise Exception('New data missing ', list(diff1), 'column data')
        
        #Проверка на добавление новых столбцов:
        diff2 = set(new_data.columns) - set(init_df.columns)
    
        if len(diff2)>0:
            new_cols = list(diff2)
            print("New features: ", new_cols, ' column data to be added')  
    
        #Объединение данных:
        new_df = pd.concat([init_df,new_data], ignore_index=True)
    
        #Очистим память:
        del [init_df,new_data]
        gc.collect()
    
        #Заполним пустые колонки(если задано значение fill_empty):
        if (self._fill_empty is not None) & (new_cols is not None):
            new_df[new_cols] = new_df.loc[:,new_cols].fillna(value=self._fill_empty)
    
        if 'user_id' in new_df.columns:
            new_df = new_df.sort_values(by=['user_id', 'order_number'])
        else:
            new_df = new_df.sort_values(by=['product_id'])
        
        new_df = self._reduce_mem_usage(new_df)
        return new_df
    
    #Для конкретной ситуации
            
    #Изменение сета с транзакциями:
    def renew_transactions(self, new_data):
        self.load_transactions()
        new_data = self.process_input_data(new_data)
        self._transactions=self._renew_data(self._transactions, new_data)
        return self
    #Изменение данных о продуктах:    
    def renew_products(self, new_data):
        self.load_products()
        new_data = self.process_input_data(new_data)
        self._products=self._renew_data(self._products, new_data)
        return self
    
    #Объединение данных в mainset
    def _merge_to_mainset(self):
         # Собственно merge:
        df = reduce(lambda left, right: pd.merge(left, right, on='product_id', how='left'), [self._transactions,self._products])
        df = df[self._cols_to_leave] #удалим неиспользуемые колонки, если они есть в сете
        #Удаление позволяет снизить нагрузку на память :)
        
        #Заполним пустые значения в days_since_prior_order:
        df['days_since_prior_order']=df['days_since_prior_order'].fillna(
            df.groupby('user_id')['days_since_prior_order'].transform('mean'))
        #Здесь и далее вычислим все коэффициенты, которые мы использовали в моделях ранее по порядку
        #Абсолютно аналогично тому, что я делала для обучения модели:
        #Коэффициент номера заказа с учетом повышения актуальности каждого следущего по времени заказа:
        df['order_num_coef'] = df['order_number'].apply(lambda x: round((np.log10(x) + 1),2))
        #Перераспределим веса порядка заказа(перевернем порядок заказа, позиции с 11 и выше примут отрицательные значения).
        df['add_to_cart_coef'] = 11-df['add_to_cart_order']
        #Теперь скоректируем его с учетом номера заказа:
        df['add_to_cart_coef'] = df['add_to_cart_coef'] * df['order_num_coef']
        #Посчитаем коэффициент перезаказов c учетом add_to_cart_coef:
        df['reordered'] = df['reordered'] * df['add_to_cart_coef']
        # Посчитаем рейтинг, как add_to_cart_coef с учетом регулярности покупки продукта покупателем (reordered)
        # Доли показателей определим как 0.8 и 0.2 в результирующем соответственно.
        df['rating']  = (df.add_to_cart_coef * 0.8) + (df.reordered * 0.2)
        #Найдем сумму по всем позициям в связке продукт-пользователь:
        df['rating'] = df.groupby(['user_id','product_id'])['rating'].transform('sum')
        df['rating'] = df['rating'].clip(lower = 0) #удалим значения меньше нуля, как неактуальные
        #Воспользуемся MinMaxScaler для слишком больших значений рейтинга, причем сделаем это в группировке по пользователю:
        autoscaler = MinMaxScaler(feature_range = (1,10))
        def sc(row):
            return autoscaler.fit_transform(row.values.reshape(-1,1))

        df['rating'] = df.groupby('user_id')['rating'].apply(sc).explode().values.astype(float)
    
        #Вычислим ряд коэффициентов в группировке пользователь+продукт

        df['pu_orders'] = df.groupby(['user_id','product_id'])['order_number'].transform('count')
        df['max_orders'] = df.groupby(['user_id','product_id'])['order_number'].transform('max')
        df['min_orders'] = df.groupby(['user_id','product_id'])['order_number'].transform('min')
        df['pu_order_ratio'] = df['pu_orders']/(df['max_orders'] - df['min_orders'] + 1)
        df['pu_median_dspo'] = df.groupby(['user_id','product_id'])['days_since_prior_order'].transform('median')
        df['pu_median_cart_pos'] = df.groupby(['user_id','product_id'])['add_to_cart_order'].transform('median')
        df['p_reordered_times'] = df.groupby(['product_id'])['reordered'].transform('sum')
        num_cols = ['pu_orders','max_orders','min_orders','pu_order_ratio','pu_median_dspo','pu_median_cart_pos','p_reordered_times']
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce', downcast='float')
    
        #Округлим float, и удалим лишние колонки
        df = df.round(4)
        df.drop(columns = ['order_num_coef','max_orders','min_orders'], inplace = True)
        df = self._reduce_mem_usage(df)
        df = df.replace([np.inf, -np.inf], 0)
        #Очистка памяти:
        gc.collect()
        return df
           
    #Расчет признаков, относящихся к пользователю (user-based)
    def _get_feature_by_user(self, df):
        res = list()
        for i, v in tqdm(df.groupby('user_id')):
            res.append(
                (
                    i,
                    len(v['product_id']),
                    v['days_since_prior_order'].median(),
                    v['order_number'].max(),
                    v['add_to_cart_order'].max(),
                    v['order_hour_of_day'].median(),
                    v['order_dow'].median()
                )
            )
    
        res = pd.DataFrame(
            res,
            columns=[
                'user_id', 'u_prods_count', 'u_median_dspo' ,'u_max_orders', 'u_max_ordlen','u_median_hod', 'u_median_dow'
            ])
        #На всякий случай, заменим также inf (если такие будут) на нули:
        res = res.replace([np.inf, -np.inf], 0)

        res = self._reduce_mem_usage(res)
        return res
    
    #Расчет признаков, относящихся к продукту (product-based)
    def _get_feature_by_product(self,df):
        res = list()
        for i, v in tqdm(df.groupby('product_id')):
            res.append(
                (
                    i,
                    len(v['user_id']),
                    v['add_to_cart_order'].median(),
                    v['order_dow'].median(),
                    v['order_hour_of_day'].median(),
                    v['days_since_prior_order'].median()
                )
            )
    
        res = pd.DataFrame(
            res,
            columns=[
                'product_id', 'p_user_cnt','p_median_cart_pos', 'p_median_dow', 'p_median_hod','p_median_dspo'])
        
        res = res.replace([np.inf, -np.inf], 0)

        res = self._reduce_mem_usage(res)
        return res
    
    
    #Расчет по всем показателям выше:
    def _get_model_features(self):
        self.load_mainset()
        print('Mainset loaded')
        df = self._mainset.groupby(['user_id', 'product_id'], as_index = False)\
            .agg({**{feats:'median' for feats in self._feature_list}})
        
        X_u = self._get_feature_by_user(self._mainset)
        print('User feats calculated')
        gc.collect()
        merged = reduce(lambda left, right: pd.merge(left, right, on='user_id', how='inner'), [X_u,df])
        
        print('User features merged')
    
        del [X_u,df]
        gc.collect()
    
        X_p = self._get_feature_by_product(self._mainset)
        print('Product feats calculated')
        gc.collect()
        merged = reduce(lambda left, right: pd.merge(left, right, on='product_id', how='inner'), [X_p,merged])
        print('Product features merged')
    
        del [X_p]
        gc.collect()
    
        merged.fillna(0, inplace=True)
    

        merged.sort_values(by=['user_id', 'product_id'], inplace=True)

        
        ordered_cols = ['user_id', 'product_id', 'p_user_cnt', 'p_median_cart_pos', 'p_reordered_times', 'p_median_dow', 
                        'p_median_hod', 'p_median_dspo', 'u_prods_count', 'u_median_dspo', 'u_max_orders', 'u_max_ordlen', 
                        'u_median_hod', 'u_median_dow', 'pu_orders', 'pu_median_dspo', 'pu_order_ratio', 'pu_median_cart_pos']
        
        features_cols = list(merged.drop(columns=['rating']).columns)
        
        #Для уже обученной модели важен порядок следования столбцов, поэтому:
        if set(ordered_cols) == set(features_cols):
            df_x = merged[ordered_cols]
        #А в случае если необходимо переобучить модель на новом сете:    
        else:
            df_x = merged[features_cols]

        df_x['qid'] = df_x['user_id']
    
        df_y = merged[['user_id', 'product_id','rating']]
        df_y['rating'] = df_y['rating'].apply(np.int64)
    
        del merged
        gc.collect()
        df_x = self._reduce_mem_usage(df_x)
        return df_x, df_y
            
    
    #И общий метод, который объединяет всё вышенаписанное:
    def preprocess_data(self, new_products = None, new_transactions = None):
        #Если меняем всё:
        if (new_products is not None) & (new_transactions is not None):
            self.renew_products(new_products).renew_transactions(new_transactions)
        #Если меняем одну часть:
        elif new_products is not None:
            self.renew_products(new_products)
        elif new_transactions is not None:
            self.renew_transactions(new_transactions)
            
        #Объединим products и transactions, вычислим необходимые столбцы и найдем X и y:
        X_train, y_train = self._get_model_features()
        return X_train, y_train

#Итоговый класс retrain model
class MyRecsys():
    def __init__(self, xgb_ranker = None, als_model = None, transactions = None, products = None,
                  X_train = None,y_train = None, n = 10, r_xgb = 0.3, r_als =  0.7, user_treshold = 50, mainset = None,
                  fill_empty = None, cols_to_leave = ['order_id', 'user_id', 'order_number','product_id', 'add_to_cart_order',
                                  'reordered',"order_dow","order_hour_of_day",'days_since_prior_order'], #что нужно для вычислений
                  feature_list = ['rating','pu_orders','p_reordered_times','pu_median_dspo','pu_order_ratio','pu_median_cart_pos']):
            #Определим модели:
            self._xgb_ranker = xgb_ranker
            self._als_model = als_model
            #Определим данные:
            self._transactions = transactions
            self._products = products
            self._mainset = mainset
            #Также определим данные для подачи в модель (уже обработанные X и y):
            self._X_train = X_train
            self._y_train = y_train
            #И значения по умолчанию для обучения моделей:
            self._fill_empty = fill_empty
            self._cols_to_leave = cols_to_leave
            self._feature_list = feature_list
            
            self._n = n #к-во продуктов в предсказании, в условиях задания равно 10
            self._r_xgb =  r_xgb #xgb ratio - доля предсказания модели бустинга в итоговом предсказании, по умолчанию = 0.3
            self._r_als = r_als  #als ratio - доля предсказания модели ALS в итоговом предсказании, по умолчанию = 0.7
            self._user_treshold = user_treshold #минимальное к-во пользователей, заказавших товар, для предскзания по популярности
    
    #Запуск наших классов с определенным набором параметров
    #Для класса работы с моделями:
    def load_model_wrapper(self):
        model_wrapper = RecsysModel(xgb_ranker = self._xgb_ranker, als_model = self._als_model, 
                                    transactions = self._transactions, X_train = self._X_train, y_train = self._y_train, 
                                    n = self._n, r_xgb = self._r_xgb, r_als =  self._r_als, 
                                    user_treshold = self._user_treshold)
        return model_wrapper
            
    #Для класса работы с данными для моделей:
    def load_data_formatter(self):
        data_formatter = FormatDataset(transactions = self._transactions, products = self._products, mainset = self._mainset,
                                       fill_empty = self._fill_empty, cols_to_leave = self._cols_to_leave, 
                                       feature_list = self._feature_list)
        return data_formatter
            
    
    #И итоговый метод: 
    def process_and_predict(self, usr_ids, products_addition = None, transactions_addition = None, refit = False,
                           eval_metric = ['ndcg@10'], lambdarank_num_pair_per_sample = 15, n_estimators =75, 
                            min_child_weight = 12, max_depth = 11, learning_rate = 0.1, bsl_method = 'als', 
                        bsl_epochs = 20, reg_u = 18, reg_i = 6):
        #Загрузим класс для обработки дынных
        data_formatter = self.load_data_formatter()
        
        #Обновим X и y:
        self._X_train,  self._y_train = data_formatter.preprocess_data(new_products = products_addition, 
                                                                      new_transactions = transactions_addition)
        #Загрузка класса для работы с моделями:
        model_wrapper = self.load_model_wrapper()
        if refit:
            model_wrapper.refit_model(eval_metric = eval_metric, lambdarank_num_pair_per_sample = lambdarank_num_pair_per_sample, 
                            n_estimators = n_estimators, min_child_weight = min_child_weight,
                            max_depth = max_depth, learning_rate = learning_rate, bsl_method = bsl_method, 
                            bsl_epochs = bsl_epochs, reg_u = reg_u, reg_i = reg_i)
            
        prediction = model_wrapper.predict(ids = usr_ids)
        return prediction

if len(sys.argv) > 1:  
	parser=argparse.ArgumentParser()
	parser.add_argument("-user_ids", '-ui', nargs='+', type=int, help= "User id (as int) or array of user ids (int) to make prediction")
	parser.add_argument("--n", '-n', type=int, help= "Products count (as int)  to be predicted for each user", default = 10)
	parser.add_argument("--new_products", '-p', help= "New product data to update. Dataframe or path to such", default = None)
	parser.add_argument("--new_transactions", '-t', help= "New transactions to update. Dataframe or path to such", default = None)
	parser.add_argument("--refit",'-r', help= "If model needs to be refit pass 'True' to refit", default = False)
	print(parser.format_help())

	args=parser.parse_args()
	if not args.user_ids:
		raise ValueError('No user ids passed for prediction!')
	recsys_wrap = MyRecsys(n = args.n)
	preds = recsys_wrap.process_and_predict(usr_ids = args.user_ids, products_addition = args.new_products, 
                               transactions_addition = args.new_transactions, refit = args.refit)
	print(preds)

else:
	try:
		usr_ids = input("Enter user id or several user ids (separated with space) to get prediction: ")
		usr_ids = [int(x) for x in usr_ids.split()]
	except ValueError:
		print('An int or list of int values is requiered')
	if  not usr_ids:
		raise ValueError('No user ids passed for prediction!')

	try:
		n_count = input("Enter number of products in prediction or leave blank:")
		#set default value
		if len (n_count) < 1:
			n_count = 10
		#else try to convert to int
		else:
			n_count = int(n_count)
			
		
	except ValueError:
		print('An int value is requiered')

	products_input = input("Enter new product dataframe/path to such (if needed) or leave blank: ")
	transactions_input = input("Enter new transactions dataframe/path to such(if needed) or leave blank: ")
	

	refit_flag = input("If model needs to be refit print any value, else just press Enter:")
	if len(refit_flag)>1:
		refit_flag=True
	else:
		refit_flag=False

	
	recsys_wrap = MyRecsys(n = n_count)
	if len(products_input)>0:
		products_input = products_input
	else:
		products_input = None

	if len(transactions_input)>0:
		transactions_input = transactions_input
	else:
		transactions_input = None
	
			
	
	preds = recsys_wrap.process_and_predict(usr_ids = usr_ids, products_addition = products_input, transactions_addition = transactions_input, 
		refit = refit_flag)
	print(preds)
		
