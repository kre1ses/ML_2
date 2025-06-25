import numpy as np
from scipy.stats import chi

from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import signature


class FeatureCreatorPlaceholder(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, new_dim, func: Callable = np.cos):
        self.n_features = n_features
        self.new_dim = new_dim
        self.w = None
        self.b = None
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class RandomFeatureCreator(FeatureCreatorPlaceholder):
    def fit(self, X, y=None):
        # raise NotImplementedError
        # sigma = ...
        # self.w = ...
        # self.b = ...
        n_samples, n_feature_of_X = X.shape
        indexes = np.random.choice(n_samples, size=(n_samples // 3, 2), replace=False)
        sigma = np.sqrt(np.median(np.sum((X[indexes[:, 0]] - X[indexes[:, 1]]) ** 2, axis=1)))
        
        self.w = np.random.normal(0, 1 / sigma, size=(self.n_features, X.shape[1]))
        self.b = np.random.uniform(-np.pi, np.pi, size=(self.n_features))
        return self

    def transform(self, X, y=None):
        # raise NotImplementedError
        return self.func(np.dot(X, self.w.T) + self.b)


class OrthogonalRandomFeatureCreator(RandomFeatureCreator):
    def fit(self, X, y=None):
        # raise NotImplementedError
        super().fit(X, y)

        n_samples, n_feature_of_X = X.shape
        indexes = np.random.choice(n_samples, size=(n_samples // 3, 2), replace=False)
        sigma = np.sqrt(np.median(np.sum((X[indexes[:, 0]] - X[indexes[:, 1]]) ** 2, axis=1)))

        G = np.random.normal(0, 1, (self.n_features, X.shape[1]))
        Q, R = np.linalg.qr(G)
        S = np.diag(chi.rvs(df=self.n_features, size=self.n_features))

        self.w = 1/sigma * np.dot(S, Q) 
        self.b = np.random.uniform(-np.pi, np.pi, size=self.n_features)
        return self

    def transform(self, X, y=None):
        if self.n_features < X.shape[1]:
            X = X[:, :self.n_features]
        return self.func(X @ self.w.T + self.b)


class RFFPipeline(BaseEstimator):
    """
    Пайплайн, делающий последовательно три шага:
        1. Применение PCA
        2. Применение RFF
        3. Применение классификатора
    """
    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=FeatureCreatorPlaceholder,
            classifier_class=LogisticRegression,
            classifier_params=None,
            func=np.cos,
    ):
        """
        :param n_features: Количество признаков, генерируемых RFF
        :param new_dim: Количество признаков, до которых сжимает PCA
        :param use_PCA: Использовать ли PCA
        :param feature_creator_class: Класс, создающий признаки, по умолчанию заглушка
        :param classifier_class: Класс классификатора
        :param classifier_params: Параметры, которыми инициализируется классификатор
        :param func: Функция, которую получает feature_creator при инициализации.
                     Если не хотите, можете не использовать этот параметр.
        """
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        if classifier_params is None:
            classifier_params = {}
        self.classifier = classifier_class(**classifier_params)
        self.feature_creator = feature_creator_class(
            n_features=self.n_features, new_dim=self.new_dim, func=func
        )
        self.pipeline = None

    def fit(self, X, y):
        pipeline_steps = []
        if self.use_PCA:
            self.new_dim = X.shape[1]
            # pipeline_steps: list[tuple] = ...  # todo!
            pipeline_steps.append(("pca", PCA(n_components=self.new_dim)))
        pipeline_steps.append(("rff", self.feature_creator))
        pipeline_steps.append(("classifier", self.classifier))
        self.pipeline = Pipeline(pipeline_steps).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)
