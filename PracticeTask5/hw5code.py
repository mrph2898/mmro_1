import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator


def gini(sorted_feature_vec):
    N = len(sorted_feature_vec)
    N_1 = sum(sorted_feature_vec)

    def H_R(p1, p0):
        return 1 - p1**2 - p0**2

    amnt_of_obj_in_l_node = np.arange(1, N)
    amnt_of_1_obj_in_l_node = sorted_feature_vec[:-1].cumsum()
    pl_1 = amnt_of_1_obj_in_l_node / amnt_of_obj_in_l_node

    amnt_of_obj_in_r_node = amnt_of_obj_in_l_node[::-1]
    amnt_of_1_obj_in_r_node = N_1 - amnt_of_1_obj_in_l_node
    pr_1 = amnt_of_1_obj_in_r_node / amnt_of_obj_in_r_node

    return -((amnt_of_obj_in_l_node / N) * H_R(pl_1, 1 - pl_1) +
             (amnt_of_obj_in_r_node / N) * H_R(pr_1, 1 - pr_1))


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # thr = np.convolve(sorted(feature_vector), [0.5,0.5], 'valid')
    mask = np.argsort(feature_vector)
    shl_feature_vec = feature_vector[mask][1:]
    shr_feature_vec = feature_vector[mask][:-1]
    unique_mask = shl_feature_vec != shr_feature_vec

    thr = ((shl_feature_vec + shr_feature_vec) / 2)[unique_mask]
    ginis = gini(target_vector[mask])[unique_mask]
    return thr, ginis, thr[np.argmax(ginis)], max(ginis)


# В результате проведения экспериментов, понял, что надо отнаследоваться от
# BaseEstimator, чтобы были метод get_params и не только
class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        # Здесь максимально неудобная ошибка - казалось бы, всё нормально с
        # названием, но cross_val_score говорит, что feature_types=None, поэтому
        # убрал _
        self.feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        # Ох уж это равенство:
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        # Сомнения, что range правильный - оправдались, т.к. шли с 1
        for feature in range(sub_X.shape[1]):
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    # Здесь было перепутано деление
                    ratio[key] = current_click / current_count
                # Тут, как мне кажется, хотели бы просто отсортировать keys()
                # по значениям величин, а не сортировать элементы, исходя из их
                # величины:
                sorted_categories = sorted(ratio.keys(), key=lambda x: ratio[x])
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError
            # Вот тут хотели проверить, как и в начале _fit_node, только на
            # равенство, чтобы продолжить цикл
            if np.all(feature_vector == feature_vector[0]):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                # Эммм, опечатка с большой буквой?)
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            # most_common не возвращает то, что нам хотелось бы ->
            # индексирование забыли
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        # Забытое отрицание в sub_y
        self._fit_node(sub_X[np.logical_not(split)],
                       sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        elif node["type"] == "nonterminal":
            type_of_feature = self.feature_types[node["feature_split"]]
            if type_of_feature == "real":
                if x[node["feature_split"]] < node["threshold"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            elif type_of_feature == "categorical":
                if x[node["feature_split"]] in node["categories_split"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
