import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class ArRecommenderFast:
    def __init__(
        self,
        top_n=20,
        min_count=1,
        min_support=0.0,
        exclude_same_course=False,
        score_mode="count",   # count | support | confidence | cosine | lift
    ):
        self.top_n = top_n
        self.min_count = min_count
        self.min_support = min_support
        self.exclude_same_course = exclude_same_course
        self.score_mode = score_mode

        self.recs_df = None
        self.course_map_ = None
        self.lesson_counts_ = None
        self.lesson_index_ = None
        self.popular_lessons_ = None

    def fit(self, data: pd.DataFrame):
        req_cols = {"order_id", "lesson_id", "course_id"}
        miss = req_cols - set(data.columns)
        if miss:
            raise KeyError(f"В датасете нет столбцов: {miss}")

        # 1) только нужные колонки + убираем дубли внутри заказа
        base = data[["order_id", "lesson_id", "course_id"]].drop_duplicates()

        # 2) lesson -> course
        lesson_course = (
            base[["lesson_id", "course_id"]]
            .drop_duplicates("lesson_id")
            .set_index("lesson_id")["course_id"]
        )
        self.course_map_ = lesson_course

        # 3) кодируем order_id и lesson_id в индексы
        order_codes, order_uniques = pd.factorize(base["order_id"], sort=False)
        lesson_codes, lesson_uniques = pd.factorize(base["lesson_id"], sort=False)

        self.lesson_index_ = pd.Index(lesson_uniques)

        # 4) sparse binary matrix: order x lesson
        X = csr_matrix(
            (
                np.ones(len(base), dtype=np.uint8),
                (order_codes, lesson_codes)
            ),
            shape=(len(order_uniques), len(lesson_uniques)),
            dtype=np.uint32
        )

        n_orders = X.shape[0]

        # 5) частоты уроков: в скольких заказах встречался урок
        lesson_counts = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        self.lesson_counts_ = pd.Series(lesson_counts, index=lesson_uniques)

        self.popular_lessons_ = (
            pd.Series(lesson_counts, index=lesson_uniques)
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        # 6) item-item co-occurrence
        co = (X.T @ X).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()

        # course array aligned to lesson_uniques
        course_arr = lesson_course.reindex(lesson_uniques).to_numpy()

        # 7) вместо materialize всех пар сразу забираем top-N по каждой строке
        rows_A = []
        rows_B = []
        scores = []
        pair_counts = []
        ranks = []

        indptr = co.indptr
        indices = co.indices
        data_vals = co.data.astype(np.float32)

        for i in range(co.shape[0]):
            start, end = indptr[i], indptr[i + 1]
            if start == end:
                continue

            idx = indices[start:end]
            cnt = data_vals[start:end]

            mask = np.ones(len(idx), dtype=bool)

            # min_count
            if self.min_count > 1:
                mask &= (cnt >= self.min_count)

            # exclude same course
            if self.exclude_same_course:
                mask &= (course_arr[idx] != course_arr[i])

            # min_support
            if self.min_support > 0:
                support_vals = cnt / n_orders
                mask &= (support_vals >= self.min_support)

            if not mask.any():
                continue

            idx = idx[mask]
            cnt = cnt[mask]

            # score
            if self.score_mode == "count":
                score = cnt

            elif self.score_mode == "support":
                score = cnt / n_orders

            elif self.score_mode == "confidence":
                # conf(A -> B) = count(A,B) / count(A)
                denom = lesson_counts[i]
                score = cnt / denom if denom > 0 else np.zeros_like(cnt)

            elif self.score_mode == "cosine":
                # count(A,B) / sqrt(count(A)*count(B))
                score = cnt / np.sqrt(lesson_counts[i] * lesson_counts[idx])

            elif self.score_mode == "lift":
                # count(A,B)*N / (count(A)*count(B))
                score = cnt * n_orders / (lesson_counts[i] * lesson_counts[idx])

            else:
                raise ValueError(
                    "score_mode должен быть одним из: "
                    "count, support, confidence, cosine, lift"
                )

            k = min(self.top_n, len(score))
            if k == 0:
                continue

            # быстрее, чем full sort
            if len(score) > k:
                top_local = np.argpartition(-score, k - 1)[:k]
            else:
                top_local = np.arange(len(score))

            # финальная сортировка только top-k
            top_local = top_local[np.argsort(-score[top_local], kind="stable")]

            rows_A.extend([i] * len(top_local))
            rows_B.extend(idx[top_local].tolist())
            scores.extend(score[top_local].tolist())
            pair_counts.extend(cnt[top_local].tolist())
            ranks.extend(np.arange(1, len(top_local) + 1).tolist())

        recs_df = pd.DataFrame({
            "lesson_A": lesson_uniques[np.array(rows_A, dtype=np.int32)],
            "lesson_B": lesson_uniques[np.array(rows_B, dtype=np.int32)],
            "score": np.array(scores, dtype=np.float32),
            "pair_count": np.array(pair_counts, dtype=np.float32),
            "rank": np.array(ranks, dtype=np.int16),
        })

        self.recs_df = recs_df
        return self

    def get_recommendations_table(self, only_score=True):
        if self.recs_df is None:
            raise ValueError("Сначала вызови fit()")

        if only_score:
            return self.recs_df[["lesson_A", "lesson_B", "score"]].copy()

        return self.recs_df.copy()

    def recommend_for_basket(self, basket, top_n=20):
        if self.recs_df is None:
            raise ValueError("Сначала вызови fit()")

        seen = set(basket)

        cand = self.recs_df[self.recs_df["lesson_A"].isin(seen)].copy()
        cand = cand[~cand["lesson_B"].isin(seen)]

        if cand.empty:
            fallback = [x for x in self.popular_lessons_ if x not in seen][:top_n]
            return pd.DataFrame({
                "lesson_B": fallback,
                "score": [np.nan] * len(fallback)
            })

        out = (
            cand.groupby("lesson_B", as_index=False)
            .agg(score=("score", "sum"))
            .sort_values(["score", "lesson_B"], ascending=[False, True])
            .head(top_n)
            .reset_index(drop=True)
        )

        return out



class ArRecommenderFast:
    def __init__(
        self,
        top_n=20,
        min_count=1,
        min_support=0.0,
        exclude_same_course=False,
        score_mode="count",   # count | support | confidence | cosine | lift
    ):
        self.top_n = top_n
        self.min_count = min_count
        self.min_support = min_support
        self.exclude_same_course = exclude_same_course
        self.score_mode = score_mode

        self.recs_df = None
        self.course_map_ = None
        self.lesson_counts_ = None
        self.lesson_index_ = None
        self.popular_lessons_ = None

    def fit(self, data: pd.DataFrame):
        req_cols = {"order_id", "lesson_id", "course_id"}
        miss = req_cols - set(data.columns)
        if miss:
            raise KeyError(f"В датасете нет столбцов: {miss}")

        # 1) только нужные колонки + убираем дубли внутри заказа
        base = data[["order_id", "lesson_id", "course_id"]].drop_duplicates()

        # 2) lesson -> course
        lesson_course = (
            base[["lesson_id", "course_id"]]
            .drop_duplicates("lesson_id")
            .set_index("lesson_id")["course_id"]
        )
        self.course_map_ = lesson_course

        # 3) кодируем order_id и lesson_id в индексы
        order_codes, order_uniques = pd.factorize(base["order_id"], sort=False)
        lesson_codes, lesson_uniques = pd.factorize(base["lesson_id"], sort=False)

        self.lesson_index_ = pd.Index(lesson_uniques)

        # 4) sparse binary matrix: order x lesson
        X = csr_matrix(
            (
                np.ones(len(base), dtype=np.uint8),
                (order_codes, lesson_codes)
            ),
            shape=(len(order_uniques), len(lesson_uniques)),
            dtype=np.uint32
        )

        n_orders = X.shape[0]

        # 5) частоты уроков: в скольких заказах встречался урок
        lesson_counts = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        self.lesson_counts_ = pd.Series(lesson_counts, index=lesson_uniques)

        self.popular_lessons_ = (
            pd.Series(lesson_counts, index=lesson_uniques)
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        # 6) item-item co-occurrence
        co = (X.T @ X).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()

        # course array aligned to lesson_uniques
        course_arr = lesson_course.reindex(lesson_uniques).to_numpy()

        # 7) вместо materialize всех пар сразу забираем top-N по каждой строке
        rows_A = []
        rows_B = []
        scores = []
        pair_counts = []
        ranks = []

        indptr = co.indptr
        indices = co.indices
        data_vals = co.data.astype(np.float32)

        for i in range(co.shape[0]):
            start, end = indptr[i], indptr[i + 1]
            if start == end:
                continue

            idx = indices[start:end]
            cnt = data_vals[start:end]

            mask = np.ones(len(idx), dtype=bool)

            # min_count
            if self.min_count > 1:
                mask &= (cnt >= self.min_count)

            # exclude same course
            if self.exclude_same_course:
                mask &= (course_arr[idx] != course_arr[i])

            # min_support
            if self.min_support > 0:
                support_vals = cnt / n_orders
                mask &= (support_vals >= self.min_support)

            if not mask.any():
                continue

            idx = idx[mask]
            cnt = cnt[mask]

            # score
            if self.score_mode == "count":
                score = cnt

            elif self.score_mode == "support":
                score = cnt / n_orders

            elif self.score_mode == "confidence":
                # conf(A -> B) = count(A,B) / count(A)
                denom = lesson_counts[i]
                score = cnt / denom if denom > 0 else np.zeros_like(cnt)

            elif self.score_mode == "cosine":
                # count(A,B) / sqrt(count(A)*count(B))
                score = cnt / np.sqrt(lesson_counts[i] * lesson_counts[idx])

            elif self.score_mode == "lift":
                # count(A,B)*N / (count(A)*count(B))
                score = cnt * n_orders / (lesson_counts[i] * lesson_counts[idx])

            else:
                raise ValueError(
                    "score_mode должен быть одним из: "
                    "count, support, confidence, cosine, lift"
                )

            k = min(self.top_n, len(score))
            if k == 0:
                continue

            # быстрее, чем full sort
            if len(score) > k:
                top_local = np.argpartition(-score, k - 1)[:k]
            else:
                top_local = np.arange(len(score))

            # финальная сортировка только top-k
            top_local = top_local[np.argsort(-score[top_local], kind="stable")]

            rows_A.extend([i] * len(top_local))
            rows_B.extend(idx[top_local].tolist())
            scores.extend(score[top_local].tolist())
            pair_counts.extend(cnt[top_local].tolist())
            ranks.extend(np.arange(1, len(top_local) + 1).tolist())

        recs_df = pd.DataFrame({
            "lesson_A": lesson_uniques[np.array(rows_A, dtype=np.int32)],
            "lesson_B": lesson_uniques[np.array(rows_B, dtype=np.int32)],
            "score": np.array(scores, dtype=np.float32),
            "pair_count": np.array(pair_counts, dtype=np.float32),
            "rank": np.array(ranks, dtype=np.int16),
        })

        self.recs_df = recs_df
        return self

    def get_recommendations_table(self, only_score=True):
        if self.recs_df is None:
            raise ValueError("Сначала вызови fit()")

        if only_score:
            return self.recs_df[["lesson_A", "lesson_B", "score"]].copy()

        return self.recs_df.copy()

    def recommend_for_basket(self, basket, top_n=20):
        if self.recs_df is None:
            raise ValueError("Сначала вызови fit()")

        seen = set(basket)

        cand = self.recs_df[self.recs_df["lesson_A"].isin(seen)].copy()
        cand = cand[~cand["lesson_B"].isin(seen)]

        if cand.empty:
            fallback = [x for x in self.popular_lessons_ if x not in seen][:top_n]
            return pd.DataFrame({
                "lesson_B": fallback,
                "score": [np.nan] * len(fallback)
            })

        out = (
            cand.groupby("lesson_B", as_index=False)
            .agg(score=("score", "sum"))
            .sort_values(["score", "lesson_B"], ascending=[False, True])
            .head(top_n)
            .reset_index(drop=True)
        )

        return out