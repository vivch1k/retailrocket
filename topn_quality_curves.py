"""
Top-N Quality Curves for Association Filtering / Co-occurrence Recommender.

Что считает файл:
1. Cumulative Top-N curves:
   top-5000, top-7000, top-10000, top-15000, top-20000, top-30000.
2. Item-to-basket offline metrics:
   для каждого урока в корзине предсказываем остальные,
   усредняем сначала внутри корзины, потом по корзинам.
3. Basket-to-basket offline metrics.
4. Catalog coverage.
5. Basket coverage.
6. Same-course share.
7. Pair strength stats.
8. Popularity bucket metrics:
   качество по сегментам популярности input lesson.

Ожидаемые колонки в данных:
    order_id, lesson_id, course_id

Ожидаемые колонки в recs_df модели:
    lesson_A, lesson_B, score
Желательно:
    pair_count, rank

Если интерфейс твоей модели отличается, правь только функцию:
    fit_model_and_get_recs_df()
"""

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# =========================
# DEFAULT CONFIG
# =========================

TOP_NS = [5000, 7000, 10000, 15000, 20000, 30000]
TOP_K = 20
MIN_COUNT = 10
RANDOM_STATE = 42
TEST_SIZE = 0.1

TRAIN_MODES = [
    "2plus_courses",
    "2plus_lessons",
]

# True:
#   обучаем модель только на top-N популярных уроках.
#
# False:
#   обучаемся на всей обучающей выборке,
#   но итоговую матрицу рекомендаций режем до top-N.
STRICT_TOPN_TRAINING = True


# =========================
# DATA SPLIT AND FILTERS
# =========================

def split_by_order(
    df: pd.DataFrame,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делит данные по order_id, чтобы один order_id не попадал одновременно в train и test.
    """
    rng = np.random.default_rng(random_state)

    order_ids = df["order_id"].drop_duplicates().to_numpy()
    rng.shuffle(order_ids)

    test_n = int(len(order_ids) * test_size)

    test_orders = set(order_ids[:test_n])
    train_orders = set(order_ids[test_n:])

    train_df = df[df["order_id"].isin(train_orders)].copy()
    test_df = df[df["order_id"].isin(test_orders)].copy()

    return train_df, test_df


def filter_orders_by_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode:
        "2plus_lessons" — заказы с >= 2 уникальными lesson_id
        "2plus_courses" — заказы с >= 2 уникальными course_id
    """

    if mode == "2plus_lessons":
        good_orders = (
            df.groupby("order_id")["lesson_id"]
            .nunique()
            .loc[lambda x: x >= 2]
            .index
        )

    elif mode == "2plus_courses":
        good_orders = (
            df.groupby("order_id")["course_id"]
            .nunique()
            .loc[lambda x: x >= 2]
            .index
        )

    else:
        raise ValueError("mode должен быть '2plus_lessons' или '2plus_courses'")

    return df[df["order_id"].isin(good_orders)].copy()


def get_top_popular_lessons(train_df: pd.DataFrame, top_n: int) -> pd.Index:
    """
    Популярность считаем только на train, чтобы не было leakage из test.
    """
    return (
        train_df
        .groupby("lesson_id")["order_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )


def restrict_to_top_lessons(df: pd.DataFrame, top_lessons: Iterable[Any]) -> pd.DataFrame:
    top_set = set(top_lessons)
    return df[df["lesson_id"].isin(top_set)].copy()


# =========================
# RECOMMENDATIONS HELPERS
# =========================

def make_top_map(recs_df: pd.DataFrame, top_k: int = 20) -> Dict[Any, List[Any]]:
    """
    Делает словарь:
        lesson_A -> [lesson_B1, lesson_B2, ...]
    """

    recs_df = recs_df.copy()

    if recs_df.empty:
        return {}

    if "rank" in recs_df.columns:
        recs_df = recs_df.sort_values(["lesson_A", "rank"])
    else:
        recs_df = recs_df.sort_values(
            ["lesson_A", "score"],
            ascending=[True, False],
        )

    top_map = (
        recs_df
        .groupby("lesson_A", sort=False)
        .head(top_k)
        .groupby("lesson_A")["lesson_B"]
        .apply(list)
        .to_dict()
    )

    return top_map


def filter_recs_to_top_lessons(
    recs_df: pd.DataFrame,
    top_lessons: Iterable[Any],
    filter_A: bool = True,
    filter_B: bool = True,
) -> pd.DataFrame:
    """
    Фильтрация итоговой матрицы рекомендаций до top-N популярных уроков.
    """
    top_set = set(top_lessons)

    out = recs_df.copy()

    if filter_A:
        out = out[out["lesson_A"].isin(top_set)]

    if filter_B:
        out = out[out["lesson_B"].isin(top_set)]

    return out.copy()


# =========================
# RANKING METRICS
# =========================

def _calc_ranking_metrics_for_one_query(
    recs: Iterable[Any],
    target: set,
    top_k: int = 20,
) -> Optional[Dict[str, float]]:
    """
    recs:
        list рекомендаций

    target:
        set релевантных lesson_id
    """

    if len(target) == 0:
        return None

    recs_clean = []
    seen_recs = set()

    for r in recs:
        if pd.isna(r):
            continue

        if r in seen_recs:
            continue

        seen_recs.add(r)
        recs_clean.append(r)

        if len(recs_clean) >= top_k:
            break

    hits_arr = np.array([1 if r in target else 0 for r in recs_clean], dtype=np.int8)
    hits = int(hits_arr.sum())

    # Важно: precision делим на top_k, как в твоей исходной логике.
    precision = hits / top_k
    recall = hits / len(target)
    hitrate = int(hits > 0)

    # AP@K
    if hits > 0:
        cum_hits = np.cumsum(hits_arr)
        precision_at_i = cum_hits / np.arange(1, len(hits_arr) + 1)
        ap = float((precision_at_i * hits_arr).sum() / min(len(target), top_k))
    else:
        ap = 0.0

    # NDCG@K
    if len(hits_arr) > 0:
        discounts = 1 / np.log2(np.arange(2, len(hits_arr) + 2))
        dcg = float((hits_arr * discounts).sum())
    else:
        dcg = 0.0

    ideal_len = min(len(target), top_k)
    if ideal_len > 0:
        ideal_discounts = 1 / np.log2(np.arange(2, ideal_len + 2))
        idcg = float(ideal_discounts.sum())
    else:
        idcg = 0.0

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "hitrate": float(hitrate),
        "map": float(ap),
        "ndcg": float(ndcg),
    }


def calc_item_to_basket_metrics_order_avg(
    test_df: pd.DataFrame,
    top_map: Dict[Any, List[Any]],
    top_k: int = 20,
    desc: str = "item-to-basket",
) -> Dict[str, float]:
    """
    Твоя item-to-basket метрика:

    1. Для каждого урока в корзине делаем рекомендации.
    2. target = остальные уроки этой корзины.
    3. Считаем метрики по каждому input-уроку.
    4. Усредняем метрики внутри корзины.
    5. Потом усредняем по корзинам.

    Это отличается от query-average тем, что большие корзины не получают
    слишком большой вес за счет большого числа input-уроков.
    """

    orders = (
        test_df
        .groupby("order_id")["lesson_id"]
        .apply(lambda x: list(pd.unique(x.dropna())))
        .tolist()
    )

    order_metrics = []
    total_queries = 0

    for basket in tqdm(orders, desc=desc):
        if len(basket) < 2:
            continue

        basket_set = set(basket)
        curr_order_metrics = []

        for les in basket:
            target = basket_set - {les}

            if len(target) == 0:
                continue

            recs = top_map.get(les, [])
            recs = [r for r in recs if r != les]

            m = _calc_ranking_metrics_for_one_query(
                recs=recs,
                target=target,
                top_k=top_k,
            )

            if m is not None:
                curr_order_metrics.append(m)
                total_queries += 1

        if curr_order_metrics:
            order_metrics.append({
                key: float(np.mean([m[key] for m in curr_order_metrics]))
                for key in curr_order_metrics[0].keys()
            })

    if not order_metrics:
        return {
            f"item_prec@{top_k}": np.nan,
            f"item_recall@{top_k}": np.nan,
            f"item_hitrate@{top_k}": np.nan,
            f"item_map@{top_k}": np.nan,
            f"item_ndcg@{top_k}": np.nan,
            "item_n_orders": 0,
            "item_n_queries": 0,
        }

    return {
        f"item_prec@{top_k}": float(np.mean([m["precision"] for m in order_metrics])),
        f"item_recall@{top_k}": float(np.mean([m["recall"] for m in order_metrics])),
        f"item_hitrate@{top_k}": float(np.mean([m["hitrate"] for m in order_metrics])),
        f"item_map@{top_k}": float(np.mean([m["map"] for m in order_metrics])),
        f"item_ndcg@{top_k}": float(np.mean([m["ndcg"] for m in order_metrics])),
        "item_n_orders": len(order_metrics),
        "item_n_queries": total_queries,
    }


def recommend_for_basket_from_top_map(
    basket: Iterable[Any],
    top_map: Dict[Any, List[Any]],
    top_k: int = 20,
    per_item_k: int = 50,
) -> List[Any]:
    """
    Агрегация рекомендаций от нескольких уроков корзины.

    Используется reciprocal rank:
        score += 1 / rank

    Это простой и устойчивый способ собрать общую рекомендацию для корзины.
    """

    seen = set(basket)
    scores = defaultdict(float)

    for les in seen:
        recs = top_map.get(les, [])
        recs = [r for r in recs if pd.notna(r) and r not in seen]
        recs = list(dict.fromkeys(recs))[:per_item_k]

        for rank, rec in enumerate(recs, start=1):
            scores[rec] += 1 / rank

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [item for item, _score in ranked[:top_k]]


def calc_basket_to_basket_metrics(
    test_df: pd.DataFrame,
    top_map: Dict[Any, List[Any]],
    top_k: int = 20,
    input_frac: float = 0.5,
    random_state: int = 42,
    desc: str = "basket-to-basket",
) -> Dict[str, float]:
    """
    Basket-to-basket:
        корзину делим на input и target.
        по input_items строим рекомендации.
        target = скрытая часть корзины.
    """
    rng = random.Random(random_state)

    orders = (
        test_df
        .groupby("order_id")["lesson_id"]
        .apply(lambda x: list(pd.unique(x.dropna())))
        .tolist()
    )

    metrics = []

    for basket in tqdm(orders, desc=desc):
        if len(basket) < 2:
            continue

        basket = list(basket)
        rng.shuffle(basket)

        n_input = max(1, int(len(basket) * input_frac))
        n_input = min(n_input, len(basket) - 1)

        input_items = basket[:n_input]
        target = set(basket[n_input:])

        if len(target) == 0:
            continue

        recs = recommend_for_basket_from_top_map(
            basket=input_items,
            top_map=top_map,
            top_k=top_k,
        )

        m = _calc_ranking_metrics_for_one_query(
            recs=recs,
            target=target,
            top_k=top_k,
        )

        if m is not None:
            metrics.append(m)

    if not metrics:
        return {
            f"basket_prec@{top_k}": np.nan,
            f"basket_recall@{top_k}": np.nan,
            f"basket_hitrate@{top_k}": np.nan,
            f"basket_map@{top_k}": np.nan,
            f"basket_ndcg@{top_k}": np.nan,
            "basket_n_orders": 0,
        }

    return {
        f"basket_prec@{top_k}": float(np.mean([m["precision"] for m in metrics])),
        f"basket_recall@{top_k}": float(np.mean([m["recall"] for m in metrics])),
        f"basket_hitrate@{top_k}": float(np.mean([m["hitrate"] for m in metrics])),
        f"basket_map@{top_k}": float(np.mean([m["map"] for m in metrics])),
        f"basket_ndcg@{top_k}": float(np.mean([m["ndcg"] for m in metrics])),
        "basket_n_orders": len(metrics),
    }


# =========================
# COVERAGE METRICS
# =========================

def calc_catalog_coverage(
    test_df: pd.DataFrame,
    top_map: Dict[Any, List[Any]],
    ks: Tuple[int, ...] = (1, 4, 20),
) -> Dict[str, float]:
    """
    Catalog coverage:
        среди уникальных lesson_id в test_df считаем,
        у какой доли есть >=K рекомендаций.
    """

    lessons = test_df["lesson_id"].dropna().unique()

    rec_lens = []

    for les in lessons:
        recs = top_map.get(les, [])
        recs = [r for r in recs if pd.notna(r)]
        rec_lens.append(len(set(recs)))

    rec_lens = np.array(rec_lens)
    total = len(lessons)

    out = {
        "catalog_lessons": total,
        "catalog_no_recs_rate": float((rec_lens == 0).mean()) if total > 0 else np.nan,
    }

    for k in ks:
        out[f"catalog_ge_{k}_rate"] = float((rec_lens >= k).mean()) if total > 0 else np.nan

    return out


def calc_basket_coverage(
    test_df: pd.DataFrame,
    top_map: Dict[Any, List[Any]],
    ks: Tuple[int, ...] = (1, 4, 20),
) -> Dict[str, float]:
    """
    Basket coverage:
        для каждой корзины объединяем рекомендации всех уроков корзины,
        убираем уже seen-уроки и считаем,
        есть ли >=K уникальных рекомендаций.
    """

    orders = (
        test_df
        .groupby("order_id")["lesson_id"]
        .apply(lambda x: list(pd.unique(x.dropna())))
        .tolist()
    )

    counts = {k: 0 for k in ks}
    max_k = max(ks)

    for basket in tqdm(orders, desc="basket coverage"):
        seen = set(basket)
        basket_recs = set()

        for les in seen:
            recs = top_map.get(les, [])

            for r in recs:
                if pd.notna(r) and r not in seen:
                    basket_recs.add(r)

            if len(basket_recs) >= max_k:
                break

        for k in ks:
            if len(basket_recs) >= k:
                counts[k] += 1

    total = len(orders)

    out = {
        "basket_orders": total,
    }

    for k in ks:
        out[f"basket_ge_{k}_rate"] = counts[k] / total if total > 0 else np.nan

    return out


# =========================
# SAME-COURSE AND STRENGTH
# =========================

def make_lesson_to_course(reference_df: pd.DataFrame) -> Dict[Any, Any]:
    return (
        reference_df[["lesson_id", "course_id"]]
        .drop_duplicates("lesson_id")
        .set_index("lesson_id")["course_id"]
        .to_dict()
    )


def calc_same_course_share(
    recs_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> float:
    """
    Доля рекомендаций, где course_A == course_B.
    """
    if recs_df.empty:
        return np.nan

    lesson_to_course = make_lesson_to_course(reference_df)

    tmp = recs_df.copy()
    tmp["course_A"] = tmp["lesson_A"].map(lesson_to_course)
    tmp["course_B"] = tmp["lesson_B"].map(lesson_to_course)

    valid = tmp["course_A"].notna() & tmp["course_B"].notna()

    if valid.sum() == 0:
        return np.nan

    return float((tmp.loc[valid, "course_A"] == tmp.loc[valid, "course_B"]).mean())


def calc_recs_strength_stats(
    recs_df: pd.DataFrame,
    min_count: int = 10,
) -> Dict[str, float]:
    """
    Статистика силы связи.
    Нужна, чтобы показать, что при расширении top-N можно добавлять более слабые пары.
    """
    if "pair_count" not in recs_df.columns or len(recs_df) == 0:
        return {
            "mean_pair_count": np.nan,
            "median_pair_count": np.nan,
            "share_min_count_pairs": np.nan,
        }

    return {
        "mean_pair_count": float(recs_df["pair_count"].mean()),
        "median_pair_count": float(recs_df["pair_count"].median()),
        "share_min_count_pairs": float((recs_df["pair_count"] <= min_count).mean()),
    }


# =========================
# POPULARITY BUCKET METRICS
# =========================

def get_lesson_popularity_rank(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ранг популярности lesson_id по числу уникальных заказов.
    Считается только на train.
    """
    pop = (
        train_df
        .groupby("lesson_id")["order_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    rank_df = pop.reset_index()
    rank_df.columns = ["lesson_id", "orders_count"]
    rank_df["pop_rank"] = np.arange(1, len(rank_df) + 1)

    return rank_df


def assign_popularity_bucket(rank: int) -> str:
    if rank <= 5000:
        return "1-5000"
    elif rank <= 7000:
        return "5001-7000"
    elif rank <= 10000:
        return "7001-10000"
    elif rank <= 15000:
        return "10001-15000"
    elif rank <= 20000:
        return "15001-20000"
    elif rank <= 30000:
        return "20001-30000"
    else:
        return "30000+"


def calc_item_metrics_by_pop_bucket_order_avg(
    test_df: pd.DataFrame,
    top_map: Dict[Any, List[Any]],
    rank_df: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Та же item-to-basket логика, но результаты группируются
    по popularity bucket input lesson.

    Это нужно, чтобы показать качество не накопительно,
    а отдельно для head/mid/tail сегментов.
    """

    rank_map = rank_df.set_index("lesson_id")["pop_rank"].to_dict()

    orders = (
        test_df
        .groupby("order_id")["lesson_id"]
        .apply(lambda x: list(pd.unique(x.dropna())))
        .tolist()
    )

    bucket_rows = []

    for basket in tqdm(orders, desc="bucket item metrics"):
        if len(basket) < 2:
            continue

        basket_set = set(basket)

        for les in basket:
            target = basket_set - {les}

            if len(target) == 0:
                continue

            rank = rank_map.get(les)

            if rank is None:
                bucket = "unknown"
            else:
                bucket = assign_popularity_bucket(rank)

            recs = top_map.get(les, [])
            recs = [r for r in recs if r != les]

            m = _calc_ranking_metrics_for_one_query(
                recs=recs,
                target=target,
                top_k=top_k,
            )

            if m is not None:
                bucket_rows.append({
                    "bucket": bucket,
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "hitrate": m["hitrate"],
                    "map": m["map"],
                    "ndcg": m["ndcg"],
                })

    raw = pd.DataFrame(bucket_rows)

    if raw.empty:
        return pd.DataFrame()

    out = (
        raw
        .groupby("bucket", as_index=False)
        .agg(
            **{
                f"bucket_prec@{top_k}": ("precision", "mean"),
                f"bucket_recall@{top_k}": ("recall", "mean"),
                f"bucket_hitrate@{top_k}": ("hitrate", "mean"),
                f"bucket_map@{top_k}": ("map", "mean"),
                f"bucket_ndcg@{top_k}": ("ndcg", "mean"),
                "bucket_n_queries": ("precision", "size"),
            }
        )
    )

    bucket_order = [
        "1-5000",
        "5001-7000",
        "7001-10000",
        "10001-15000",
        "15001-20000",
        "20001-30000",
        "30000+",
        "unknown",
    ]

    out["bucket"] = pd.Categorical(out["bucket"], categories=bucket_order, ordered=True)
    out = out.sort_values("bucket").reset_index(drop=True)

    return out


# =========================
# MODEL ADAPTER
# =========================

def fit_model_and_get_recs_df(
    model_class: Any,
    train_df: pd.DataFrame,
    top_k: int = 20,
    min_count: int = 10,
    score_mode: str = "custom_index",
) -> Tuple[Any, pd.DataFrame]:
    """
    Адаптер под твою модель.

    Ожидаемый конструктор:
        model_class(top_n=..., min_count=..., score_mode=...)

    Если твой AssociationFilteringBaseline принимает другие параметры,
    поправь только эту функцию.
    """

    model = model_class(
        top_n=top_k,
        min_count=min_count,
        score_mode=score_mode,
    )

    model.fit(train_df)

    if hasattr(model, "recs_df"):
        recs_df = model.recs_df.copy()

    elif hasattr(model, "get_recommendations_table"):
        recs_df = model.get_recommendations_table(only_score=False).copy()

    else:
        raise AttributeError(
            "Не нашел у модели recs_df или get_recommendations_table(). "
            "Поправь fit_model_and_get_recs_df под свой класс."
        )

    required_cols = {"lesson_A", "lesson_B", "score"}
    miss = required_cols - set(recs_df.columns)

    if miss:
        raise ValueError(f"В recs_df нет колонок: {miss}")

    return model, recs_df


# =========================
# MAIN EXPERIMENT LOOP
# =========================

def run_topn_quality_curves(
    all_data: pd.DataFrame,
    model_class: Any,
    top_ns: List[int] = TOP_NS,
    train_modes: List[str] = TRAIN_MODES,
    top_k: int = 20,
    min_count: int = 10,
    score_mode: str = "custom_index",
    test_size: float = 0.1,
    random_state: int = 42,
    strict_topn_training: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Основной цикл экспериментов.

    Возвращает:
        results_df:
            метрики по каждой комбинации train_mode × top_n

        bucket_results_df:
            метрики по popularity bucket
    """

    required_data_cols = {"order_id", "lesson_id", "course_id"}
    miss = required_data_cols - set(all_data.columns)
    if miss:
        raise ValueError(f"В all_data нет колонок: {miss}")

    train_all, test_all = split_by_order(
        all_data,
        test_size=test_size,
        random_state=random_state,
    )

    # Offline quality считаем только на тестовых заказах с 2+ уроками.
    test_quality = filter_orders_by_mode(test_all, "2plus_lessons")

    results = []
    bucket_results = []

    for train_mode in train_modes:
        print("\n==============================")
        print(f"TRAIN MODE: {train_mode}")
        print("==============================")

        train_base = filter_orders_by_mode(train_all, train_mode)
        rank_df = get_lesson_popularity_rank(train_base)

        for top_n in top_ns:
            print(f"\n--- Top-N = {top_n} ---")

            top_lessons = get_top_popular_lessons(train_base, top_n)

            if strict_topn_training:
                # Жесткий режим:
                # обучаемся только на top-N популярных уроках.
                train_exp = restrict_to_top_lessons(train_base, top_lessons)

                # После top-N фильтрации снова применяем тот же train_mode.
                # Это важно, потому что часть заказов могла стать 1-item.
                train_exp = filter_orders_by_mode(train_exp, train_mode)
            else:
                # Мягкий режим:
                # обучаемся на всей train_base,
                # а итоговую матрицу рекомендаций потом режем до top-N.
                train_exp = train_base.copy()

            if train_exp["order_id"].nunique() == 0:
                print("Нет заказов после фильтрации. Пропускаю.")
                continue

            model, recs_df = fit_model_and_get_recs_df(
                model_class=model_class,
                train_df=train_exp,
                top_k=top_k,
                min_count=min_count,
                score_mode=score_mode,
            )

            if not strict_topn_training:
                # Если обучались на всех, режем output до top-N.
                recs_df = filter_recs_to_top_lessons(
                    recs_df,
                    top_lessons=top_lessons,
                    filter_A=True,
                    filter_B=True,
                )

            top_map = make_top_map(recs_df, top_k=top_k)

            row = {
                "train_mode": train_mode,
                "top_popular_n": top_n,
                "strict_topn_training": strict_topn_training,
                "min_count": min_count,
                "top_k": top_k,
                "train_orders": train_exp["order_id"].nunique(),
                "train_lessons": train_exp["lesson_id"].nunique(),
                "test_orders": test_all["order_id"].nunique(),
                "test_quality_orders": test_quality["order_id"].nunique(),
                "recs_rows": len(recs_df),
            }

            # Coverage
            row.update(calc_catalog_coverage(test_all, top_map, ks=(1, 4, 20)))
            row.update(calc_basket_coverage(test_all, top_map, ks=(1, 4, 20)))

            # Offline quality: твоя item-to-basket логика
            row.update(calc_item_to_basket_metrics_order_avg(
                test_quality,
                top_map,
                top_k=top_k,
                desc=f"item-to-basket {train_mode} top={top_n}",
            ))

            # Offline quality: basket-to-basket
            row.update(calc_basket_to_basket_metrics(
                test_quality,
                top_map,
                top_k=top_k,
                input_frac=0.5,
                random_state=random_state,
                desc=f"basket-to-basket {train_mode} top={top_n}",
            ))

            # Same-course + strength
            row["same_course_share"] = calc_same_course_share(recs_df, train_exp)
            row.update(calc_recs_strength_stats(recs_df, min_count=min_count))

            results.append(row)

            # Bucket metrics
            bucket_df = calc_item_metrics_by_pop_bucket_order_avg(
                test_quality,
                top_map,
                rank_df=rank_df,
                top_k=top_k,
            )

            if not bucket_df.empty:
                bucket_df["train_mode"] = train_mode
                bucket_df["top_popular_n"] = top_n
                bucket_df["strict_topn_training"] = strict_topn_training
                bucket_df["min_count"] = min_count
                bucket_results.append(bucket_df)

    results_df = pd.DataFrame(results)

    if bucket_results:
        bucket_results_df = pd.concat(bucket_results, ignore_index=True)
    else:
        bucket_results_df = pd.DataFrame()

    return results_df, bucket_results_df


# =========================
# PLOTS
# =========================

def plot_topn_metric(
    results_df: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
) -> None:
    """
    Cumulative Top-N curve.
    """
    plt.figure(figsize=(10, 5))

    for train_mode in results_df["train_mode"].unique():
        part = (
            results_df[results_df["train_mode"] == train_mode]
            .sort_values("top_popular_n")
        )

        plt.plot(
            part["top_popular_n"],
            part[metric],
            marker="o",
            label=train_mode,
        )

    plt.xlabel("Top-N популярных уроков")
    plt.ylabel(metric)
    plt.title(title or f"{metric} vs Top-N")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_bucket_metric(
    bucket_results_df: pd.DataFrame,
    metric: str,
    top_n: Optional[int] = None,
    train_mode: Optional[str] = None,
) -> None:
    """
    Quality by popularity bucket.
    """
    df = bucket_results_df.copy()

    if top_n is not None:
        df = df[df["top_popular_n"] == top_n]

    if train_mode is not None:
        df = df[df["train_mode"] == train_mode]

    if df.empty:
        print("Нет данных для графика")
        return

    bucket_order = [
        "1-5000",
        "5001-7000",
        "7001-10000",
        "10001-15000",
        "15001-20000",
        "20001-30000",
        "30000+",
        "unknown",
    ]

    df["bucket"] = pd.Categorical(df["bucket"], categories=bucket_order, ordered=True)

    plt.figure(figsize=(11, 5))

    for mode in df["train_mode"].unique():
        part = df[df["train_mode"] == mode].sort_values("bucket")

        plt.plot(
            part["bucket"].astype(str),
            part[metric],
            marker="o",
            label=mode,
        )

    title_parts = [metric]
    if top_n is not None:
        title_parts.append(f"Top-{top_n}")
    if train_mode is not None:
        title_parts.append(train_mode)

    plt.title(" | ".join(title_parts))
    plt.xlabel("Popularity bucket input lesson")
    plt.ylabel(metric)
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.legend()
    plt.show()


def make_summary_table(results_df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """
    Компактная итоговая таблица для руководителя.
    """
    candidate_cols = [
        "train_mode",
        "top_popular_n",
        "train_orders",
        "train_lessons",
        "recs_rows",

        "catalog_no_recs_rate",
        "catalog_ge_20_rate",
        "basket_ge_20_rate",

        f"item_prec@{top_k}",
        f"item_recall@{top_k}",
        f"item_hitrate@{top_k}",
        f"item_map@{top_k}",
        f"item_ndcg@{top_k}",

        f"basket_prec@{top_k}",
        f"basket_recall@{top_k}",
        f"basket_hitrate@{top_k}",
        f"basket_map@{top_k}",
        f"basket_ndcg@{top_k}",

        "same_course_share",
        "median_pair_count",
        "share_min_count_pairs",
    ]

    existing_cols = [c for c in candidate_cols if c in results_df.columns]

    summary = (
        results_df[existing_cols]
        .sort_values(["train_mode", "top_popular_n"])
        .reset_index(drop=True)
    )

    return summary


def plot_default_curves(
    results_df: pd.DataFrame,
    bucket_results_df: Optional[pd.DataFrame] = None,
    top_k: int = 20,
    bucket_top_n: int = 30000,
) -> None:
    """
    Рисует основной набор графиков.
    """

    main_metrics = [
        (f"item_recall@{top_k}", f"Item-to-basket Recall@{top_k}"),
        (f"item_hitrate@{top_k}", f"Item-to-basket Hitrate@{top_k}"),
        (f"item_ndcg@{top_k}", f"Item-to-basket NDCG@{top_k}"),

        (f"basket_recall@{top_k}", f"Basket-to-basket Recall@{top_k}"),
        (f"basket_hitrate@{top_k}", f"Basket-to-basket Hitrate@{top_k}"),
        (f"basket_ndcg@{top_k}", f"Basket-to-basket NDCG@{top_k}"),

        ("catalog_ge_20_rate", "Catalog coverage: lessons with >=20 recs"),
        ("basket_ge_20_rate", "Basket coverage: baskets with >=20 recs"),
        ("same_course_share", "Same-course share"),
        ("median_pair_count", "Median pair_count"),
        ("share_min_count_pairs", "Share of pairs near min_count"),
    ]

    for metric, title in main_metrics:
        if metric in results_df.columns:
            plot_topn_metric(results_df, metric, title)

    if bucket_results_df is not None and not bucket_results_df.empty:
        bucket_metrics = [
            f"bucket_recall@{top_k}",
            f"bucket_hitrate@{top_k}",
            f"bucket_ndcg@{top_k}",
        ]

        for metric in bucket_metrics:
            if metric in bucket_results_df.columns:
                plot_bucket_metric(bucket_results_df, metric, top_n=bucket_top_n)


# =========================
# EXAMPLE USAGE
# =========================
#
# from models import AssociationFilteringBaseline
#
# results_df, bucket_results_df = run_topn_quality_curves(
#     all_data=all_data,
#     model_class=AssociationFilteringBaseline,
#     top_ns=[5000, 7000, 10000, 15000, 20000, 30000],
#     train_modes=["2plus_courses", "2plus_lessons"],
#     top_k=20,
#     min_count=10,
#     score_mode="custom_index",
#     test_size=0.1,
#     random_state=42,
#     strict_topn_training=True,
# )
#
# results_df.to_csv("topn_quality_curves_results.csv", index=False)
# bucket_results_df.to_csv("topn_bucket_quality_results.csv", index=False)
#
# summary = make_summary_table(results_df, top_k=20)
# summary.to_csv("topn_quality_summary.csv", index=False)
#
# plot_default_curves(results_df, bucket_results_df, top_k=20, bucket_top_n=30000)
#
