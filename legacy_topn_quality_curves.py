"""
legacy_topn_quality_curves.py

Этот файл специально адаптирован под твою текущую логику из notebooks и metrics.py,
чтобы цифры совпадали с тем, как ты уже считал.

Что здесь важно:
1. Метрики calc_metric / calc_basket_to_basket сохранены в legacy-виде.
2. calc_metric усредняет по каждому input-уроку, как в твоем metrics.py.
3. calc_basket_to_basket использует random.Random(seed).shuffle(basket) и мутирует basket,
   как в твоем metrics.py.
4. split делается через random.shuffle(unique_orders), как в notebooks.
5. top-N уроки выбираются через train_data["lesson_id"].value_counts()[:N].index,
   как в notebooks.
6. Для Top-30000 можно включить legacy режим:
   Top-30000 = full train_data, как в calc_metrics_crs.ipynb / calc_metrics_les.ipynb.

Ожидаемые колонки:
    order_id, lesson_id, course_id

Ожидаемый интерфейс модели:
    model = ArRecommender(top_k=20, min_count=10, min_support=0.0)
    model.fit(train_df)
    recs_df = model.get_recommendations()

Ожидаемые колонки recs_df:
    lesson_A, lesson_B
Дополнительно, если есть:
    score, pair_count, rank
"""

import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


# =============================================================================
# 1. LEGACY METRICS: copied/adapted from your metrics.py
# =============================================================================

def calc_metric(orders, top_map, top_k=3):
    """
    Твоя исходная item-to-basket метрика.

    Для каждой корзины:
        для каждого les в basket:
            true_les = set(basket) - {les}
            recs = top_map.get(les)[:top_k]
            считаем prec / recall / hitrate / AP

    Важно:
    - усреднение идет по всем input-урокам, а не сначала по корзине;
    - если recs_df is None, добавляются нули;
    - recs_df[:top_k] берется как есть;
    - дубли рекомендаций не удаляются;
    - seen-уроки из рекомендаций не фильтруются, кроме самого эффекта target.
    """
    prec = []
    recall = []
    hitrate = []
    ap = []

    n_orders = len(orders)
    total = sum(len(basket) for basket in orders)

    with tqdm(total=total) as pbar:
        for i, basket in enumerate(orders):
            for _, les in enumerate(basket):
                true_les = set(basket) - {les}
                n_true_les = len(true_les)

                if n_true_les == 0:
                    pbar.update(1)
                    continue

                recs_df = top_map.get(les)

                if recs_df is None:
                    prec.append(0)
                    recall.append(0)
                    hitrate.append(0)
                    ap.append(0)
                    pbar.update(1)
                    continue

                recs = recs_df[:top_k]

                hits = 0
                sum_prec = 0.0

                for idx, rec in enumerate(recs):
                    if rec in true_les:
                        hits += 1
                        sum_prec += hits / (idx + 1)

                prec.append(hits / top_k)
                recall.append(hits / n_true_les)
                hitrate.append(int(hits > 0))
                ap.append(sum_prec / min(n_true_les, top_k))

                pbar.update(1)
                pbar.set_description(f"Orders {i+1}/{n_orders}")

    return {
        f"prec@{top_k}": np.mean(prec),
        f"recall@{top_k}": np.mean(recall),
        f"hitrate@{top_k}": np.mean(hitrate),
        f"map@{top_k}": np.mean(ap),
    }


def calc_metrics_simple(df_basket, top_map, top_k=10, n_input=3):
    """
    Твоя исходная next-item/simple метрика из metrics.py.
    Оставлена без изменений по логике.
    """
    hitrate = []
    maps = []

    for basket in tqdm(df_basket):
        basket = list(set(basket))
        if len(basket) <= n_input:
            continue

        target_item = random.choice(basket)
        input_items = [i for i in basket if i != target_item]

        if len(input_items) > n_input:
            input_items = random.sample(input_items, n_input)

        combined_recs = []
        for item in input_items:
            if item in top_map:
                combined_recs.extend(top_map[item])

        seen = set(input_items)
        final_recs = []
        for r in combined_recs:
            if r not in seen:
                final_recs.append(r)
                seen.add(r)

        final_recs = final_recs[:top_k]

        hit = 1 if target_item in final_recs else 0
        ap = 0
        if hit:
            rank = final_recs.index(target_item) + 1
            ap = 1 / rank

        hitrate.append(hit)
        maps.append(ap)

    return {
        f"hitrate@{top_k}": np.mean(hitrate),
        f"precision@{top_k}": np.mean(hitrate) / top_k,
        f"map@{top_k}": np.mean(maps),
    }


def calc_basket_to_basket(orders, top_map, top_k=20, input_frac=0.5, min_basket_size=2, seed=42):
    """
    Твоя исходная basket-to-basket метрика.

    Важно:
    - rnd.shuffle(basket) мутирует список basket in-place;
    - дубли в fin_recs НЕ удаляются, если один и тот же les пришел повторно после seen;
    - seen обновляется только input_basket, а не рекомендациями;
    - target_basket — list, не set.
    """
    rnd = random.Random(seed)

    prec_list = []
    recall_list = []
    hitrate_list = []

    for basket in tqdm(orders):

        if len(basket) < min_basket_size:
            continue

        rnd.shuffle(basket)

        n_input = max(1, int(len(basket) * input_frac))
        n_input = min(n_input, len(basket) - 1)

        input_basket = basket[:n_input]
        target_basket = basket[n_input:]

        if len(target_basket) == 0:
            continue

        recs = []
        fin_recs = []
        seen = set(input_basket)

        for l in input_basket:
            l_r = top_map.get(l, [])
            recs.extend(l_r)

        for les in recs:
            if les not in seen:
                fin_recs.append(les)

        fin_recs = fin_recs[:top_k]
        hits = [1 if r in target_basket else 0 for r in fin_recs]
        n_hits = sum(hits)

        prec = n_hits / top_k
        recall = n_hits / len(target_basket)
        hitrate = int(n_hits > 0)

        prec_list.append(prec)
        recall_list.append(recall)
        hitrate_list.append(hitrate)

    return {
        f"prec@{top_k}": np.mean(prec_list),
        f"recall@{top_k}": np.mean(recall_list),
        f"hitrate@{top_k}": np.mean(hitrate_list),
    }


# =============================================================================
# 2. LEGACY COVERAGE FROM YOUR NOTEBOOKS
# =============================================================================

def coverage_lesson_legacy(recs, df):
    """
    Твоя coverage_lesson, но возвращает dict вместо print.
    Логика сохранена:
        recs.get(int(les), [])
        len(recs_s)
    """
    recs_len_lesson = []
    test_les = df["lesson_id"].unique()

    for les in test_les:
        recs_s = recs.get(int(les), [])
        recs_len_lesson.append(len(recs_s))

    arr = np.array(recs_len_lesson)

    no_recs_lesson = np.where(arr == 0, 1, 0).sum()
    no_less_k_lesson = np.where(arr >= 20, 1, 0).sum()
    no_less_4_lesson = np.where(arr >= 4, 1, 0).sum()
    no_less_1_lesson = np.where(arr >= 1, 1, 0).sum()

    return {
        "lesson_total": len(test_les),
        "lesson_no_recs_count": int(no_recs_lesson),
        "lesson_ge_20_count": int(no_less_k_lesson),
        "lesson_ge_4_count": int(no_less_4_lesson),
        "lesson_ge_1_count": int(no_less_1_lesson),
        "lesson_no_recs_rate": no_recs_lesson / len(test_les),
        "lesson_ge_20_rate": no_less_k_lesson / len(test_les),
        "lesson_ge_4_rate": no_less_4_lesson / len(test_les),
        "lesson_ge_1_rate": no_less_1_lesson / len(test_les),
    }


def print_coverage_lesson_legacy(recs, df):
    out = coverage_lesson_legacy(recs, df)
    total = out["lesson_total"]

    print(
        f"Доля уроков без рекомендаций: "
        f"{out['lesson_no_recs_rate']:.2%} ({out['lesson_no_recs_count']}/{total})"
    )
    print(
        f"Доля уроков с рекомендациями >= 20: "
        f"{out['lesson_ge_20_rate']:.2%} ({out['lesson_ge_20_count']}/{total})"
    )
    print(
        f"Доля уроков с рекомендациями >= 4: "
        f"{out['lesson_ge_4_rate']:.2%} ({out['lesson_ge_4_count']}/{total})"
    )
    print(
        f"Доля уроков с рекомендациями >= 1: "
        f"{out['lesson_ge_1_rate']:.2%} ({out['lesson_ge_1_count']}/{total})"
    )


def coverage_by_K_legacy(recs, df, ks=(1, 2, 3, 4, 10, 20)):
    """
    Твоя coverage_by_K, но возвращает dict вместо print.

    Логика сохранена:
    - basket_rec = list, не set;
    - дубли рекомендаций считаются;
    - recs.get(les, []);
    - уроки из seen не добавляем.
    """
    df_basket = df.groupby("order_id")["lesson_id"].apply(list)

    out = {
        "basket_total": len(df_basket),
    }

    for i in ks:
        count_orders_least_one_no_recs = 0

        for basket in df_basket:
            seen = set(basket)
            basket_rec = []

            for les in basket:
                les_rec = recs.get(les, [])

                for les_r in les_rec:
                    if les_r not in seen:
                        basket_rec.append(les_r)

            if len(basket_rec) >= i:
                count_orders_least_one_no_recs += 1

        coverage_orders_soft = count_orders_least_one_no_recs / len(df_basket)

        out[f"basket_ge_{i}_count"] = int(count_orders_least_one_no_recs)
        out[f"basket_ge_{i}_rate"] = coverage_orders_soft

    return out


def print_coverage_by_K_legacy(recs, df, ks=(1, 2, 3, 4, 10, 20)):
    out = coverage_by_K_legacy(recs, df, ks=ks)
    total = out["basket_total"]

    for i in ks:
        print(
            f"Доля где в корзине >= {i} рекомендаций: "
            f"{out[f'basket_ge_{i}_rate']:.2%} "
            f"({out[f'basket_ge_{i}_count']}/{total})"
        )


def calculate_greedy_coverage_legacy(df):
    """
    Твой calculate_greedy_coverage из notebook.
    """
    lesson_to_orders = df.groupby("lesson_id")["order_id"].apply(set).to_dict()

    sorted_lessons = sorted(
        lesson_to_orders.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    total_orders_count = df["order_id"].nunique()
    covered_orders = set()
    metrics = []

    for i, (lesson_id, orders) in enumerate(sorted_lessons, 1):
        covered_orders.update(orders)
        coverage = len(covered_orders) / total_orders_count

        if i in [1, 100, 1000, 5000, 7000, 10000, 15000, 20000, 30000]:
            metrics.append((i, coverage))

        if coverage >= 0.99:
            metrics.append((i, coverage))
            break

    return metrics


# =============================================================================
# 3. MODEL ADAPTER
# =============================================================================

def init_model_legacy(model_class, top_k=20, min_count=10, min_support=0.0, model_kwargs=None):
    """
    По умолчанию повторяет:
        ArRecommender(top_k=20, min_count=10, min_support=0.0)

    Если у твоей модели другие параметры, передай model_kwargs.
    """
    model_kwargs = dict(model_kwargs or {})

    try:
        return model_class(
            top_k=top_k,
            min_count=min_count,
            min_support=min_support,
            **model_kwargs,
        )
    except TypeError:
        # fallback для классов, где параметр называется top_n
        return model_class(
            top_n=top_k,
            min_count=min_count,
            min_support=min_support,
            **model_kwargs,
        )


def get_recommendations_legacy(model):
    """
    Повторяет твой notebook:
        ar_recs = ar_rec.get_recommendations()

    Если у модели другой интерфейс, fallback на recs_df.
    """
    if hasattr(model, "get_recommendations"):
        return model.get_recommendations()

    if hasattr(model, "recs_df"):
        return model.recs_df.copy()

    if hasattr(model, "get_recommendations_table"):
        return model.get_recommendations_table(only_score=False).copy()

    raise AttributeError(
        "Не нашел get_recommendations(), recs_df или get_recommendations_table()."
    )


def recs_df_to_top_map_legacy(recs_df):
    """
    Повторяет:
        ar_recs.groupby('lesson_A')['lesson_B'].apply(list).to_dict()
    """
    if recs_df.empty:
        return {}

    return recs_df.groupby("lesson_A")["lesson_B"].apply(list).to_dict()


# =============================================================================
# 4. TRAIN/TEST PREPARATION EXACTLY LIKE NOTEBOOKS
# =============================================================================

def make_order_split_legacy(all_data, train_frac=0.9, seed=None):
    """
    Повторяет notebook:

        unique_orders = all_data["order_id"].unique()
        random.shuffle(unique_orders)
        train_size = int(len(unique_orders) * 0.9)
        train_orders = unique_orders[:train_size]
        test_orders = unique_orders[train_size:]

    Если seed=None, random не фиксируется, как в твоем notebook.
    Если seed=42, split становится воспроизводимым.
    """
    if seed is not None:
        random.seed(seed)

    unique_orders = all_data["order_id"].unique()
    random.shuffle(unique_orders)

    train_size = int(len(unique_orders) * train_frac)

    train_orders = unique_orders[:train_size]
    test_orders = unique_orders[train_size:]

    return train_orders, test_orders


def make_train_test_for_source_legacy(source_data, all_data, train_frac=0.9, seed=None):
    train_orders, test_orders = make_order_split_legacy(
        all_data=all_data,
        train_frac=train_frac,
        seed=seed,
    )

    train_data = source_data[source_data["order_id"].isin(train_orders)]
    test_data = source_data[source_data["order_id"].isin(test_orders)]

    return train_data, test_data, train_orders, test_orders


def make_topn_train_data_legacy(
    train_data,
    top_n,
    legacy_30000_full_train=True,
):
    """
    Повторяет notebooks:

    top5000_les = train_data["lesson_id"].value_counts()[:5000].index
    data5000 = train_data[train_data["lesson_id"].isin(top5000_les)]

    В calc_metrics_crs/les.ipynb Top-30000 обучался на full train_data:
        ar_rec_30000.fit(train_data)

    Поэтому по умолчанию:
        top_n == 30000 -> train_data
    """
    if legacy_30000_full_train and top_n == 30000:
        return train_data

    top_lessons = train_data["lesson_id"].value_counts()[:top_n].index
    return train_data[train_data["lesson_id"].isin(top_lessons)]


# =============================================================================
# 5. MAIN LEGACY EXPERIMENT
# =============================================================================

def run_one_train_source_legacy(
    source_data,
    all_data,
    model_class,
    source_name,
    top_ns=(5000, 7000, 10000, 15000, 20000, 30000),
    top_k=20,
    min_count=10,
    min_support=0.0,
    train_frac=0.9,
    seed=None,
    legacy_30000_full_train=True,
    model_kwargs=None,
    calc_next_item=False,
    next_item_n_input=10,
):
    """
    Полностью повторяет схему одного notebook:
        calc_metrics_crs.ipynb или calc_metrics_les.ipynb.

    source_data:
        data_crs_more2 или data_les_more2

    all_data:
        all_data для split по order_id

    source_name:
        например "2plus_courses" или "2plus_lessons"

    seed:
        None -> как в notebooks без фиксации random.seed
        42 -> воспроизводимый split

    ВАЖНО:
    - сначала обучаем все модели и строим top_map;
    - потом считаем calc_metric для всех top-N;
    - потом calc_metrics_simple, если нужно;
    - потом calc_basket_to_basket для всех top-N.
      Это повторяет порядок notebooks и сохраняет эффект in-place shuffle
      внутри calc_basket_to_basket.
    """

    train_data, test_data, train_orders, test_orders = make_train_test_for_source_legacy(
        source_data=source_data,
        all_data=all_data,
        train_frac=train_frac,
        seed=seed,
    )

    print(f"\n==============================")
    print(f"SOURCE: {source_name}")
    print(f"train_data: {train_data.shape}")
    print(f"test_data:  {test_data.shape}")
    print(f"==============================")

    # как в notebook:
    test_baskets = test_data.groupby("order_id")["lesson_id"].apply(list)

    recs_by_topn = {}
    top_map_by_topn = {}
    train_shape_by_topn = {}

    # 1. Fit all models
    for top_n in top_ns:
        print(f"\n--- FIT Top-{top_n} ---")

        train_topn = make_topn_train_data_legacy(
            train_data=train_data,
            top_n=top_n,
            legacy_30000_full_train=legacy_30000_full_train,
        )

        model = init_model_legacy(
            model_class=model_class,
            top_k=top_k,
            min_count=min_count,
            min_support=min_support,
            model_kwargs=model_kwargs,
        )

        model.fit(train_topn)

        recs_df = get_recommendations_legacy(model)
        top_map = recs_df_to_top_map_legacy(recs_df)

        recs_by_topn[top_n] = recs_df
        top_map_by_topn[top_n] = top_map
        train_shape_by_topn[top_n] = {
            "train_rows": len(train_topn),
            "train_orders": train_topn["order_id"].nunique(),
            "train_lessons": train_topn["lesson_id"].nunique(),
            "recs_rows": len(recs_df),
        }

    # 2. calc_metric for all top-N
    metric_rows = []
    metric_table = {}

    for top_n in top_ns:
        print(f"\n--- calc_metric Top-{top_n} ---")
        m = calc_metric(test_baskets, top_map_by_topn[top_n], top_k=top_k)
        metric_table[f"Top-{top_n}"] = m

        row = {
            "source_name": source_name,
            "metric_type": "item_to_basket_legacy",
            "top_n": top_n,
            **train_shape_by_topn[top_n],
            **m,
        }
        metric_rows.append(row)

    item_metrics_df = pd.DataFrame(metric_rows)
    item_metrics_wide = pd.DataFrame(metric_table)

    # 3. optional next-item
    next_item_metrics_df = None
    next_item_metrics_wide = None

    if calc_next_item:
        next_rows = []
        next_table = {}

        for top_n in top_ns:
            print(f"\n--- calc_metrics_simple Top-{top_n} ---")
            m = calc_metrics_simple(
                test_baskets,
                top_map_by_topn[top_n],
                top_k=top_k,
                n_input=next_item_n_input,
            )
            next_table[f"Top-{top_n}"] = m

            row = {
                "source_name": source_name,
                "metric_type": "next_item_legacy",
                "top_n": top_n,
                **train_shape_by_topn[top_n],
                **m,
            }
            next_rows.append(row)

        next_item_metrics_df = pd.DataFrame(next_rows)
        next_item_metrics_wide = pd.DataFrame(next_table)

    # 4. basket-to-basket for all top-N
    # ВАЖНО: это использует тот же test_baskets объект и мутирует baskets in-place,
    # как в notebook.
    basket_rows = []
    basket_table = {}

    for top_n in top_ns:
        print(f"\n--- calc_basket_to_basket Top-{top_n} ---")
        m = calc_basket_to_basket(
            test_baskets,
            top_map_by_topn[top_n],
            top_k=top_k,
            input_frac=0.5,
            min_basket_size=2,
            seed=42,
        )
        basket_table[f"Top-{top_n}"] = m

        row = {
            "source_name": source_name,
            "metric_type": "basket_to_basket_legacy",
            "top_n": top_n,
            **train_shape_by_topn[top_n],
            **m,
        }
        basket_rows.append(row)

    basket_metrics_df = pd.DataFrame(basket_rows)
    basket_metrics_wide = pd.DataFrame(basket_table)

    return {
        "source_name": source_name,
        "train_data": train_data,
        "test_data": test_data,
        "train_orders": train_orders,
        "test_orders": test_orders,
        "test_baskets": test_baskets,
        "recs_by_topn": recs_by_topn,
        "top_map_by_topn": top_map_by_topn,
        "item_metrics_df": item_metrics_df,
        "item_metrics_wide": item_metrics_wide,
        "basket_metrics_df": basket_metrics_df,
        "basket_metrics_wide": basket_metrics_wide,
        "next_item_metrics_df": next_item_metrics_df,
        "next_item_metrics_wide": next_item_metrics_wide,
    }


def run_crs_and_les_legacy(
    all_data,
    data_crs_more2,
    data_les_more2,
    model_class,
    top_ns=(5000, 7000, 10000, 15000, 20000, 30000),
    top_k=20,
    min_count=10,
    min_support=0.0,
    train_frac=0.9,
    seed=None,
    legacy_30000_full_train=True,
    model_kwargs=None,
    calc_next_item=False,
):
    """
    Запускает оба сценария:
        1. обучение на 2+ курсах
        2. обучение на 2+ уроках

    Если хочешь максимально повторить notebooks, можешь запускать отдельно:
        run_one_train_source_legacy(data_crs_more2, ...)
        run_one_train_source_legacy(data_les_more2, ...)

    Если seed=42, split будет одинаково воспроизводимым.
    Если seed=None, split будет как random.shuffle без seed.
    """

    crs_res = run_one_train_source_legacy(
        source_data=data_crs_more2,
        all_data=all_data,
        model_class=model_class,
        source_name="2plus_courses",
        top_ns=top_ns,
        top_k=top_k,
        min_count=min_count,
        min_support=min_support,
        train_frac=train_frac,
        seed=seed,
        legacy_30000_full_train=legacy_30000_full_train,
        model_kwargs=model_kwargs,
        calc_next_item=calc_next_item,
    )

    # Если seed задан, les получит такой же split.
    # Если seed=None, split будет другим, как при отдельных notebook-запусках без seed.
    les_res = run_one_train_source_legacy(
        source_data=data_les_more2,
        all_data=all_data,
        model_class=model_class,
        source_name="2plus_lessons",
        top_ns=top_ns,
        top_k=top_k,
        min_count=min_count,
        min_support=min_support,
        train_frac=train_frac,
        seed=seed,
        legacy_30000_full_train=legacy_30000_full_train,
        model_kwargs=model_kwargs,
        calc_next_item=calc_next_item,
    )

    item_metrics_all = pd.concat(
        [crs_res["item_metrics_df"], les_res["item_metrics_df"]],
        ignore_index=True,
    )

    basket_metrics_all = pd.concat(
        [crs_res["basket_metrics_df"], les_res["basket_metrics_df"]],
        ignore_index=True,
    )

    return {
        "crs": crs_res,
        "les": les_res,
        "item_metrics_all": item_metrics_all,
        "basket_metrics_all": basket_metrics_all,
    }


# =============================================================================
# 6. ADDITIONAL CURVES: buckets and strength, adapted to legacy outputs
# =============================================================================

def get_popularity_rank_from_value_counts(train_data):
    """
    Ранг популярности как в твоем выборе top-N:
        train_data["lesson_id"].value_counts()
    """
    vc = train_data["lesson_id"].value_counts()
    rank_df = vc.reset_index()
    rank_df.columns = ["lesson_id", "row_count"]
    rank_df["pop_rank"] = np.arange(1, len(rank_df) + 1)
    return rank_df


def assign_popularity_bucket(rank):
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


def calc_metric_by_input_bucket_legacy(orders, top_map, rank_df, top_k=20):
    """
    Версия calc_metric по bucket-ам input lesson.
    Логика calc_metric сохранена:
        recs_df[:top_k]
        дубли не чистим
        seen не фильтруем
        missing recs -> нули
    """
    rank_map = rank_df.set_index("lesson_id")["pop_rank"].to_dict()

    rows = []

    n_orders = len(orders)
    total = sum(len(basket) for basket in orders)

    with tqdm(total=total) as pbar:
        for i, basket in enumerate(orders):
            for _, les in enumerate(basket):
                true_les = set(basket) - {les}
                n_true_les = len(true_les)

                if n_true_les == 0:
                    pbar.update(1)
                    continue

                rank = rank_map.get(les)
                bucket = "unknown" if rank is None else assign_popularity_bucket(rank)

                recs_df = top_map.get(les)

                if recs_df is None:
                    rows.append({
                        "bucket": bucket,
                        f"prec@{top_k}": 0.0,
                        f"recall@{top_k}": 0.0,
                        f"hitrate@{top_k}": 0.0,
                        f"map@{top_k}": 0.0,
                    })
                    pbar.update(1)
                    continue

                recs = recs_df[:top_k]

                hits = 0
                sum_prec = 0.0

                for idx, rec in enumerate(recs):
                    if rec in true_les:
                        hits += 1
                        sum_prec += hits / (idx + 1)

                rows.append({
                    "bucket": bucket,
                    f"prec@{top_k}": hits / top_k,
                    f"recall@{top_k}": hits / n_true_les,
                    f"hitrate@{top_k}": int(hits > 0),
                    f"map@{top_k}": sum_prec / min(n_true_les, top_k),
                })

                pbar.update(1)
                pbar.set_description(f"Orders {i+1}/{n_orders}")

    raw = pd.DataFrame(rows)

    if raw.empty:
        return raw

    out = (
        raw
        .groupby("bucket", as_index=False)
        .agg(
            **{
                f"bucket_prec@{top_k}": (f"prec@{top_k}", "mean"),
                f"bucket_recall@{top_k}": (f"recall@{top_k}", "mean"),
                f"bucket_hitrate@{top_k}": (f"hitrate@{top_k}", "mean"),
                f"bucket_map@{top_k}": (f"map@{top_k}", "mean"),
                "n_queries": (f"prec@{top_k}", "size"),
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


def calc_pair_strength_legacy(recs_df, min_count=10):
    """
    Статистика силы пар, если в recs_df есть pair_count или score.

    В твоем get_recommendations score может быть кастомным score,
    поэтому pair_count надежнее. Если pair_count нет, используем score.
    """
    if recs_df is None or len(recs_df) == 0:
        return {
            "mean_pair_count_or_score": np.nan,
            "median_pair_count_or_score": np.nan,
            "share_min_count_pairs": np.nan,
        }

    if "pair_count" in recs_df.columns:
        s = recs_df["pair_count"]
    elif "score" in recs_df.columns:
        s = recs_df["score"]
    else:
        return {
            "mean_pair_count_or_score": np.nan,
            "median_pair_count_or_score": np.nan,
            "share_min_count_pairs": np.nan,
        }

    return {
        "mean_pair_count_or_score": float(s.mean()),
        "median_pair_count_or_score": float(s.median()),
        "share_min_count_pairs": float((s <= min_count).mean()),
    }


def calc_same_course_share(recs_df, reference_df):
    """
    Доля рекомендаций внутри того же course_id.
    """
    if recs_df is None or len(recs_df) == 0:
        return np.nan

    lesson_to_course = (
        reference_df[["lesson_id", "course_id"]]
        .drop_duplicates("lesson_id")
        .set_index("lesson_id")["course_id"]
        .to_dict()
    )

    tmp = recs_df.copy()
    tmp["course_A"] = tmp["lesson_A"].map(lesson_to_course)
    tmp["course_B"] = tmp["lesson_B"].map(lesson_to_course)

    valid = tmp["course_A"].notna() & tmp["course_B"].notna()

    if valid.sum() == 0:
        return np.nan

    return float((tmp.loc[valid, "course_A"] == tmp.loc[valid, "course_B"]).mean())


def build_extra_diagnostics_for_result(result, all_data, top_k=20, min_count=10):
    """
    Для результата run_one_train_source_legacy строит:
    - coverage по всем данным;
    - same_course_share;
    - pair strength;
    - bucket metrics.

    Важно:
    coverage считается legacy-функциями, чтобы совпадать с твоими notebook-выводами.
    """
    train_data = result["train_data"]
    test_data = result["test_data"]
    test_baskets = test_data.groupby("order_id")["lesson_id"].apply(list)
    rank_df = get_popularity_rank_from_value_counts(train_data)

    rows = []
    bucket_dfs = []

    for top_n, recs_df in result["recs_by_topn"].items():
        top_map = result["top_map_by_topn"][top_n]

        lesson_cov = coverage_lesson_legacy(top_map, all_data)
        basket_cov = coverage_by_K_legacy(top_map, all_data)
        strength = calc_pair_strength_legacy(recs_df, min_count=min_count)
        same_course = calc_same_course_share(recs_df, train_data)

        row = {
            "source_name": result["source_name"],
            "top_n": top_n,
            "same_course_share": same_course,
            **lesson_cov,
            **basket_cov,
            **strength,
        }
        rows.append(row)

        bdf = calc_metric_by_input_bucket_legacy(
            test_baskets,
            top_map,
            rank_df,
            top_k=top_k,
        )
        if not bdf.empty:
            bdf["source_name"] = result["source_name"]
            bdf["top_n"] = top_n
            bucket_dfs.append(bdf)

    diagnostics_df = pd.DataFrame(rows)
    bucket_metrics_df = pd.concat(bucket_dfs, ignore_index=True) if bucket_dfs else pd.DataFrame()

    return diagnostics_df, bucket_metrics_df


# =============================================================================
# 7. PLOTS
# =============================================================================

def plot_metric_curve_legacy(df, metric, group_col="source_name", x_col="top_n", title=None):
    plt.figure(figsize=(10, 5))

    for group_value in df[group_col].unique():
        part = df[df[group_col] == group_value].sort_values(x_col)

        plt.plot(
            part[x_col],
            part[metric],
            marker="o",
            label=str(group_value),
        )

    plt.xlabel("Top-N популярных уроков")
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_bucket_curve_legacy(bucket_df, metric, top_n=30000, source_name=None):
    df = bucket_df.copy()

    if top_n is not None:
        df = df[df["top_n"] == top_n]

    if source_name is not None:
        df = df[df["source_name"] == source_name]

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

    for group_value in df["source_name"].unique():
        part = df[df["source_name"] == group_value].sort_values("bucket")

        plt.plot(
            part["bucket"].astype(str),
            part[metric],
            marker="o",
            label=str(group_value),
        )

    plt.xlabel("Popularity bucket input lesson")
    plt.ylabel(metric)
    plt.title(f"{metric} by popularity bucket | Top-{top_n}")
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.legend()
    plt.show()


# =============================================================================
# 8. EXAMPLE USAGE
# =============================================================================
#
# from model import ArRecommender
#
# all_data = pd.read_csv("../data/all_data_obr_28-04.csv")
# data_crs_more2 = pd.read_csv("../data/more_2crs_data_obr_28-04.csv")
# data_les_more2 = pd.read_csv("../data/more_2les_data_obr_28-04.csv")
#
# # Вариант 1: максимально повторить notebook без seed:
# res = run_crs_and_les_legacy(
#     all_data=all_data,
#     data_crs_more2=data_crs_more2,
#     data_les_more2=data_les_more2,
#     model_class=ArRecommender,
#     top_ns=(5000, 7000, 10000, 15000, 20000, 30000),
#     top_k=20,
#     min_count=10,
#     min_support=0.0,
#     seed=None,
#     legacy_30000_full_train=True,
#     calc_next_item=False,
# )
#
# # Вариант 2: воспроизводимо, но split может не совпасть со старыми notebook-цифрами,
# # если они считались без random.seed:
# # res = run_crs_and_les_legacy(..., seed=42)
#
# item_metrics_all = res["item_metrics_all"]
# basket_metrics_all = res["basket_metrics_all"]
#
# item_metrics_all.to_csv("../data/metrics/item_metrics_legacy_long.csv", index=False)
# basket_metrics_all.to_csv("../data/metrics/basket_metrics_legacy_long.csv", index=False)
#
# # Широкие таблицы один-в-один как в notebook:
# res["crs"]["item_metrics_wide"].to_csv("../data/metrics/no_all_data_metrics_crs_recalc.csv")
# res["crs"]["basket_metrics_wide"].to_csv("../data/metrics/basket_to_basket_metrics_crs_recalc.csv")
# res["les"]["item_metrics_wide"].to_csv("../data/metrics/no_all_metrics_les_recalc.csv")
# res["les"]["basket_metrics_wide"].to_csv("../data/metrics/basket_to_basket_metrics_les_recalc.csv")
#
# # Диагностики: coverage / buckets / same-course / pair strength
# crs_diag, crs_bucket = build_extra_diagnostics_for_result(
#     res["crs"],
#     all_data=all_data,
#     top_k=20,
#     min_count=10,
# )
#
# les_diag, les_bucket = build_extra_diagnostics_for_result(
#     res["les"],
#     all_data=all_data,
#     top_k=20,
#     min_count=10,
# )
#
# diagnostics_all = pd.concat([crs_diag, les_diag], ignore_index=True)
# bucket_all = pd.concat([crs_bucket, les_bucket], ignore_index=True)
#
# diagnostics_all.to_csv("../data/metrics/diagnostics_legacy.csv", index=False)
# bucket_all.to_csv("../data/metrics/bucket_metrics_legacy.csv", index=False)
#
# # Графики cumulative-кривых
# plot_metric_curve_legacy(item_metrics_all, "recall@20", title="Item-to-basket Recall@20")
# plot_metric_curve_legacy(item_metrics_all, "hitrate@20", title="Item-to-basket Hitrate@20")
# plot_metric_curve_legacy(item_metrics_all, "map@20", title="Item-to-basket MAP@20")
#
# plot_metric_curve_legacy(basket_metrics_all, "recall@20", title="Basket-to-basket Recall@20")
# plot_metric_curve_legacy(basket_metrics_all, "hitrate@20", title="Basket-to-basket Hitrate@20")
#
# # Графики bucket-кривых: тут видно качество head/mid/tail
# plot_bucket_curve_legacy(bucket_all, "bucket_recall@20", top_n=30000)
# plot_bucket_curve_legacy(bucket_all, "bucket_hitrate@20", top_n=30000)
# plot_bucket_curve_legacy(bucket_all, "bucket_map@20", top_n=30000)
#
