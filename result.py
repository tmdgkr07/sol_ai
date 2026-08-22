"""CSV 의견 임베딩·클러스터링 분석 Gradio 애플리케이션."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)
GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"

GEMINI_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "issue": {
            "type": "string",
            "description": "클러스터 의견들이 공통으로 제기하는 핵심 문제 한 문장",
        },
        "root_cause": {
            "type": "string",
            "description": (
                "의견에 근거한 핵심 원인 한 문장. 근거가 부족하면 "
                "그 사실을 분명히 표시"
            ),
        },
        "action": {
            "type": "string",
            "description": "문제와 원인에 대응하는 구체적이고 실행 가능한 조치 한 문장",
        },
    },
    "required": ["issue", "root_cause", "action"],
}

DEFAULT_STOPWORDS = {
    "청년",
    "지역",
    "광주",
    "전남",
    "정보",
    "경우",
    "부분",
    "요즘",
    "실제로",
    "개인적으로",
    "생각합니다",
    "좋겠습니다",
    "어렵다",
    "어렵습니다",
    "필요하다",
    "필요합니다",
    "있으면이야",
    "그리고",
    "하지만",
    "때문",
    "정도",
    "관련",
}

EMPTY_CLUSTER_SUMMARY = pd.DataFrame(
    columns=[
        "cluster",
        "의견 수",
        "keywords",
        "대표 의견",
        "Issue",
        "Root Cause",
        "Action",
    ]
)
EMPTY_REPRESENTATIVES = pd.DataFrame(
    columns=["cluster", "rank", "similarity_to_center", "text"]
)
EMPTY_CLUSTER_OPINIONS = pd.DataFrame(
    columns=["번호", "cluster", "text"]
)
EMPTY_EXACT_RESULTS = pd.DataFrame(
    columns=["rank", "cluster", "text"]
)
EMPTY_SEMANTIC_RESULTS = pd.DataFrame(
    columns=["rank", "score", "cluster", "text"]
)
EMPTY_GEMINI_SUMMARY = pd.DataFrame(
    columns=["cluster", "Issue", "Root Cause", "Action"]
)


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    """모델을 서버 프로세스에서 한 번만 로드해 재사용한다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device}...")
    return SentenceTransformer(MODEL_NAME, device=device)


def _get_gemini_api_key() -> str | None:
    """환경변수 또는 Streamlit Secrets에서 Gemini API 키를 읽는다."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key

    try:
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get(
            "GOOGLE_API_KEY"
        )
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_gemini_client() -> genai.Client:
    """환경변수의 API 키로 Gemini 클라이언트를 한 번만 생성한다."""
    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini 요약을 사용하려면 배포 환경에 "
            "GEMINI_API_KEY를 설정해야 합니다."
        )
    return genai.Client(api_key=api_key)


def _parse_gemini_summary_response(response: Any) -> dict[str, str]:
    """Gemini structured output을 검증 가능한 dict로 변환한다."""
    parsed = getattr(response, "parsed", None)

    if hasattr(parsed, "model_dump"):
        parsed = parsed.model_dump()
    if not isinstance(parsed, dict):
        response_text = getattr(response, "text", None)
        if not response_text:
            raise ValueError("Gemini API가 비어 있는 응답을 반환했습니다.")
        parsed = json.loads(response_text)

    required_keys = ("issue", "root_cause", "action")
    missing_keys = [key for key in required_keys if not parsed.get(key)]
    if missing_keys:
        raise ValueError(
            "Gemini 요약 응답에 필수 항목이 없습니다: "
            + ", ".join(missing_keys)
        )

    return {key: str(parsed[key]).strip() for key in required_keys}


def _select_cluster_opinions_for_summary(
    cluster_df: pd.DataFrame,
    cluster_id: int,
    max_opinions: int = 60,
    max_total_chars: int = 18000,
) -> list[str]:
    """비용과 지연을 제한하면서 클러스터 의견을 대표 표본으로 고른다."""
    if len(cluster_df) > max_opinions:
        selected = cluster_df.sample(
            n=max_opinions,
            random_state=42 + cluster_id,
        )
    else:
        selected = cluster_df

    opinions: list[str] = []
    total_chars = 0

    for text in selected["text"].astype(str):
        # 지나치게 긴 단일 의견이 전체 입력을 차지하지 않도록 제한한다.
        text = text.strip()[:1200]
        if not text:
            continue
        if opinions and total_chars + len(text) > max_total_chars:
            break
        opinions.append(text)
        total_chars += len(text)

    return opinions


def _summarize_clusters_with_gemini(
    df: pd.DataFrame,
    keyword_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """클러스터별 Issue / Root Cause / Action을 자동 생성한다."""
    cluster_ids = sorted(int(value) for value in df["cluster"].unique())
    api_key = _get_gemini_api_key()

    if not api_key:
        summary_df = pd.DataFrame(
            {
                "cluster": cluster_ids,
                "Issue": "Gemini API 키 미설정",
                "Root Cause": "자동 요약을 실행하지 않았습니다.",
                "Action": "배포 환경에 GEMINI_API_KEY를 설정하세요.",
            }
        )
        return summary_df, "건너뜀: GEMINI_API_KEY가 설정되지 않았습니다."

    client = get_gemini_client()
    keyword_map = keyword_df.set_index("cluster")["keywords"].to_dict()
    results: list[dict[str, Any]] = []
    failed_clusters: list[int] = []

    for cluster_id in cluster_ids:
        cluster_df = df[df["cluster"] == cluster_id]
        opinions = _select_cluster_opinions_for_summary(
            cluster_df,
            cluster_id,
        )
        opinion_block = "\n".join(
            f"- {text}" for text in opinions
        )
        prompt = (
            "당신은 시민·사용자 의견을 분석하는 정책 분석가입니다. "
            "아래 클러스터 의견에 명시된 내용만 근거로 분석하세요. "
            "추측하거나 의견에 없는 지역적·행정적 사실을 만들지 마세요. "
            "Issue는 공통 핵심 문제, Root Cause는 의견에서 확인되거나 "
            "합리적으로 추론 가능한 원인, Action은 구체적이고 실행 가능한 "
            "대응 조치로 각각 간결한 한국어 한 문장으로 작성하세요. "
            "원인 근거가 부족하면 Root Cause에 이를 명시하세요.\n\n"
            f"클러스터: {cluster_id}\n"
            f"전체 의견 수: {len(cluster_df)}\n"
            f"TF-IDF 키워드: {keyword_map.get(cluster_id, '')}\n"
            f"분석 의견 표본 수: {len(opinions)}\n\n"
            f"의견:\n{opinion_block}"
        )

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": GEMINI_SUMMARY_SCHEMA,
                },
            )
            summary = _parse_gemini_summary_response(response)
            results.append(
                {
                    "cluster": cluster_id,
                    "Issue": summary["issue"],
                    "Root Cause": summary["root_cause"],
                    "Action": summary["action"],
                }
            )
        except Exception as error:
            failed_clusters.append(cluster_id)
            print(
                f"Gemini summary failed for cluster {cluster_id}: {error}"
            )
            results.append(
                {
                    "cluster": cluster_id,
                    "Issue": "Gemini 요약 생성 실패",
                    "Root Cause": "API 응답을 확인하지 못했습니다.",
                    "Action": "API 키, 할당량 및 서버 로그를 확인하세요.",
                }
            )

    completed_count = len(cluster_ids) - len(failed_clusters)
    if failed_clusters:
        status = (
            f"부분 완료: {completed_count}/{len(cluster_ids)}개 성공, "
            f"실패 cluster={failed_clusters}"
        )
    else:
        status = f"완료: {completed_count}개 클러스터 요약 생성"

    return pd.DataFrame(results), status


def read_and_clean_csv(file_path: str | Path) -> pd.DataFrame:
    """CSV를 읽고 text 컬럼을 정제한 DataFrame을 반환한다."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")

    raw_df: pd.DataFrame | None = None
    used_encoding: str | None = None
    encoding_errors: list[str] = []

    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            raw_df = pd.read_csv(path, encoding=encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError as error:
            encoding_errors.append(f"{encoding}: {error}")
        except pd.errors.EmptyDataError as error:
            raise ValueError("CSV 파일이 비어 있습니다.") from error
        except pd.errors.ParserError as error:
            raise ValueError(
                "CSV 형식이 올바르지 않습니다. "
                "열 개수, 쉼표 또는 따옴표를 확인해 주세요. "
                f"상세 오류: {error}"
            ) from error

    if raw_df is None:
        details = "\n".join(encoding_errors)
        raise ValueError(
            "UTF-8-SIG, UTF-8, CP949 인코딩으로 "
            f"CSV를 읽지 못했습니다.\n{details}"
        )

    # 원본 보호
    clean_df = raw_df.copy()
    clean_df.columns = clean_df.columns.astype(str).str.strip()

    if "text" not in clean_df.columns:
        raise ValueError(
            "'text' 컬럼이 없습니다. "
            f"현재 컬럼: {clean_df.columns.tolist()}"
        )

    before_rows = len(clean_df)
    before_nulls = int(clean_df["text"].isna().sum())
    normalized_text = clean_df["text"].astype("string").str.strip()
    before_empty = int(normalized_text.eq("").fillna(False).sum())

    valid_text = normalized_text.dropna()
    valid_text = valid_text[valid_text != ""]
    duplicate_count = int(valid_text.duplicated().sum())

    clean_df["text"] = normalized_text
    clean_df = clean_df.dropna(subset=["text"])
    clean_df = clean_df[clean_df["text"] != ""]
    clean_df = clean_df.drop_duplicates(subset=["text"])
    clean_df = clean_df.reset_index(drop=True)
    clean_df["n_chars"] = clean_df["text"].str.len()

    clean_df.attrs["cleaning_stats"] = {
        "encoding": used_encoding,
        "before_rows": before_rows,
        "after_rows": len(clean_df),
        "removed_rows": before_rows - len(clean_df),
        "null_count": before_nulls,
        "empty_count": before_empty,
        "duplicate_count": duplicate_count,
    }

    return clean_df


def _calculate_silhouette_scores(
    embeddings: np.ndarray,
    min_k: int = 3,
    max_k: int = 10,
) -> tuple[pd.DataFrame, int | None, Any | None]:
    """k별 silhouette score와 추천 k, Plotly Figure를 반환한다."""
    sample_count = len(embeddings)
    max_valid_k = min(max_k, sample_count - 1)

    if max_valid_k < min_k:
        return (
            pd.DataFrame(columns=["k", "silhouette_score"]),
            None,
            None,
        )

    results: list[dict[str, float | int]] = []
    silhouette_sample_size = 2000 if sample_count > 2000 else None

    for k in range(min_k, max_valid_k + 1):
        candidate_model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init="auto",
        )
        labels = candidate_model.fit_predict(embeddings)

        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(
            embeddings,
            labels,
            metric="euclidean",
            sample_size=silhouette_sample_size,
            random_state=42,
        )
        results.append({"k": k, "silhouette_score": float(score)})

    score_df = pd.DataFrame(results)
    if score_df.empty:
        return score_df, None, None

    best_index = score_df["silhouette_score"].idxmax()
    recommended_k = int(score_df.loc[best_index, "k"])
    best_score = float(score_df.loc[best_index, "silhouette_score"])

    figure = px.line(
        score_df,
        x="k",
        y="silhouette_score",
        markers=True,
        title="Silhouette Score by Number of Clusters",
        labels={
            "k": "Number of Clusters (k)",
            "silhouette_score": "Silhouette Score",
        },
    )
    figure.add_vline(x=recommended_k, line_dash="dash", line_color="red")
    figure.add_annotation(
        x=recommended_k,
        y=best_score,
        text=f"Recommended k={recommended_k}",
        showarrow=True,
        arrowhead=2,
    )
    figure.update_layout(height=450)

    return score_df, recommended_k, figure


def _get_cluster_keywords(
    df: pd.DataFrame,
    top_n: int = 6,
    additional_stopwords: list[str] | None = None,
) -> pd.DataFrame:
    """클러스터별 TF-IDF 상위 unigram·bigram을 반환한다."""
    stopwords = set(DEFAULT_STOPWORDS)
    if additional_stopwords:
        stopwords.update(additional_stopwords)

    cluster_documents = (
        df.groupby("cluster", sort=True)["text"]
        .apply(lambda values: " ".join(values))
        .reset_index(name="document")
    )

    vectorizer = TfidfVectorizer(
        stop_words=sorted(stopwords),
        token_pattern=r"(?u)\b[가-힣]{2,}\b",
        ngram_range=(1, 2),
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(
            cluster_documents["document"]
        )
    except ValueError:
        return pd.DataFrame(
            {
                "cluster": cluster_documents["cluster"],
                "keywords": "",
            }
        )

    feature_names = vectorizer.get_feature_names_out()
    results: list[dict[str, int | str]] = []

    for row_position, cluster_id in enumerate(
        cluster_documents["cluster"]
    ):
        scores = tfidf_matrix[row_position].toarray().ravel()
        ranked_indices = np.argsort(-scores)
        keywords = [
            feature_names[index]
            for index in ranked_indices
            if scores[index] > 0
        ][:top_n]
        results.append(
            {
                "cluster": int(cluster_id),
                "keywords": ", ".join(keywords),
            }
        )

    return pd.DataFrame(results)


def _get_representative_opinions(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    kmeans: KMeans,
    top_n: int = 3,
) -> pd.DataFrame:
    """클러스터 중심과 cosine similarity가 높은 실제 의견을 반환한다."""
    results: list[dict[str, Any]] = []

    for cluster_id in range(kmeans.n_clusters):
        row_positions = np.flatnonzero(kmeans.labels_ == cluster_id)
        if len(row_positions) == 0:
            continue

        center = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        similarities = cosine_similarity(
            embeddings[row_positions], center
        ).ravel()
        ranked_local_positions = np.argsort(
            -similarities, kind="stable"
        )[:top_n]

        for rank, local_position in enumerate(
            ranked_local_positions, start=1
        ):
            row_position = row_positions[local_position]
            results.append(
                {
                    "cluster": cluster_id,
                    "rank": rank,
                    "similarity_to_center": float(
                        similarities[local_position]
                    ),
                    "text": df.iloc[row_position]["text"],
                }
            )

    return pd.DataFrame(
        results,
        columns=[
            "cluster",
            "rank",
            "similarity_to_center",
            "text",
        ],
    )


def _make_topic_figure(
    plot_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
) -> Any:
    """동일한 스타일의 Plotly topic map을 만든다."""
    cluster_order = [
        f"Cluster {cluster_id}"
        for cluster_id in sorted(plot_df["cluster"].unique())
    ]

    figure = px.scatter(
        plot_df,
        x=x_column,
        y=y_column,
        color="cluster_label",
        hover_name="text",
        hover_data={
            "cluster": True,
            "cluster_label": False,
            x_column: ":.3f",
            y_column: ":.3f",
        },
        category_orders={"cluster_label": cluster_order},
        labels={
            x_column: x_label,
            y_column: y_label,
            "cluster_label": "Cluster",
        },
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    figure.update_traces(
        marker={
            "size": 9,
            "opacity": 0.75,
            "line": {"width": 0.5, "color": "white"},
        }
    )
    figure.update_layout(
        height=650,
        legend_title_text="Cluster",
        hoverlabel={"align": "left", "namelength": -1},
    )
    return figure


def _build_topic_maps(
    df: pd.DataFrame, embeddings: np.ndarray
) -> dict[str, Any]:
    """PCA와 UMAP 기반 2차원 topic map을 생성한다."""
    pca_model = PCA(n_components=2)
    pca_coordinates = pca_model.fit_transform(embeddings)
    df["pca_x"] = pca_coordinates[:, 0]
    df["pca_y"] = pca_coordinates[:, 1]

    n_neighbors = max(2, min(15, len(df) - 1))
    umap_model = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        n_jobs=1,
        init="random",
    )
    umap_coordinates = umap_model.fit_transform(embeddings)
    df["umap_x"] = umap_coordinates[:, 0]
    df["umap_y"] = umap_coordinates[:, 1]

    plot_df = df.copy()
    plot_df["cluster_label"] = (
        "Cluster " + plot_df["cluster"].astype(str)
    )

    return {
        "pca_figure": _make_topic_figure(
            plot_df,
            "pca_x",
            "pca_y",
            "PCA Opinion Topic Map",
            "PCA Component 1",
            "PCA Component 2",
        ),
        "umap_figure": _make_topic_figure(
            plot_df,
            "umap_x",
            "umap_y",
            "UMAP Opinion Topic Map",
            "UMAP Dimension 1",
            "UMAP Dimension 2",
        ),
        "pca_model": pca_model,
        "umap_model": umap_model,
        "pca_explained_variance": float(
            pca_model.explained_variance_ratio_.sum()
        ),
    }


def build_analysis(file_path: str | Path, n_clusters: int) -> dict[str, Any]:
    """정제부터 검색 state 생성까지 전체 분석 파이프라인을 실행한다."""
    df = read_and_clean_csv(file_path)
    n_clusters = int(n_clusters)

    if n_clusters < 2:
        raise ValueError("n_clusters는 2 이상이어야 합니다.")
    if len(df) < 3:
        raise ValueError("분석할 문장이 최소 3개 필요합니다.")
    if n_clusters >= len(df):
        raise ValueError(
            f"n_clusters={n_clusters}는 정제 후 문장 수 "
            f"{len(df)}보다 작아야 합니다."
        )

    cleaning_stats = dict(df.attrs.get("cleaning_stats", {}))
    model = get_embedding_model()
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    silhouette_df, recommended_k, silhouette_figure = (
        _calculate_silhouette_scores(embeddings)
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto",
    )
    df["cluster"] = kmeans.fit_predict(embeddings)

    keyword_df = _get_cluster_keywords(df, top_n=6)
    representative_df = _get_representative_opinions(
        df, embeddings, kmeans, top_n=3
    )
    gemini_summary_df, gemini_summary_status = (
        _summarize_clusters_with_gemini(df, keyword_df)
    )
    first_representative_df = (
        representative_df[representative_df["rank"] == 1][
            ["cluster", "text"]
        ]
        .rename(columns={"text": "대표 의견"})
        .reset_index(drop=True)
    )
    count_df = (
        df.groupby("cluster").size().reset_index(name="의견 수")
    )
    cluster_summary = (
        count_df.merge(
            keyword_df,
            on="cluster",
            how="left",
            validate="one_to_one",
        )
        .merge(
            first_representative_df,
            on="cluster",
            how="left",
            validate="one_to_one",
        )
        .merge(
            gemini_summary_df,
            on="cluster",
            how="left",
            validate="one_to_one",
        )
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    topic_maps = _build_topic_maps(df, embeddings)
    search_state = {
        "texts": df["text"].tolist(),
        "clusters": df["cluster"].to_numpy(copy=True),
        "embeddings": embeddings,
    }

    return {
        "df": df,
        "embeddings": embeddings,
        "kmeans": kmeans,
        "keyword_df": keyword_df,
        "representative_df": representative_df,
        "gemini_summary_df": gemini_summary_df,
        "gemini_summary_status": gemini_summary_status,
        "cluster_summary": cluster_summary,
        "silhouette_df": silhouette_df,
        "recommended_k": recommended_k,
        "silhouette_figure": silhouette_figure,
        "pca_topic_map": topic_maps["pca_figure"],
        "umap_topic_map": topic_maps["umap_figure"],
        "pca_model": topic_maps["pca_model"],
        "umap_model": topic_maps["umap_model"],
        "pca_explained_variance": topic_maps[
            "pca_explained_variance"
        ],
        "search_state": search_state,
        "cleaning_stats": cleaning_stats,
    }


def semantic_search(
    query: str,
    search_state: dict[str, Any],
    top_k: int = 5,
) -> pd.DataFrame:
    """의미적으로 유사한 Top-K 의견을 반환한다."""
    if not query or not query.strip():
        raise ValueError("검색어를 입력해 주세요.")
    if not search_state:
        raise ValueError("먼저 CSV 분석을 실행해 주세요.")

    texts = search_state["texts"]
    clusters = search_state["clusters"]
    embeddings = search_state["embeddings"]
    top_k = min(max(int(top_k), 1), len(texts))

    query_embedding = get_embedding_model().encode(
        [query.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(-scores, kind="stable")[:top_k]

    return pd.DataFrame(
        {
            "rank": range(1, len(top_indices) + 1),
            "score": scores[top_indices],
            "cluster": clusters[top_indices],
            "text": [texts[index] for index in top_indices],
        }
    )


def exact_keyword_search(
    keyword: str,
    search_state: dict[str, Any],
    top_k: int = 5,
) -> pd.DataFrame:
    """키워드 문자열이 그대로 포함된 의견을 반환한다."""
    if not keyword or not keyword.strip():
        raise ValueError("검색 키워드를 입력해 주세요.")
    if not search_state:
        raise ValueError("먼저 CSV 분석을 실행해 주세요.")

    keyword = keyword.strip()
    texts = search_state["texts"]
    clusters = search_state["clusters"]
    matched_positions = [
        position
        for position, text in enumerate(texts)
        if keyword in text
    ][: max(int(top_k), 1)]

    return pd.DataFrame(
        {
            "rank": range(1, len(matched_positions) + 1),
            "cluster": [clusters[pos] for pos in matched_positions],
            "text": [texts[pos] for pos in matched_positions],
        }
    )


def filter_cluster_opinions(
    cluster_id: int | float | str | None,
    search_state: dict[str, Any] | None,
) -> tuple[str, pd.DataFrame]:
    """선택한 클러스터의 의견만 반환한다."""
    if cluster_id is None or not search_state:
        return "클러스터를 선택해 주세요.", EMPTY_CLUSTER_OPINIONS.copy()

    cluster_id = int(cluster_id)
    texts = search_state["texts"]
    clusters = search_state["clusters"]
    matched_positions = np.flatnonzero(clusters == cluster_id)
    filtered_df = pd.DataFrame(
        {
            "번호": range(1, len(matched_positions) + 1),
            "cluster": [
                int(clusters[position])
                for position in matched_positions
            ],
            "text": [texts[position] for position in matched_positions],
        }
    )
    summary = (
        f"### Cluster {cluster_id} 의견 — 총 {len(filtered_df):,}개"
    )
    return summary, filtered_df


def _save_uploaded_csv(uploaded_file: Any) -> str:
    """Streamlit UploadedFile을 임시 CSV 경로로 저장한다."""
    suffix = Path(getattr(uploaded_file, "name", "upload.csv")).suffix
    if suffix.lower() != ".csv":
        suffix = ".csv"

    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        delete=False,
    ) as temporary_file:
        temporary_file.write(uploaded_file.getvalue())
        return temporary_file.name


def _build_analysis_from_upload(
    uploaded_file: Any,
    n_clusters: int,
) -> dict[str, Any]:
    """업로드 파일을 분석하고 임시 파일을 즉시 정리한다."""
    temporary_path = _save_uploaded_csv(uploaded_file)
    try:
        return build_analysis(temporary_path, int(n_clusters))
    finally:
        Path(temporary_path).unlink(missing_ok=True)


def _initialize_streamlit_state() -> None:
    """Streamlit rerun 사이에 유지할 사용자별 상태를 초기화한다."""
    defaults: dict[str, Any] = {
        "analysis": None,
        "selected_cluster": None,
        "search_query": "취업지원",
        "search_top_k": 5,
        "exact_results": EMPTY_EXACT_RESULTS.copy(),
        "semantic_results": EMPTY_SEMANTIC_RESULTS.copy(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _set_default_search_results(analysis: dict[str, Any]) -> None:
    """새 분석 직후 기본 검색어의 exact/semantic 결과를 저장한다."""
    query = str(st.session_state.get("search_query", "취업지원")).strip()
    top_k = int(st.session_state.get("search_top_k", 5))

    if not query:
        st.session_state.exact_results = EMPTY_EXACT_RESULTS.copy()
        st.session_state.semantic_results = EMPTY_SEMANTIC_RESULTS.copy()
        return

    search_state = analysis["search_state"]
    st.session_state.exact_results = exact_keyword_search(
        query,
        search_state,
        top_k,
    )
    st.session_state.semantic_results = semantic_search(
        query,
        search_state,
        top_k,
    )


def _render_analysis_header(analysis: dict[str, Any]) -> None:
    """정제 및 분석 상태를 요약해서 표시한다."""
    stats = analysis["cleaning_stats"]
    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "정제 전",
        f"{stats.get('before_rows', 0):,}",
    )
    metric_columns[1].metric(
        "정제 후",
        f"{stats.get('after_rows', 0):,}",
    )
    metric_columns[2].metric(
        "선택한 k",
        analysis["kmeans"].n_clusters,
    )
    metric_columns[3].metric(
        "Silhouette 추천 k",
        analysis["recommended_k"]
        if analysis["recommended_k"] is not None
        else "계산 불가",
    )
    metric_columns[4].metric(
        "PCA 설명 분산",
        f"{analysis['pca_explained_variance']:.4f}",
    )

    gemini_status = analysis["gemini_summary_status"]
    if gemini_status.startswith("완료"):
        st.success(f"Gemini Issue / Root Cause / Action: {gemini_status}")
    elif gemini_status.startswith("부분"):
        st.warning(f"Gemini Issue / Root Cause / Action: {gemini_status}")
    else:
        st.info(f"Gemini Issue / Root Cause / Action: {gemini_status}")


def _render_cluster_filter(analysis: dict[str, Any]) -> None:
    """선택한 클러스터의 실제 의견을 표시한다."""
    search_state = analysis["search_state"]
    cluster_ids = sorted(
        int(cluster_id)
        for cluster_id in np.unique(search_state["clusters"])
    )

    if st.session_state.selected_cluster not in cluster_ids:
        st.session_state.selected_cluster = cluster_ids[0]

    selected_cluster = st.selectbox(
        "클러스터 선택",
        options=cluster_ids,
        format_func=lambda value: f"Cluster {value}",
        key="selected_cluster",
    )
    cluster_status, filtered_df = filter_cluster_opinions(
        selected_cluster,
        search_state,
    )
    st.markdown(cluster_status)
    st.dataframe(
        filtered_df,
        width="stretch",
        hide_index=True,
        height=520,
    )


def _render_search(analysis: dict[str, Any]) -> None:
    """Exact keyword search와 semantic search를 비교한다."""
    with st.form("opinion_search_form"):
        search_columns = st.columns([3, 1])
        with search_columns[0]:
            st.text_input(
                "검색어",
                key="search_query",
                placeholder="예: 취업지원",
            )
        with search_columns[1]:
            st.slider(
                "Top-K",
                minimum=1,
                maximum=20,
                step=1,
                key="search_top_k",
            )

        submitted = st.form_submit_button(
            "검색",
            type="primary",
            width="stretch",
        )

    if submitted:
        query = str(st.session_state.search_query).strip()
        if not query:
            st.warning("검색어를 입력해 주세요.")
        else:
            try:
                search_state = analysis["search_state"]
                st.session_state.exact_results = exact_keyword_search(
                    query,
                    search_state,
                    int(st.session_state.search_top_k),
                )
                st.session_state.semantic_results = semantic_search(
                    query,
                    search_state,
                    int(st.session_state.search_top_k),
                )
            except Exception as error:
                st.error(str(error))

    exact_column, semantic_column = st.columns(2)

    with exact_column:
        st.subheader("Exact Keyword Search")
        st.dataframe(
            st.session_state.exact_results,
            width="stretch",
            hide_index=True,
            height=430,
        )

    with semantic_column:
        st.subheader("Semantic Search")
        semantic_results = st.session_state.semantic_results.copy()
        if "score" in semantic_results.columns:
            semantic_results["score"] = semantic_results["score"].round(4)
        st.dataframe(
            semantic_results,
            width="stretch",
            hide_index=True,
            height=430,
        )


def _render_analysis(analysis: dict[str, Any]) -> None:
    """전체 분석 결과 탭을 렌더링한다."""
    _render_analysis_header(analysis)

    tabs = st.tabs(
        [
            "클러스터 요약",
            "Gemini 요약",
            "Silhouette",
            "PCA Topic Map",
            "UMAP Topic Map",
            "클러스터 의견",
            "의견 검색",
        ]
    )

    with tabs[0]:
        st.dataframe(
            analysis["cluster_summary"],
            width="stretch",
            hide_index=True,
            height=480,
        )
        with st.expander("클러스터별 대표 의견 Top 3"):
            representative_df = analysis["representative_df"].copy()
            representative_df["similarity_to_center"] = (
                representative_df["similarity_to_center"].round(4)
            )
            st.dataframe(
                representative_df,
                width="stretch",
                hide_index=True,
            )

    with tabs[1]:
        st.caption(
            "GEMINI_API_KEY가 설정된 경우, 클러스터별 의견 표본이 "
            "Gemini API로 전송되어 아래 요약이 자동 생성됩니다."
        )
        st.dataframe(
            analysis["gemini_summary_df"],
            width="stretch",
            hide_index=True,
            height=500,
        )

    with tabs[2]:
        if analysis["silhouette_figure"] is None:
            st.info("Silhouette score를 계산할 데이터가 부족합니다.")
        else:
            st.plotly_chart(
                analysis["silhouette_figure"],
                width="stretch",
                config={"displaylogo": False},
            )
            with st.expander("Silhouette score 원본 값"):
                score_df = analysis["silhouette_df"].copy()
                score_df["silhouette_score"] = (
                    score_df["silhouette_score"].round(4)
                )
                st.dataframe(score_df, width="stretch", hide_index=True)

    with tabs[3]:
        st.plotly_chart(
            analysis["pca_topic_map"],
            width="stretch",
            config={"displaylogo": False},
        )

    with tabs[4]:
        st.plotly_chart(
            analysis["umap_topic_map"],
            width="stretch",
            config={"displaylogo": False},
        )

    with tabs[5]:
        _render_cluster_filter(analysis)

    with tabs[6]:
        _render_search(analysis)


def main() -> None:
    """Streamlit 애플리케이션 진입점."""
    st.set_page_config(
        page_title="의견 분석 대시보드",
        page_icon="📊",
        layout="wide",
    )
    _initialize_streamlit_state()

    st.title("의견 분석 대시보드")
    st.write(
        "CSV의 `text` 컬럼을 임베딩하고 클러스터별 핵심 주제, "
        "대표 의견, Issue / Root Cause / Action을 분석합니다."
    )

    with st.sidebar:
        st.header("분석 설정")
        uploaded_file = st.file_uploader(
            "CSV 파일",
            type=["csv"],
            accept_multiple_files=False,
            help="text 컬럼이 포함된 CSV 파일을 업로드하세요.",
        )
        n_clusters = st.number_input(
            "클러스터 수 (k)",
            min_value=2,
            max_value=50,
            value=7,
            step=1,
        )
        analyze_button = st.button(
            "분석 실행",
            type="primary",
            width="stretch",
        )
        st.divider()
        st.caption(
            "Gemini 자동 요약을 사용하려면 Streamlit Secrets 또는 "
            "서버 환경변수에 GEMINI_API_KEY를 설정하세요. 의견 표본은 "
            "요약 생성을 위해 Gemini API로 전송됩니다."
        )

    if analyze_button:
        if uploaded_file is None:
            st.error("CSV 파일을 업로드해 주세요.")
        else:
            try:
                with st.spinner(
                    "CSV 정제, 임베딩, 클러스터링 및 AI 요약을 "
                    "실행하고 있습니다..."
                ):
                    analysis = _build_analysis_from_upload(
                        uploaded_file,
                        int(n_clusters),
                    )
                    st.session_state.analysis = analysis
                    st.session_state.selected_cluster = None
                    _set_default_search_results(analysis)
                st.success("분석이 완료되었습니다.")
            except Exception as error:
                st.error(f"분석에 실패했습니다: {error}")

    analysis = st.session_state.analysis
    if analysis is None:
        st.info(
            "왼쪽 사이드바에서 CSV 파일과 클러스터 수를 설정한 뒤 "
            "'분석 실행'을 누르세요."
        )
        return

    _render_analysis(analysis)


if __name__ == "__main__":
    main()
