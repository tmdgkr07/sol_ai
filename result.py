"""CSV 의견 임베딩·클러스터링 분석 Streamlit 애플리케이션."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
from sklearn.neighbors import NearestNeighbors
from umap import UMAP


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)
GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"

VOICE_TYPE_ORDER = [
    "대표 의견",
    "경계 의견",
    "숨은 목소리",
    "일반 의견",
]
VOICE_SYMBOL_MAP = {
    "대표 의견": "star",
    "경계 의견": "diamond-open",
    "숨은 목소리": "x",
    "일반 의견": "circle",
}

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


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    """배열을 0~1로 정규화하되 상수 배열은 0으로 반환한다."""
    values = np.asarray(values, dtype=float)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.isclose(minimum, maximum):
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / (maximum - minimum)


def _classify_opinion_roles(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    kmeans: KMeans,
    representative_per_cluster: int = 3,
) -> pd.DataFrame:
    """대표·경계·숨은 목소리 후보를 임베딩 거리로 분류한다.

    경계 의견은 1·2순위 클러스터 유사도 차이가 작은 문장이고,
    숨은 목소리는 클러스터 중심에서는 멀지만 가까운 이웃끼리의
    응집도가 상대적으로 높은 작은 의제 후보이다.
    """
    sample_count = len(df)
    center_similarities = cosine_similarity(
        embeddings,
        kmeans.cluster_centers_,
    )
    assigned_clusters = kmeans.labels_.astype(int)
    assigned_similarity = center_similarities[
        np.arange(sample_count), assigned_clusters
    ]

    other_similarities = center_similarities.copy()
    other_similarities[
        np.arange(sample_count), assigned_clusters
    ] = -np.inf
    second_similarity = other_similarities.max(axis=1)
    cluster_margin = assigned_similarity - second_similarity

    neighbor_count = min(6, sample_count)
    neighbor_model = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
    )
    neighbor_distances, _ = neighbor_model.fit(embeddings).kneighbors(
        embeddings
    )
    if neighbor_count > 1:
        local_cohesion = 1.0 - neighbor_distances[:, 1:].mean(axis=1)
    else:
        local_cohesion = np.ones(sample_count, dtype=float)

    representative_mask = np.zeros(sample_count, dtype=bool)
    for cluster_id in range(kmeans.n_clusters):
        positions = np.flatnonzero(assigned_clusters == cluster_id)
        ranked = positions[
            np.argsort(
                -assigned_similarity[positions],
                kind="stable",
            )[:representative_per_cluster]
        ]
        representative_mask[ranked] = True

    boundary_cutoff = float(np.quantile(cluster_margin, 0.15))
    boundary_mask = (
        (cluster_margin <= boundary_cutoff)
        & ~representative_mask
    )

    hidden_voice_score = (
        _minmax_scale(1.0 - assigned_similarity) * 0.65
        + _minmax_scale(local_cohesion) * 0.35
    )
    hidden_eligible = ~representative_mask & ~boundary_mask
    hidden_mask = np.zeros(sample_count, dtype=bool)
    eligible_positions = np.flatnonzero(hidden_eligible)
    if len(eligible_positions) > 0:
        hidden_cutoff = float(
            np.quantile(hidden_voice_score[eligible_positions], 0.90)
        )
        center_cutoff = float(np.quantile(assigned_similarity, 0.50))
        hidden_mask = (
            hidden_eligible
            & (hidden_voice_score >= hidden_cutoff)
            & (assigned_similarity <= center_cutoff)
        )
        if not hidden_mask.any():
            best_position = eligible_positions[
                np.argmax(hidden_voice_score[eligible_positions])
            ]
            hidden_mask[best_position] = True

    role = np.full(sample_count, "일반 의견", dtype=object)
    role[hidden_mask] = "숨은 목소리"
    role[boundary_mask] = "경계 의견"
    role[representative_mask] = "대표 의견"

    result = df[["text", "cluster"]].copy()
    result.insert(0, "row_id", np.arange(sample_count, dtype=int))
    result["voice_type"] = role
    result["center_similarity"] = assigned_similarity
    result["second_similarity"] = second_similarity
    result["cluster_margin"] = cluster_margin
    result["local_cohesion"] = local_cohesion
    result["hidden_voice_score"] = hidden_voice_score
    return result


def _make_voice_topic_figure(opinion_role_df: pd.DataFrame) -> Any:
    """의견 역할을 도형으로 구분한 선택 가능한 UMAP 지도를 만든다."""
    plot_df = opinion_role_df.copy()
    plot_df["cluster_label"] = (
        "Cluster " + plot_df["cluster"].astype(str)
    )
    cluster_order = [
        f"Cluster {cluster_id}"
        for cluster_id in sorted(plot_df["cluster"].unique())
    ]

    figure = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color="cluster_label",
        symbol="voice_type",
        hover_name="text",
        custom_data=[
            "row_id",
            "voice_type",
            "cluster_margin",
            "center_similarity",
            "local_cohesion",
        ],
        hover_data={
            "row_id": False,
            "cluster": True,
            "cluster_label": False,
            "voice_type": True,
            "cluster_margin": ":.4f",
            "center_similarity": ":.4f",
            "local_cohesion": ":.4f",
            "umap_x": ":.3f",
            "umap_y": ":.3f",
        },
        category_orders={
            "cluster_label": cluster_order,
            "voice_type": VOICE_TYPE_ORDER,
        },
        symbol_map=VOICE_SYMBOL_MAP,
        labels={
            "umap_x": "UMAP Dimension 1",
            "umap_y": "UMAP Dimension 2",
            "cluster_label": "Cluster",
            "voice_type": "의견 유형",
            "cluster_margin": "클러스터 경계 여유",
            "center_similarity": "중심 유사도",
            "local_cohesion": "주변 응집도",
        },
        title="숨은 목소리 UMAP 정책 탐색 지도",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    for trace in figure.data:
        name = str(getattr(trace, "name", ""))
        if "대표 의견" in name:
            trace.update(marker={"size": 15, "opacity": 0.95})
        elif "경계 의견" in name:
            trace.update(marker={"size": 12, "opacity": 0.9})
        elif "숨은 목소리" in name:
            trace.update(marker={"size": 13, "opacity": 0.95})
        else:
            trace.update(marker={"size": 8, "opacity": 0.55})

    figure.update_layout(
        height=720,
        dragmode="lasso",
        legend_title_text="Cluster · 의견 유형",
        hoverlabel={"align": "left", "namelength": -1},
    )
    return figure


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
    opinion_role_df = _classify_opinion_roles(
        df,
        embeddings,
        kmeans,
    )
    opinion_role_df["pca_x"] = df["pca_x"].to_numpy(copy=True)
    opinion_role_df["pca_y"] = df["pca_y"].to_numpy(copy=True)
    opinion_role_df["umap_x"] = df["umap_x"].to_numpy(copy=True)
    opinion_role_df["umap_y"] = df["umap_y"].to_numpy(copy=True)
    voice_topic_map = _make_voice_topic_figure(opinion_role_df)
    search_state = {
        "texts": df["text"].tolist(),
        "clusters": df["cluster"].to_numpy(copy=True),
        "embeddings": embeddings,
    }

    analysis_id = hashlib.sha256(
        ("\n".join(df["text"].tolist()) + f"\nk={n_clusters}").encode(
            "utf-8"
        )
    ).hexdigest()[:12]

    return {
        "analysis_id": analysis_id,
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
        "voice_topic_map": voice_topic_map,
        "opinion_role_df": opinion_role_df,
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


def _extract_keywords_from_texts(
    texts: list[str],
    top_n: int = 8,
) -> str:
    """선택 의견들의 평균 TF-IDF 기준 상위 키워드를 반환한다."""
    if not texts:
        return ""

    vectorizer = TfidfVectorizer(
        stop_words=sorted(DEFAULT_STOPWORDS),
        token_pattern=r"(?u)\b[가-힣]{2,}\b",
        ngram_range=(1, 2),
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return ""

    mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
    feature_names = vectorizer.get_feature_names_out()
    ranked = np.argsort(-mean_scores, kind="stable")
    keywords = [
        feature_names[index]
        for index in ranked
        if mean_scores[index] > 0
    ][:top_n]
    return ", ".join(keywords)


def _extract_selected_row_ids(event: Any) -> list[int]:
    """Streamlit Plotly 선택 이벤트에서 전역 row_id를 복원한다."""
    if event is None:
        return []

    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if not selection:
        return []

    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points", [])

    row_ids: list[int] = []
    for point in points or []:
        custom_data = (
            point.get("customdata", [])
            if isinstance(point, dict)
            else getattr(point, "customdata", [])
        )
        if custom_data is None or len(custom_data) == 0:
            continue
        try:
            row_ids.append(int(custom_data[0]))
        except (TypeError, ValueError):
            continue
    return sorted(set(row_ids))


def _get_selected_opinions(
    analysis: dict[str, Any],
    row_ids: list[int],
) -> pd.DataFrame:
    """선택된 row_id에 해당하는 의견을 지도 순서와 무관하게 반환한다."""
    role_df = analysis["opinion_role_df"]
    valid_ids = [
        row_id
        for row_id in row_ids
        if 0 <= row_id < len(role_df)
    ]
    if not valid_ids:
        return role_df.iloc[0:0].copy()
    return (
        role_df.set_index("row_id", drop=False)
        .loc[valid_ids]
        .reset_index(drop=True)
    )


def _get_selection_representatives(
    selected_df: pd.DataFrame,
    embeddings: np.ndarray,
    top_n: int = 3,
) -> pd.DataFrame:
    """선택 영역의 임베딩 중심에 가까운 실제 의견을 반환한다."""
    if selected_df.empty:
        return pd.DataFrame(
            columns=["순위", "중심 유사도", "의견 유형", "text"]
        )

    row_ids = selected_df["row_id"].astype(int).to_numpy()
    selected_embeddings = embeddings[row_ids]
    centroid = selected_embeddings.mean(axis=0, keepdims=True)
    similarities = cosine_similarity(
        selected_embeddings,
        centroid,
    ).ravel()
    ranked = np.argsort(-similarities, kind="stable")[:top_n]
    return pd.DataFrame(
        {
            "순위": range(1, len(ranked) + 1),
            "중심 유사도": similarities[ranked],
            "의견 유형": selected_df.iloc[ranked]["voice_type"].tolist(),
            "text": selected_df.iloc[ranked]["text"].tolist(),
        }
    )


def _is_safe_http_url(value: Any) -> bool:
    """Grounding 결과 링크가 일반 HTTP(S) URL인지 확인한다."""
    try:
        parsed = urlparse(str(value))
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _escape_markdown_label(value: Any) -> str:
    """Markdown 링크 라벨에서 구문 문자를 제거한다."""
    return (
        str(value or "출처")
        .replace("[", "(")
        .replace("]", ")")
        .replace("\n", " ")
        .strip()
    )


def _extract_grounding_result(response: Any) -> dict[str, Any]:
    """Gemini 응답에 인라인 출처 링크와 출처 목록을 결합한다."""
    text = str(getattr(response, "text", "") or "").strip()
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return {
            "markdown": text,
            "sources": [],
            "queries": [],
            "search_widget_html": "",
        }

    grounding = getattr(candidates[0], "grounding_metadata", None)
    if grounding is None:
        return {
            "markdown": text,
            "sources": [],
            "queries": [],
            "search_widget_html": "",
        }

    chunks = list(getattr(grounding, "grounding_chunks", None) or [])
    supports = list(getattr(grounding, "grounding_supports", None) or [])
    sources: list[dict[str, Any]] = []
    source_by_chunk: dict[int, dict[str, Any]] = {}

    for chunk_index, chunk in enumerate(chunks):
        web = getattr(chunk, "web", None)
        uri = getattr(web, "uri", None) if web is not None else None
        title = getattr(web, "title", None) if web is not None else None
        if not _is_safe_http_url(uri):
            continue
        source = {
            "number": chunk_index + 1,
            "title": _escape_markdown_label(title),
            "uri": str(uri),
        }
        sources.append(source)
        source_by_chunk[chunk_index] = source

    for support in sorted(
        supports,
        key=lambda item: int(
            getattr(getattr(item, "segment", None), "end_index", 0) or 0
        ),
        reverse=True,
    ):
        segment = getattr(support, "segment", None)
        end_index = getattr(segment, "end_index", None)
        chunk_indices = list(
            getattr(support, "grounding_chunk_indices", None) or []
        )
        if end_index is None:
            continue
        links = [
            f"[{source_by_chunk[index]['number']}]"
            f"({source_by_chunk[index]['uri']})"
            for index in chunk_indices
            if index in source_by_chunk
        ]
        if not links:
            continue
        position = min(max(int(end_index), 0), len(text))
        text = text[:position] + " " + " ".join(links) + text[position:]

    search_entry_point = getattr(grounding, "search_entry_point", None)
    rendered_content = (
        getattr(search_entry_point, "rendered_content", "")
        if search_entry_point is not None
        else ""
    )
    queries = list(getattr(grounding, "web_search_queries", None) or [])
    return {
        "markdown": text,
        "sources": sources,
        "queries": [str(query) for query in queries],
        "search_widget_html": str(rendered_content or ""),
    }


def _generate_grounded_policy_report(
    selected_df: pd.DataFrame,
    keywords: str,
    policy_scope: str,
) -> dict[str, Any]:
    """선택 의견을 실제 정책·지원사업 근거가 있는 제안으로 변환한다."""
    if selected_df.empty:
        raise ValueError("정책 제안을 생성할 의견을 먼저 선택해 주세요.")

    role_priority = {
        "숨은 목소리": 0,
        "경계 의견": 1,
        "대표 의견": 2,
        "일반 의견": 3,
    }
    prompt_df = selected_df.copy()
    prompt_df["_priority"] = prompt_df["voice_type"].map(
        role_priority
    ).fillna(4)
    prompt_df = prompt_df.sort_values(
        ["_priority", "hidden_voice_score"],
        ascending=[True, False],
        kind="stable",
    ).head(50)

    opinion_lines: list[str] = []
    total_chars = 0
    for row in prompt_df.itertuples(index=False):
        line = f"- [{row.voice_type} / Cluster {row.cluster}] {row.text}"
        if opinion_lines and total_chars + len(line) > 16000:
            break
        opinion_lines.append(line)
        total_chars += len(line)

    scope = str(policy_scope or "대한민국").strip()[:200]
    prompt = (
        "당신은 시민 의견을 실제 공공정책과 연결하는 정책 분석가입니다. "
        "Google Search를 사용해 현재 운영 중이거나 최근 공식 발표된 정책·"
        "지원사업·공공서비스를 확인하고, 공식 정부·지자체·공공기관 출처를 "
        "우선하세요. 확인하지 못한 사업명, 수치, 시행기관을 만들지 마세요. "
        "종료되었거나 현재 상태가 불명확하면 그 사실을 명시하세요.\n\n"
        "아래 제목을 정확히 유지한 한국어 Markdown 보고서를 작성하세요.\n"
        "## 선택 의견의 Issue\n"
        "## Root Cause\n"
        "## 권장 Action\n"
        "## 실제 유사 정책·지원사업\n"
        "실제 사례를 2~4개 제시하고, 사업명·운영기관·핵심 내용·현재 의견과의 "
        "연결점을 설명하세요. 검색으로 확인한 사실만 쓰세요.\n"
        "## 차별화 제안\n"
        "기존 정책과 겹치는 부분, 부족한 부분, 새로 설계할 기능을 구분하세요.\n"
        "## 90일 실행안\n"
        "담당 주체와 검증 지표가 포함된 3단계 실행안을 제시하세요.\n\n"
        f"기준일: {date.today().isoformat()}\n"
        f"정책 검색 범위: {scope}\n"
        f"선택 의견 수: {len(selected_df)}\n"
        f"선택 영역 키워드: {keywords or '추출되지 않음'}\n\n"
        "선택 의견:\n" + "\n".join(opinion_lines)
    )

    client = get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
        config={"tools": [{"google_search": {}}]},
    )
    report = _extract_grounding_result(response)
    if not report["markdown"]:
        raise ValueError("Gemini API가 비어 있는 정책 보고서를 반환했습니다.")
    report["scope"] = scope
    report["selected_count"] = len(selected_df)
    return report


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
        "policy_scope": "광주광역시·전라남도 및 대한민국",
        "policy_report": None,
        "policy_report_signature": None,
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
    search_state = analysis["search_state"]
    max_top_k = max(1, min(20, len(search_state["texts"])))
    st.session_state.search_top_k = min(
        max(int(st.session_state.get("search_top_k", 5)), 1),
        max_top_k,
    )

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
                min_value=1,
                max_value=max_top_k,
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


def _render_policy_report(report: dict[str, Any]) -> None:
    """Grounding 정책 보고서와 검색 출처를 표시한다."""
    st.divider()
    st.subheader("근거 기반 정책 제안")
    st.markdown(report["markdown"])

    search_widget_html = report.get("search_widget_html", "")
    if search_widget_html:
        st.html(search_widget_html)

    sources = report.get("sources", [])
    if sources:
        with st.expander(f"Google Search 근거 출처 {len(sources)}개"):
            for source in sources:
                st.markdown(
                    f"{source['number']}. [{source['title']}]"
                    f"({source['uri']})"
                )
    else:
        st.warning(
            "응답에 Grounding 출처 메타데이터가 없습니다. 정책명과 "
            "시행 여부를 공식 사이트에서 다시 확인해 주세요."
        )

    queries = report.get("queries", [])
    if queries:
        st.caption("Google Search 검색어: " + " · ".join(queries))


def _render_policy_explorer(analysis: dict[str, Any]) -> None:
    """UMAP 선택 → 숨은 의견 확인 → 정책 근거 검색 흐름을 렌더링한다."""
    st.markdown("### AI 정책 탐색기")
    st.caption(
        "별은 대표 의견, 빈 마름모는 경계 의견, X는 숨은 목소리 "
        "후보입니다. 툴바의 올가미 또는 박스로 관심 영역을 선택하세요."
    )

    event = st.plotly_chart(
        analysis["voice_topic_map"],
        width="stretch",
        key=f"policy_explorer_map_{analysis['analysis_id']}",
        on_select="rerun",
        selection_mode=["points", "box", "lasso"],
        config={"displaylogo": False, "scrollZoom": True},
    )
    selected_row_ids = _extract_selected_row_ids(event)
    selected_df = _get_selected_opinions(analysis, selected_row_ids)

    if selected_df.empty:
        st.info(
            "지도에서 하나 이상의 점을 선택하면 해당 영역의 키워드, "
            "대표 원문과 실제 정책 근거 검색 기능이 열립니다."
        )
        candidates = analysis["opinion_role_df"]
        candidates = candidates[
            candidates["voice_type"].isin(["경계 의견", "숨은 목소리"])
        ].copy()
        if not candidates.empty:
            candidates = candidates.sort_values(
                ["voice_type", "hidden_voice_score"],
                ascending=[True, False],
                kind="stable",
            )
            display_candidates = candidates[
                [
                    "voice_type",
                    "cluster",
                    "cluster_margin",
                    "center_similarity",
                    "local_cohesion",
                    "text",
                ]
            ].head(20).copy()
            for column in (
                "cluster_margin",
                "center_similarity",
                "local_cohesion",
            ):
                display_candidates[column] = display_candidates[column].round(4)
            with st.expander("우선 확인할 숨은·경계 의견 후보 Top 20"):
                st.dataframe(
                    display_candidates,
                    width="stretch",
                    hide_index=True,
                )
        return

    keywords = _extract_keywords_from_texts(
        selected_df["text"].astype(str).tolist(),
        top_n=8,
    )
    metrics = st.columns(4)
    metrics[0].metric("선택 의견", f"{len(selected_df):,}개")
    metrics[1].metric(
        "포함 클러스터",
        f"{selected_df['cluster'].nunique():,}개",
    )
    metrics[2].metric(
        "숨은 목소리",
        int((selected_df["voice_type"] == "숨은 목소리").sum()),
    )
    metrics[3].metric(
        "경계 의견",
        int((selected_df["voice_type"] == "경계 의견").sum()),
    )
    st.markdown(f"**선택 영역 키워드:** {keywords or '추출되지 않음'}")

    representatives = _get_selection_representatives(
        selected_df,
        analysis["embeddings"],
        top_n=3,
    )
    representatives["중심 유사도"] = representatives[
        "중심 유사도"
    ].round(4)
    st.markdown("#### 선택 영역 대표 원문")
    st.dataframe(
        representatives,
        width="stretch",
        hide_index=True,
    )

    with st.expander(f"선택한 원문 전체 {len(selected_df):,}개"):
        display_df = selected_df[
            [
                "row_id",
                "voice_type",
                "cluster",
                "cluster_margin",
                "center_similarity",
                "local_cohesion",
                "text",
            ]
        ].copy()
        for column in (
            "cluster_margin",
            "center_similarity",
            "local_cohesion",
        ):
            display_df[column] = display_df[column].round(4)
        st.dataframe(display_df, width="stretch", hide_index=True)

    st.markdown("#### 실제 정책·지원사업 연결")
    st.text_input(
        "정책 검색 지역·범위",
        key="policy_scope",
        help="예: 광주광역시·전라남도 및 대한민국",
    )
    st.caption(
        "버튼을 누르면 선택 의견 일부가 Gemini API로 전송되고 Google "
        "Search Grounding 검색이 실행됩니다. 검색 쿼리 수에 따라 비용과 "
        "응답 시간이 늘어날 수 있습니다."
    )

    api_key_available = bool(_get_gemini_api_key())
    generate_button = st.button(
        "선택 의견으로 근거 기반 정책 제안 생성",
        type="primary",
        width="stretch",
        disabled=not api_key_available,
        key=f"generate_policy_{analysis['analysis_id']}",
    )
    if not api_key_available:
        st.info(
            "정책 검색을 사용하려면 Streamlit Secrets에 "
            "GEMINI_API_KEY를 설정하세요."
        )

    signature = (
        analysis["analysis_id"],
        tuple(selected_row_ids),
        str(st.session_state.policy_scope).strip(),
    )
    if generate_button:
        try:
            with st.spinner(
                "공식 정책·지원사업을 검색하고 차별화 제안을 작성하고 "
                "있습니다..."
            ):
                st.session_state.policy_report = (
                    _generate_grounded_policy_report(
                        selected_df,
                        keywords,
                        st.session_state.policy_scope,
                    )
                )
                st.session_state.policy_report_signature = signature
        except Exception as error:
            st.error(f"정책 제안 생성에 실패했습니다: {error}")

    if (
        st.session_state.policy_report is not None
        and st.session_state.policy_report_signature == signature
    ):
        _render_policy_report(st.session_state.policy_report)
    elif st.session_state.policy_report is not None:
        st.info(
            "선택 영역 또는 검색 범위가 바뀌었습니다. 현재 선택으로 "
            "정책 제안을 다시 생성해 주세요."
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
            "AI 정책 탐색기",
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
        _render_policy_explorer(analysis)

    with tabs[6]:
        _render_cluster_filter(analysis)

    with tabs[7]:
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
        "대표·경계·숨은 의견, Issue / Root Cause / Action과 실제 "
        "정책 근거를 분석합니다."
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
            "요약과 선택 영역 정책 검색을 위해 Gemini API로 전송됩니다."
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
                    st.session_state.policy_report = None
                    st.session_state.policy_report_signature = None
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
