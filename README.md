# 의견 분석 대시보드

한국어 의견이 담긴 CSV 파일을 업로드하면 문장 임베딩, KMeans 클러스터링,
클러스터별 키워드 및 대표 의견 추출, PCA·UMAP 시각화, 의미 검색을 수행하는
Streamlit 애플리케이션입니다.

Gemini API 키를 설정하면 Gemini 3.5 Flash-Lite가 각 클러스터를
`Issue / Root Cause / Action` 형식으로 자동 요약합니다.

## 주요 기능

- UTF-8-SIG, UTF-8, CP949 CSV 인코딩 지원
- `text` 컬럼의 결측치, 빈 문자열, 앞뒤 공백 및 중복 제거
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 문장 임베딩
- KMeans 클러스터링
- `k=3~10` Silhouette score 계산 및 추천 `k` 표시
- 클러스터별 TF-IDF unigram·bigram 상위 키워드 추출
- 클러스터 중심과 가장 가까운 실제 대표 의견 Top 3 추출
- PCA 및 UMAP 기반 인터랙티브 Plotly Topic Map
- 선택한 클러스터의 실제 의견 필터링
- Exact Keyword Search와 Semantic Search 비교
- Gemini 3.5 Flash-Lite 기반 `Issue / Root Cause / Action` 자동 요약

## 프로젝트 구조

```text
.
├── result.py          # Streamlit 애플리케이션과 전체 분석 함수
├── requirements.txt  # Python 패키지 의존성
└── README.md          # 프로젝트 설명 및 배포 방법
```

## 실행 환경

- Python 3.10 이상
- GPU는 선택 사항이며 CPU에서도 실행 가능
- Gemini 자동 요약을 사용할 경우 Gemini API 키 필요

## 설치 및 로컬 실행

프로젝트 디렉터리에서 다음 명령을 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run result.py
```

Windows PowerShell에서는 가상환경을 다음과 같이 활성화합니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run result.py
```

실행 후 브라우저에서 기본적으로 다음 주소를 엽니다.

```text
http://localhost:8501
```

SentenceTransformer 모델과 UMAP 관련 리소스는 최초 분석 시 다운로드되므로
첫 실행이 이후 실행보다 오래 걸릴 수 있습니다.

## CSV 파일 형식

CSV에는 반드시 `text` 컬럼이 있어야 합니다.

```csv
text
지역기업의 채용 정보를 한곳에서 확인하고 싶어요.
버스 배차간격이 너무 길어서 이동하기 불편합니다.
청년을 위한 취업 지원 프로그램을 쉽게 찾기 어렵습니다.
```

추가 컬럼이 있어도 분석할 수 있지만 문장 임베딩과 클러스터링에는 `text`
컬럼을 사용합니다.

지원하는 인코딩은 다음 순서로 자동 확인합니다.

1. UTF-8-SIG
2. UTF-8
3. CP949

## Gemini API 설정

Gemini 자동 요약에는 안정화 모델 ID인 `gemini-3.5-flash-lite`를 사용합니다.

### 환경변수 사용

Linux 또는 macOS:

```bash
export GEMINI_API_KEY="발급받은_API_KEY"
streamlit run result.py
```

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="발급받은_API_KEY"
streamlit run result.py
```

### Streamlit Secrets 사용

로컬 프로젝트에 `.streamlit/secrets.toml` 파일을 만들고 다음 내용을
추가합니다.

```toml
GEMINI_API_KEY = "발급받은_API_KEY"
```

`secrets.toml`에는 민감한 키가 포함되므로 Git에 커밋하지 마세요.

API 키가 없으면 Gemini 요약만 건너뛰며 임베딩, 클러스터링, 키워드,
Topic Map 및 검색 기능은 정상적으로 사용할 수 있습니다.

## Streamlit Community Cloud 배포

1. `result.py`, `requirements.txt`, `README.md`를 GitHub 저장소에 올립니다.
2. [Streamlit Community Cloud](https://share.streamlit.io/)에서 새 앱을 만듭니다.
3. 저장소와 브랜치를 선택합니다.
4. Main file path를 `result.py`로 설정합니다.
5. App settings의 Secrets에 다음 값을 등록합니다.

```toml
GEMINI_API_KEY = "발급받은_API_KEY"
```

6. Deploy를 실행합니다.

배포 저장소나 코드에 API 키를 직접 작성하지 마세요.

## 사용 방법

1. 왼쪽 사이드바에서 `text` 컬럼이 포함된 CSV를 업로드합니다.
2. 사용할 클러스터 수 `k`를 선택합니다.
3. **분석 실행**을 누릅니다.
4. 분석 완료 후 다음 탭에서 결과를 확인합니다.

| 탭 | 내용 |
| --- | --- |
| 클러스터 요약 | 의견 수, 키워드, 대표 의견, Issue, Root Cause, Action |
| Gemini 요약 | 클러스터별 AI 자동 요약 |
| Silhouette | `k=3~10` 평가 결과와 추천값 |
| PCA Topic Map | 선형 차원 축소 기반 Topic Map |
| UMAP Topic Map | 국소 의미 관계 중심 Topic Map |
| 클러스터 의견 | 선택한 클러스터의 실제 원문 |
| 의견 검색 | Exact Keyword Search와 Semantic Search 비교 |

Silhouette 추천값은 참고용이며 사용자가 입력한 `k`를 자동으로 변경하지
않습니다.

## 분석 처리 순서

```text
CSV 업로드
  → text 정제
  → SentenceTransformer embedding
  → Silhouette score 계산
  → KMeans clustering
  → TF-IDF keywords
  → 대표 의견 추출
  → Gemini Issue / Root Cause / Action 요약
  → PCA·UMAP Topic Map
  → 검색용 session state 생성
```

주요 함수는 다음과 같습니다.

```python
read_and_clean_csv(file_path)
build_analysis(file_path, n_clusters)
semantic_search(query, search_state, top_k=5)
exact_keyword_search(keyword, search_state, top_k=5)
filter_cluster_opinions(cluster_id, search_state)
```

## Gemini 요약 방식과 개인정보 주의사항

- 클러스터당 최대 60개 의견을 표본으로 선택합니다.
- API 입력 크기를 제한하기 위해 클러스터당 최대 약 18,000자를 전송합니다.
- Gemini에는 클러스터 번호, 의견 수, TF-IDF 키워드 및 의견 표본이 전달됩니다.
- Gemini API를 사용하는 경우 업로드된 의견 일부가 Google의 외부 API로
  전송됩니다.
- 개인정보, 연락처, 계정 정보 등 민감한 내용이 포함된 데이터는 업로드 전에
  비식별화하는 것을 권장합니다.

## Topic Map 해석 시 주의사항

- PCA는 전체 데이터의 선형 분산 구조를 보여줍니다.
- UMAP은 가까운 문장들의 국소적인 의미 관계를 강조합니다.
- UMAP 축 자체에는 직접적인 의미가 없습니다.
- 2차원 시각화는 원래 임베딩 정보를 일부 손실하므로 클러스터 품질을
  Topic Map 모양만으로 판단하지 마세요.

## 문제 해결

### `'text' 컬럼이 없습니다` 오류

CSV 컬럼명이 정확히 `text`인지 확인합니다. 컬럼명 앞뒤 공백은 자동으로
제거하지만 `Text`, `content`, `문장` 등 다른 이름은 자동 변환하지 않습니다.

### `n_clusters는 정제 후 문장 수보다 작아야 합니다` 오류

결측치, 빈 문자열과 중복 문장이 제거된 후의 문장 수보다 작은 `k`를
선택하세요.

### Gemini 요약이 생성되지 않는 경우

- `GEMINI_API_KEY`가 정확히 설정되었는지 확인합니다.
- Gemini API 사용 권한과 할당량을 확인합니다.
- Streamlit Community Cloud에서는 앱의 Secrets 메뉴에 키를 등록합니다.
- 변경한 Secrets를 적용하기 위해 앱을 재부팅합니다.

### `PreTrainedConfig` ImportError

기존 환경의 `transformers 5.x`와 패키지가 섞였을 가능성이 있습니다. 새
가상환경을 만들고 `requirements.txt`로 다시 설치하세요.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 첫 분석이 오래 걸리는 경우

최초 실행 시 SentenceTransformer 모델 다운로드와 UMAP의 Numba 컴파일이
진행될 수 있습니다. 이후 실행에서는 Streamlit resource cache를 통해 모델을
재사용합니다.

## 참고 링크

- [Streamlit 문서](https://docs.streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [Gemini 3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
- [UMAP](https://umap-learn.readthedocs.io/)

