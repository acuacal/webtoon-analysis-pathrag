# 코드 구조 및 데이터 흐름 설계 (v2: PathRAG 통합)

## 1. 코드 구조 (Code Structure)
    webtoon-analysis-pathrag/
    │
    ├── app.py                  # Streamlit 웹 애플리케이션 메인 파일
    │
    ├── requirements.txt        # Python 의존성 목록
    ├── .env                    # API 키 등 환경 변수 (Git 미포함)
    ├── .gitignore              # Git 추적 제외 목록
    │
    ├── src/                    # 소스 코드 루트
    │   ├── init.py
    │   │
    │   ├── agents/             # LangGraph 기반 웹툰 분석 에이전트
    │   │   ├── init.py
    │   │   └── webtoon_agent.py # Agent 상태, 노드, 그래프 정의
    │   │
    │   ├── analysis/           # 웹툰 특징 추출 관련 모듈
    │   │   ├── init.py
    │   │   └── feature_extractor.py # WCAI Agent 결과 기반 특징 추출 로직
    │   │
    │   ├── graph/              # 그래프 데이터 및 PathRAG 관련 모듈
    │   │   ├── init.py
    │   │   ├── graph_model.py    # NetworkX 기반 그래프 모델 정의
    │   │   ├── graph_manager.py  # 그래프 생성, 관리, 시각화 로직
    │   │   └── pathrag_integration.py # PathRAG 라이브러리 연동 및 쿼리 처리
    │   │
    │   └── utils/              # 유틸리티 함수
    │       ├── init.py
    │       └── data_loader.py    # 샘플 데이터 로딩 유틸리티
    │
    ├── data/                   # 데이터 파일
    │   ├── images/             # 샘플 웹툰 이미지
    │   ├── metadata/           # 웹툰/에피소드 메타데이터 (JSON)
    │   │   ├── webtoons.json
    │   │   └── episodes.json (선택적)
    │   └── graph_exports/      # 생성된 그래프 파일(GraphML 등), 시각화(HTML) 저장
    │
    ├── config/                 # 설정 파일 (필요시)
    │
    ├── tests/                  # 테스트 코드
    │   ├── init.py
    │   ├── test_data_loader.py
    │   ├── test_agent_workflow.py
    │   └── test_pathrag_integration.py # PathRAG 연동 테스트
    │
    ├── scripts/                # 데이터 처리, 모델 빌드 등 스크립트
    │   ├── build_graph_pipeline.py # 전체 그래프 구축 자동화 스크립트
    │   └── build_pathrag_index.py  # PathRAG 인덱싱 스크립트
    │
    ├── docs/                   # 문서 및 설계 파일
    │   ├── design.md           # 본 설계 문서
    │   └── images/             # README용 이미지, 데모 스크린샷 등
    │
    └── pathrag_working_dir/    # PathRAG 라이브러리 작업 디렉토리 (Git 미포함)


## 2. 데이터 흐름 (Data Flow)

1.  **입력 (Input):** 사용자가 Streamlit UI를 통해 웹툰 이미지 파일(들)을 업로드합니다 (`app.py`).
2.  **개별 웹툰 분석 (WCAI Agent):**
    * 업로드된 이미지를 `src.agents.webtoon_agent`의 LangGraph 에이전트로 전달합니다.
    * `extract_text_and_analyze_node`가 MLLM을 호출하여 각 이미지에서 텍스트(OCR)와 시각적 분석 결과를 추출합니다.
    * `summarize_episode_node`가 추출된 텍스트와 시각 분석 결과를 종합하여 LLM으로 최종 에피소드 요약을 생성합니다.
    * 분석 결과 (추출 텍스트, 시각 분석, 요약)는 UI에 즉시 표시될 수 있습니다 (`app.py` -> `display_results`).
3.  **특징 추출 (Feature Extraction):**
    * `src.analysis.feature_extractor`가 WCAI Agent의 분석 결과와 메타데이터(`data/metadata`)를 입력받습니다.
    * LLM 등을 활용하여 정제된 특징(주요 태그, 주제, 분위기, 등장인물 등)을 구조화된 형태로 추출합니다.
4.  **그래프 구축 (Graph Construction):**
    * `scripts.build_graph_pipeline.py`가 실행되거나, 실시간 요청 시 `src.graph.graph_manager`가 호출됩니다.
    * `graph_manager`는 메타데이터와 추출된 특징을 사용하여 `src.graph.graph_model`의 NetworkX 그래프에 노드(웹툰, 작가, 장르, 태그 등)와 엣지(관계: HAS_GENRE, CREATED_BY, SIMILAR_TO 등)를 추가/업데이트합니다.
    * (선택) `graph_manager`를 통해 그래프 시각화 파일(`data/graph_exports`)을 생성합니다.
5.  **PathRAG 인덱싱 (PathRAG Indexing):**
    * `scripts.build_pathrag_index.py`가 실행되거나, 실시간 요청 시 `src.graph.pathrag_integration`의 `WebtoonPathRAG` 클래스가 호출됩니다.
    * 각 웹툰의 메타데이터와 **분석/추출된 특징 정보**를 상세한 **텍스트 설명**으로 변환하여 PathRAG 라이브러리의 `insert` 메서드를 통해 인덱싱합니다.
6.  **쿼리 처리 (Query Processing):**
    * 사용자가 Streamlit UI (`app.py`)를 통해 자연어 쿼리(예: "액션 판타지 웹툰 추천해줘")를 입력합니다.
    * `app.py`는 해당 쿼리를 `src.graph.pathrag_integration`의 쿼리 처리 함수(예: `recommend_webtoons`)로 전달합니다.
    * 쿼리 처리 함수는 PathRAG 라이브러리의 `query` 메서드를 호출합니다. PathRAG는 내부 그래프 탐색 및 LLM 생성을 통해 답변 경로와 최종 답변 텍스트를 생성합니다.
7.  **결과 표시 (Result Display):**
    * PathRAG에서 반환된 답변 텍스트를 `app.py`가 받아 사용자에게 적절한 형태로 표시합니다.
    * (선택) 관련 그래프 시각화 정보를 함께 제공할 수 있습니다.