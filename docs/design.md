# 코드 구조 및 데이터 흐름 설계 (v3: Sequence + PathRAG)

## 1. 코드 구조 (Code Structure)
```
webtoon-analysis-pathrag/
│
├── [app.py](http://app.py/) # Streamlit 웹 애플리케이션 메인 파일
│
├── pyproject.toml # Python 의존성 목록
├── uv.lock # Python 의존성 목록
├── .env # API 키 등 환경 변수 (Git 미포함)
├── .gitignore # Git 추적 제외 목록
│
├── src/ # 소스 코드 루트
│ ├── **init**.py
│ │
│ ├── agents/ # LangGraph 기반 웹툰 분석 에이전트
│ │ ├── **init**.py
│ │ └── webtoon_agent.py # SequenceAgentState, 노드 함수, 그래프 정의
│ │
│ ├── analysis/ # 웹툰 특징 추출 및 PathRAG 분석 관련
│ │ ├── **init**.py
│ │ ├── feature_extractor.py # 시퀀스 분석 결과 기반 특징 추출 로직
│ │ └── pathrag_analyzer.py # PathRAG 쿼리 함수 구현 (추천, 패턴 등)
│ │
│ ├── graph/ # 그래프 데이터 및 PathRAG 연동
│ │ ├── **init**.py
│ │ ├── graph_model.py # NetworkX 기반 그래프 모델 정의
│ │ ├── graph_manager.py # 그래프 생성, 관리, 시각화 로직
│ │ └── pathrag_integration.py# PathRAG 라이브러리 연동 및 인덱싱/쿼리
│ │
│ └── PathRag/ # 벤더링 된 PathRAG 라이브러리
│ │
│ └── utils/ # 유틸리티 함수
│ ├── **init**.py
│ └── data_loader.py # 에피소드 시퀀스 데이터 로딩 유틸리티
│
├── data/ # 데이터 파일
│ ├── images/ # 샘플 웹툰 이미지 (에피소드별 긴 스크롤)
│ ├── metadata/ # 웹툰/에피소드 메타데이터 (JSON)
│ │ ├── webtoons.json
│ │ └── episodes.json
│ └── graph_exports/ # 생성된 그래프 파일(GraphML), 시각화(HTML)
│
├── config/ # 설정 파일 (필요시, 예: LLM 모델명, PathRAG 파라미터)
│
├── tests/ # 테스트 코드
│ ├── **init**.py
│ ├── test_api_connection.py
│ ├── test_data_loader.py
│ ├── test_agent_workflow.py # 시퀀스 분석 Agent 테스트
│ ├── test_feature_extractor.py
│ ├── test_graph_manager.py
│ └── test_pathrag_integration.py # PathRAG 인덱싱 및 기본 쿼리 테스트
│
├── scripts/ # 데이터 처리, 모델 빌드 등 독립 실행 스크립트
│ ├── build_graph_pipeline.py # 전체 그래프 구축 자동화 스크립트
│ └── build_pathrag_index.py # PathRAG 인덱싱 스크립트
│
├── docs/ # 문서 및 설계 파일
│ ├── [design.md](http://design.md/) # 본 설계 문서
│ ├── graph_schema.md # 그래프 스키마 상세 정의 (선택적)
│ └── images/ # README용 이미지, 데모 스크린샷 등
│
└── 
```

## 2. 데이터 흐름 (Data Flow)

1.  **입력:** 사용자가 Streamlit UI 통해 웹툰의 연속된 에피소드 **전체 스크롤 이미지 여러 개** 업로드 (`app.py`).
2.  **시퀀스 분석 (WCAI Agent):** `src.agents.webtoon_agent`의 LangGraph 에이전트가 이미지 리스트 입력받음.
    * `extract_text_and_analyze_node`: 각 이미지별 OCR + 시각 분석 수행 (MLLM 호출).
    * `summarize_sequence_node`: 모든 에피소드의 분석 결과를 종합하여 시퀀스 전체 요약 생성 (LLM 호출).
    * 결과: 에피소드별 분석 결과 리스트, 시퀀스 요약 등.
3.  **(UI표시)** 시퀀스 분석 결과가 Streamlit UI에 표시됨 (`app.py`).
4.  **(백그라운드/배치) 특징 추출:** `src.analysis.feature_extractor`가 시퀀스 분석 결과 + 메타데이터 기반으로 구조화된 특징(태그, 주제, 관계 변화 등) 추출 (LLM 활용).
5.  **(백그라운드/배치) 그래프 구축:** `src.graph.graph_manager`가 메타데이터와 추출된 특징 사용하여 NetworkX 지식 그래프 생성/업데이트 (`data/graph_exports` 저장).
    * 관계(엣지)는 공통 특징 또는 LLM/임베딩 유사도 기반으로 **자동 생성**.
6.  **(백그라운드/배치) PathRAG 인덱싱:** `src.graph.pathrag_integration`이 웹툰 메타데이터 + 시퀀스 분석 결과 + 추출된 특징을 **상세 텍스트로 변환**하여 PathRAG에 인덱싱 (`pathrag_working_dir`).
7.  **PathRAG 쿼리:** 사용자가 Streamlit UI 통해 자연어 쿼리 입력 (`app.py`).
8.  **쿼리 처리:** `src.analysis.pathrag_analyzer`가 쿼리 유형에 맞춰 `src.graph.pathrag_integration`의 PathRAG 쿼리 함수 호출.
9.  **PathRAG 응답:** PathRAG가 인덱싱된 정보와 내부 그래프 기반으로 추론하여 최종 답변 텍스트 생성.
10. **(UI표시)** PathRAG 답변이 Streamlit UI에 표시됨 (`app.py`).
11. **(선택적 UI표시)** 그래프 시각화 결과 표시 (`app.py`).