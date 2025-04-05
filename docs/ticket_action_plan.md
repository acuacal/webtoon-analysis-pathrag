---

# **Webtoon Sequence Analysis & Recommendation Agent with PathRAG - 상세 실행 계획 (v3 - 최종 통합본)**

## **Phase 1: 환경 설정 및 다중 에피소드 분석 Agent 구현**

### **WCAI-P01: 프로젝트 환경 설정 (PathRAG 포함)** (예상 소요시간: 60분)

- **설명:** PathRAG 라이브러리를 포함하여 개발 환경을 설정하고 기본 구조를 생성합니다.
- **수행 단계:**
    1. **GitHub 저장소 생성 및 클론:**
        - GitHub에서 `webtoon-analysis-pathrag` 저장소 생성.
        - 로컬에 클론: `git clone <https://github.com/[사용자명]/webtoon-analysis-pathrag.git`>
        - 프로젝트 폴더로 이동: `cd webtoon-analysis-pathrag`
    2. **Git 초기화 및 기본 설정:**
        - `.gitignore` 파일 생성 및 내용 추가:
            
            ```
            # Virtual environment
            venv/
            .venv/
            
            # Environment variables
            .env
            
            # Python cache
            __pycache__/
            *.pyc
            *.pyo
            
            # OS generated files
            .DS_Store
            Thumbs.db
            
            # PathRAG working directory
            pathrag_working_dir/
            
            # Data exports (optional)
            data/graph_exports/*.html
            data/graph_exports/*.graphml
            
            ```
            
    3. **가상환경 설정:**
        - `python -m venv venv` (또는 `.venv`)
        - 활성화 (Windows: `venv\\Scripts\\activate`, macOS/Linux: `source venv/bin/activate`)
    4. **PathRAG 저장소 클론 및 설치:**
        
        ```bash
        git clone <https://github.com/BUPT-GAMMA/PathRAG.git>
        cd PathRAG
        # (선택적) 필요시 특정 버전/브랜치 checkout
        pip install -e . # 개발 모드로 설치 (의존성 자동 설치)
        cd ..
        
        ```
        
    5. **requirements.txt 생성/수정:** 프로젝트의 다른 주요 라이브러리를 명시적으로 관리합니다.
        
        ```
        # LangChain Core
        langchain==0.1.11 # 또는 최신 호환 버전
        langchain-openai==0.0.8 # 또는 최신 호환 버전
        langgraph==0.0.26 # 또는 최신 호환 버전
        
        # Web UI & Image Handling
        streamlit==1.32.2 # 또는 최신 버전
        pillow==10.2.0 # 또는 최신 버전
        
        # Environment & Utilities
        python-dotenv==1.0.1 # 또는 최신 버전
        
        # Graph Handling & Visualization
        networkx # 그래프 생성/관리용 (PathRAG 설치 시 포함될 수 있음)
        pyvis # 그래프 시각화용
        
        # PathRAG (설치는 위에서 별도 수행, 의존성 명시 목적)
        # 필요한 경우 PathRAG의 핵심 의존성 추가 확인
        # 예: transformers, torch, faiss-cpu 등 (PathRAG의 setup.py 참고)
        
        ```
        
    6. **라이브러리 설치:** `pip install -r requirements.txt` (PathRAG 설치 시 이미 설치된 것은 건너<0xEB><01>니다.)
    7. **기본 폴더 구조 생성:**
        
        ```bash
        mkdir -p src/agents src/utils src/graph src/analysis data/images data/metadata data/graph_exports config tests docs/images pathrag_working_dir
        touch src/__init__.py src/agents/__init__.py src/utils/__init__.py src/graph/__init__.py src/analysis/__init__.py tests/__init__.py
        
        ```
        
    8. **초기 커밋:**
        
        ```bash
        git add .
        git commit -m "Initial project setup with environment, structure, and PathRAG integration"
        
        ```
        
- **완료 기준:**
    - GitHub 저장소가 로컬에 연결되고 `.gitignore` 파일이 설정됨.
    - 가상환경 활성화 및 PathRAG 포함 필요 라이브러리 설치 완료.
    - 정의된 폴더 구조 생성 완료.
    - 초기 설정 상태가 GitHub에 커밋됨.

### **WCAI-P02: API 키 설정 및 테스트** (예상 소요시간: 15분)

- **설명:** LLM/MLLM API 사용을 위한 키를 설정하고 정상적으로 연결되는지 테스트합니다.
- **수행 단계:**
    1. **.env 파일 생성:** 프로젝트 루트에 `.env` 파일 생성 (`touch .env`).
    2. **API 키 설정:** `.env` 파일에 사용하는 모델(OpenAI, Google 등)의 API 키 추가.
        
        ```
        OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx # 실제 키 입력
        # GOOGLE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx # 필요시
        
        ```
        
        - **주의:** `.env` 파일이 `.gitignore`에 포함되어 있는지 확인합니다.
    3. **API 연결 테스트 스크립트 생성 (`tests/test_api_connection.py`):**
        
        ```python
        import os
        from dotenv import load_dotenv
        from langchain_openai import ChatOpenAI
        # from langchain_google_genai import ChatGoogleGenerativeAI # Gemini 사용 시
        
        print("Loading environment variables from .env file...")
        load_dotenv()
        
        print("--- Testing OpenAI API Connection ---")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not api_key.startswith("sk-"): # 기본 형식 체크
            print(f"RESULT: FAIL - OPENAI_API_KEY not found or invalid in .env file (Value: {api_key})")
        else:
            print("OpenAI API Key found.")
            try:
                # 간단한 호출로 테스트 (비용 적고 빠른 모델 권장)
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, request_timeout=30)
                response = llm.invoke("Ping: Reply with just 'Pong'.")
                if "pong" in response.content.lower():
                     print("OpenAI API Test Response:", response.content)
                     print("RESULT: SUCCESS - OpenAI API connection successful!")
                else:
                     print(f"RESULT: UNEXPECTED RESPONSE - API connected but response was: {response.content}")
        
            except Exception as e:
                print(f"RESULT: FAIL - Error connecting to OpenAI API: {e}")
        
        # 필요한 경우 다른 API(Google 등) 테스트 로직 추가
        # ...
        
        ```
        
    4. **테스트 실행:** 프로젝트 루트에서 `python tests/test_api_connection.py` 실행.
- **완료 기준:**
    - `.env` 파일에 유효한 API 키가 설정됨.
    - 테스트 스크립트 실행 시 사용하는 API에 대해 "RESULT: SUCCESS" 메시지 출력.

### **WCAI-P03: 프로젝트 범위 정의 및 구조 설계 (다중 에피소드 + PathRAG)** (예상 소요시간: 60분)

- **설명:** 최소 3개 연속 에피소드 분석 및 PathRAG를 이용한 관계 분석/추천 기능을 포함한 전체 프로젝트 범위와 아키텍처를 정의합니다.
- **수행 단계:**
    1. [**README.md](http://readme.md/) 작성/업데이트:** 프로젝트 루트에 `README.md` 파일 생성 또는 업데이트. 아래 내용 포함.
        
        ```markdown
        # Webtoon Sequence Analysis & Recommendation Agent with PathRAG (PoC)
        
        ## 🚀 프로젝트 개요
        
        이 프로젝트는 웹툰 콘텐츠(이미지+텍스트)를 자동으로 이해하고 분석하는 AI 에이전트의 Proof-of-Concept(PoC)입니다. LangGraph 기반 에이전트를 사용하여 개별 웹툰의 **연속된 에피소드 시퀀스(최소 3회차)**의 내용을 심층 분석하고, **PathRAG 기술을 통합하여 웹툰 간의 복잡한 관계성을 분석하며 개인화된 추천 기능을 제공**하는 것을 목표로 합니다.
        
        ## 🎯 PoC 범위
        
        * **입력:** 사용자가 업로드한 특정 웹툰의 **연속된 에피소드 전체 스크롤 이미지 파일 여러 개**.
        * **처리 1 (WCAI Agent - Sequence Analysis):**
            * 각 에피소드 이미지에서 MLLM(GPT-4o 등)을 이용해 텍스트 추출(OCR) 및 시각적 내용 동시 분석.
            * 추출된 텍스트와 분석 결과를 종합하여 LLM(GPT-4o-mini 등)으로 **시퀀스 전체 요약** 생성 (LangGraph 워크플로우).
        * **처리 2 (Graph Construction):**
            * 에이전트 분석 결과 및 메타데이터 기반 **특징 추출** (주제, 태그, 캐릭터 관계 변화 등).
            * 웹툰 지식 그래프 자동 구축 (NetworkX): 웹툰, 작가, 장르, 태그, 특징 등을 노드로, 관계를 엣지로 표현.
        * **처리 3 (PathRAG Integration):**
            * 웹툰 메타데이터 + 시퀀스 분석 결과 + 추출된 특징을 상세 텍스트로 변환하여 PathRAG 인덱싱.
            * PathRAG를 활용한 자연어 쿼리 기반 **관계 분석** 및 **추천** 수행.
        * **출력:** 개별 에피소드 분석 결과, 시퀀스 전체 요약, PathRAG 쿼리 결과(텍스트), 그래프 시각화(HTML).
        * **UI:** Streamlit 기반 웹 인터페이스 (다중 이미지 업로드, 샘플 시퀀스 선택, 분석 결과 표시, PathRAG 쿼리 입력/결과 표시, 그래프 시각화 연동).
        
        ## ✨ 주요 기능 목록
        
        * 웹툰 전체 스크롤 이미지 기반 자동 텍스트 추출(OCR) 및 시각 분석.
        * **연속된 에피소드 시퀀스**에 대한 통합 요약 생성.
        * 시퀀스 분석 기반 **웹툰 특징 자동 추출** (주제, 태그, 캐릭터 관계 등).
        * 웹툰 지식 그래프 자동 구축 및 시각화.
        * PathRAG를 이용한 문맥 기반 웹툰 정보 검색 및 관계 추론.
        * **다각적 유사 웹툰 추천** (내용, 스타일, 장르, 관계 등 기반).
        * 웹툰 트렌드 및 **인기 패턴 분석** 지원 (PathRAG 쿼리 활용).
        * **크리에이터 전략 분석** 지원 (PathRAG 쿼리 활용).
        
        ## 🛠️ 기술 스택
        
        * **AI Workflow:** LangChain & LangGraph
        * **Core Models:** OpenAI GPT-4o / GPT-4o-mini (또는 Gemini Pro Vision/1.5 Pro)
        * **Graph RAG:** **PathRAG (BUPT-GAMMA)**
        * **Graph Handling:** **NetworkX**
        * **Web Interface:** Streamlit
        * **Development:** Python
        * **Graph Visualization:** PyVis
        
        ---
        *(README의 나머지 부분: 설치 방법, 사용 방법, 프로젝트 구조 등은 이후 단계에서 구체화됩니다.)*
        
        ```
        
    2. **코드 구조 설계 문서 작성/업데이트 (`docs/design.md`):**
    webtoon-analysis-pathrag/
    │
    ├── [app.py](http://app.py/) # Streamlit 웹 애플리케이션 메인 파일
    │
    ├── requirements.txt # Python 의존성 목록
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
    │ └── utils/ # 유틸리티 함수
    │ ├── **init**.py
    │ └── data_loader.py # 에피소드 시퀀스 데이터 로딩 유틸리티
    │
    ├── data/ # 데이터 파일
    │ ├── images/ # 샘플 웹툰 이미지 (에피소드별 긴 스크롤)
    │ ├── metadata/ # 웹툰/에피소드 메타데이터 (JSON)
    │ │ ├── webtoons.json
    │ │ └── episodes.json (선택적)
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
    └── pathrag_working_dir/ # PathRAG 라이브러리 작업 디렉토리 (Git 미포함)
        
        ```markdown
        # 코드 구조 및 데이터 흐름 설계 (v3: Sequence + PathRAG)
        
        ## 1. 코드 구조 (Code Structure)
        
        ```
        
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
        
        ```
        
- **완료 기준:**
    - 확장된 범위와 구조가 반영된 README 초안 완료.
    - 상세화된 코드 구조 및 데이터 흐름 설계 문서 초안 완료.

### **WCAI-P04: 샘플 웹툰 데이터 확보 (다중 에피소드 + 메타데이터 강화)** (예상 소요시간: 120분+)

- **설명:** 분석할 **최소 10개 웹툰**에 대해, 각 웹툰당 **최소 3개 연속 회차의 에피소드 전체 스크롤 이미지**와 상세 메타데이터를 준비합니다.
- **수행 단계:**
    1. **10~15개 웹툰 선정:** 다양한 장르/스타일/플랫폼 고려. (저작권 유의, 필요시 CC 라이선스 작품 활용)
    2. **연속 에피소드 선정:** 각 웹툰별로 내용 흐름상 의미 있는 **최소 3개 연속 에피소드** 선정 (예: 시즌 초반부, 특정 사건 구간).
    3. **전체 스크롤 이미지 확보:** 선정된 각 에피소드의 **전체 스크롤**을 **하나의 긴 이미지 파일(PNG 또는 JPG)**로 **수동 캡처/저장**. 파일명 규칙 정의 (예: `w001_ep1_full.png`). -> `data/images/`에 저장. (자동 다운로드 절대 금지, 저작권 및 약관 준수)
    4. **상세 웹툰 메타데이터 파일 작성 (`data/metadata/webtoons.json`):** 각 웹툰별 상세 정보 포함.
        
        ```json
        [
          {
            "webtoon_id": "w001",
            "title": "나 혼자만 레벨업", // 예시
            "genre": ["판타지", "액션", "헌터물"],
            "tags": ["먼치킨", "성장형 주인공", "게임 시스템", "현대 판타지"],
            "art_style": "실사체",
            "update_frequency": "완결", // 예: 주1회, 완결
            "platform": "카카오웹툰", // 예: 네이버웹툰, 카카오웹툰
            "creator": "추공 (원작), 장성락 (그림)",
            "popularity_score": 9.8, // 예: 10점 만점
            "is_romance": false,
            "description": "E급 헌터 성진우, 죽음의 위기에서 시스템을 통해 레벨업 능력을 각성하다...", // 간단한 소개글
            "analysis_episode_sequence": ["w001_ep1", "w001_ep2", "w001_ep3"] // 분석 대상 시퀀스 ID 리스트
          },
          {
            "webtoon_id": "w002",
            "title": "유미의 세포들", // 예시
            "genre": ["로맨스", "일상", "드라마", "코미디"],
            "tags": ["세포", "연애", "성장", "직장인"],
            "art_style": "캐주얼",
            "update_frequency": "완결",
            "platform": "네이버웹툰",
            "creator": "이동건",
            "popularity_score": 9.5,
            "is_romance": true,
            "description": "30대 평범한 직장인 유미의 연애와 일상을 머릿속 세포들의 시각으로 그린 이야기.",
            "analysis_episode_sequence": ["w002_ep100", "w002_ep101", "w002_ep102"] // 특정 구간 시퀀스
          }
          // ... 최소 8~18개 웹툰 정보 추가 ...
        ]
        
        ```
        
    5. **(선택적) 에피소드 메타데이터 파일 작성 (`data/metadata/episodes.json`):** 에피소드 ID, 웹툰 ID, 제목, 이미지 파일명 등 관리.
        
        ```json
        [
          {"episode_id": "w001_ep1", "webtoon_id": "w001", "title": "1화 프롤로그", "image_file": "w001_ep1_full.png"},
          {"episode_id": "w001_ep2", "webtoon_id": "w001", "title": "2화 새로운 시작", "image_file": "w001_ep2_full.png"},
          {"episode_id": "w001_ep3", "webtoon_id": "w001", "title": "3화 던전 입장", "image_file": "w001_ep3_full.png"},
          // ... 다른 에피소드 정보 ...
        ]
        
        ```
        
- **완료 기준:**
    - 최소 10개 웹툰의 연속 3회차 에피소드 전체 스크롤 이미지 준비 완료 (`data/images/`).
    - 상세 웹툰 메타데이터 (`webtoons.json`) 준비 완료.
    - (선택적) 에피소드 메타데이터 (`episodes.json`) 준비 완료.

### **WCAI-P05: 데이터 로딩 함수 구현 (에피소드 시퀀스)** (예상 소요시간: 45분)

- **설명:** 웹툰 ID와 에피소드 시퀀스 정보를 기반으로 해당 연속 에피소드들의 전체 스크롤 이미지 리스트와 관련 메타데이터를 로드하는 유틸리티를 구현합니다.
- **수행 단계:**
    1. **`WebtoonDataLoader` 클래스 구현/수정 (`src/utils/data_loader.py`):**
        
        ```python
        import os
        import json
        from PIL import Image
        from typing import Dict, List, Optional, Tuple
        
        class WebtoonDataLoader:
            def __init__(self, data_dir: str = "data"):
                self.data_dir = data_dir
                self.images_dir = os.path.join(data_dir, "images")
                self.metadata_dir = os.path.join(data_dir, "metadata")
                self.webtoons_meta_path = os.path.join(self.metadata_dir, "webtoons.json")
                self.episodes_meta_path = os.path.join(self.metadata_dir, "episodes.json")
                self.webtoon_metadata = self._load_json(self.webtoons_meta_path) or []
                self.episode_metadata = self._load_json(self.episodes_meta_path) or []
                # 에피소드 ID로 메타데이터 빠르게 찾기 위한 딕셔너리 (선택적 최적화)
                self.episode_map = {ep['episode_id']: ep for ep in self.episode_metadata}
        
            def _load_json(self, file_path: str) -> Optional[List[Dict]]:
                """JSON 파일 로드 유틸리티"""
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from {file_path}")
                else:
                    print(f"Warning: Metadata file not found at {file_path}")
                return None
        
            def get_all_webtoon_ids(self) -> List[str]:
                """분석 가능한 웹툰 ID 목록 반환"""
                return [wt['webtoon_id'] for wt in self.webtoon_metadata]
        
            def get_webtoon_metadata(self, webtoon_id: str) -> Optional[Dict]:
                """특정 웹툰의 메타데이터 반환"""
                for wt in self.webtoon_metadata:
                    if wt['webtoon_id'] == webtoon_id:
                        return wt
                return None
        
            def get_episode_sequence_info(self, webtoon_id: str) -> Optional[Tuple[List[str], List[str]]]:
                """웹툰 메타데이터에서 분석 대상 에피소드 ID 및 제목 리스트 반환"""
                webtoon_meta = self.get_webtoon_metadata(webtoon_id)
                if webtoon_meta:
                    episode_ids = webtoon_meta.get("analysis_episode_sequence", [])
                    episode_titles = []
                    for ep_id in episode_ids:
                        ep_meta = self.episode_map.get(ep_id)
                        episode_titles.append(ep_meta.get("title", "Unknown Title") if ep_meta else "Unknown Title")
                    return episode_ids, episode_titles
                return None
        
            def load_episode_sequence_images(self, webtoon_id: str, episode_ids: List[str]) -> List[Image.Image]:
                """주어진 에피소드 ID 목록에 해당하는 이미지 파일 리스트 로드"""
                images = []
                for ep_id in episode_ids:
                    ep_meta = self.episode_map.get(ep_id)
                    if ep_meta and 'image_file' in ep_meta:
                        img_path = os.path.join(self.images_dir, ep_meta['image_file'])
                        if os.path.exists(img_path):
                            try:
                                img = Image.open(img_path)
                                # RGBA -> RGB 변환 (일부 모델 호환성)
                                if img.mode == 'RGBA':
                                    img = img.convert('RGB')
                                images.append(img)
                                print(f"Loaded image: {img_path}")
                            except Exception as e:
                                print(f"Warning: Failed to load image {img_path}. Error: {e}")
                        else:
                            print(f"Warning: Image file not found: {img_path}")
                    else:
                        print(f"Warning: Metadata or image file not defined for episode {ep_id}")
                return images
        
        ```
        
    2. **테스트 스크립트 작성/수정 (`tests/test_data_loader.py`):**
        
        ```python
        import sys, os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils.data_loader import WebtoonDataLoader
        
        print("--- DataLoader Test ---")
        loader = WebtoonDataLoader()
        
        webtoon_ids = loader.get_all_webtoon_ids()
        print(f"Available Webtoon IDs: {webtoon_ids}")
        
        if webtoon_ids:
            test_webtoon_id = webtoon_ids[0]
            print(f"\\nTesting with Webtoon ID: {test_webtoon_id}")
        
            webtoon_meta = loader.get_webtoon_metadata(test_webtoon_id)
            print(f"Webtoon Metadata: {webtoon_meta}")
        
            sequence_info = loader.get_episode_sequence_info(test_webtoon_id)
            if sequence_info:
                episode_ids, episode_titles = sequence_info
                print(f"Episode Sequence IDs: {episode_ids}")
                print(f"Episode Sequence Titles: {episode_titles}")
        
                images = loader.load_episode_sequence_images(test_webtoon_id, episode_ids)
                print(f"Loaded {len(images)} images for the sequence.")
                if images:
                    print(f"First image size: {images[0].size}, mode: {images[0].mode}")
                print("RESULT: SUCCESS - Data loading functions seem operational.")
            else:
                print("RESULT: FAIL - Could not retrieve episode sequence info.")
        else:
            print("RESULT: FAIL - No webtoons found in metadata.")
        
        ```
        
    3. **테스트 실행:** 프로젝트 루트에서 `python tests/test_data_loader.py` 실행.
- **완료 기준:**
    - 에피소드 시퀀스의 이미지 리스트 및 관련 정보 로딩 함수 구현 완료.
    - 테스트 스크립트로 정상 동작 확인.

### **WCAI-P06: LangGraph 상태 정의 (다중 에피소드)** (예상 소요시간: 30분)

- **설명:** 여러 개의 연속 에피소드 분석 결과를 처리할 수 있도록 LangGraph 상태(State)를 정의합니다.
- **수행 단계:**
    1. **`WebtoonSequenceAgentState` 정의 (`src/agents/webtoon_agent.py`):**
        
        ```python
        from typing import TypedDict, List, Optional, Dict, Any
        from PIL import Image
        
        class WebtoonSequenceAgentState(TypedDict):
            """
            웹툰 시퀀스 분석 에이전트의 상태 정의 (v3)
            각 필드는 분석 파이프라인을 통해 채워지거나 업데이트됩니다.
            """
            # === 입력 데이터 ===
            webtoon_id: str                  # 분석 대상 웹툰의 고유 ID
            episode_ids: List[str]           # 분석할 에피소드 ID 목록 (순서 중요)
            episode_titles: List[str]        # 에피소드 제목 목록 (입력 IDs와 순서 일치)
            images: List[Image.Image]        # 에피소드별 전체 스크롤 이미지 객체 리스트
        
            # === 중간/최종 처리 결과 ===
            # 에피소드별 분석 결과
            per_episode_extracted_text: List[str]  # 각 에피소드 이미지에서 추출된 텍스트 리스트
            per_episode_visual_analysis: List[str] # 각 에피소드 이미지의 시각적 분석 결과 리스트
        
            # 시퀀스 전체 종합 결과
            sequence_combined_text: str        # 모든 에피소드에서 추출된 텍스트를 순서대로 결합한 문자열
            sequence_combined_analysis: str    # 모든 에피소드의 시각적 분석 결과를 순서대로 결합한 문자열
            sequence_summary: str              # 시퀀스 전체 내용을 요약한 최종 결과 문자열
        
            # (선택적) 추가 분석 결과
            extracted_features: Optional[Dict[str, Any]] # G02에서 추출된 특징 (태그, 주제 등)
        
            # === 오류 처리 ===
            error: Optional[str]             # 파이프라인 실행 중 발생한 오류 메시지
        
        ```
        
- **완료 기준:** 다중 에피소드 처리를 위한 `WebtoonSequenceAgentState` 최종 정의 완료. 각 필드 역할 주석 명시.

### **WCAI-P07: OCR 및 이미지 분석 노드 (다중 이미지 처리)** (예상 소요시간: 120분)

- **설명:** 입력된 여러 개의 긴 에피소드 이미지 각각에 대해 MLLM으로 OCR 및 시각적 분석을 수행하고, 결과를 상태에 리스트 형태로 저장하는 노드를 구현합니다.
- **수행 단계:**
    1. **필요 모듈 import 및 유틸리티 함수 준비 (`src/agents/webtoon_agent.py`):**
        
        ```python
        import base64
        import io
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from PIL import Image
        import re
        from typing import Dict, List, Tuple, Optional
        
        # 이미지 인코딩 유틸리티
        def encode_image(image: Image.Image, format="JPEG") -> str:
            buffer = io.BytesIO()
            try:
                image.save(buffer, format=format)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"Error encoding image: {e}")
                return ""
        
        # 응답 파싱 유틸리티 (개선 필요)
        def parse_mllm_response(response_content: str) -> Tuple[str, str]:
            """MLLM 응답에서 텍스트와 분석 부분을 분리 (개선된 버전)"""
            # 마커 기반 파싱을 더 명확하게 시도
            text_marker = "추출된 텍스트:"
            analysis_marker = "시각적 분석:"
        
            text_start = response_content.find(text_marker)
            analysis_start = response_content.find(analysis_marker)
        
            extracted_text = "텍스트 추출 실패"
            visual_analysis = "시각적 분석 실패"
        
            if text_start != -1:
                text_content_start = text_start + len(text_marker)
                if analysis_start != -1 and analysis_start > text_start:
                    extracted_text = response_content[text_content_start:analysis_start].strip()
                else: # 시각 분석 마커가 없거나 텍스트 뒤에 없으면 끝까지
                    extracted_text = response_content[text_content_start:].strip()
        
            if analysis_start != -1:
                analysis_content_start = analysis_start + len(analysis_marker)
                # 텍스트 분석 마커가 뒤에 오는지 체크 (순서 보장 안될 시)
                # if text_start != -1 and text_start > analysis_start:
                #     visual_analysis = response_content[analysis_content_start:text_start].strip()
                # else:
                visual_analysis = response_content[analysis_content_start:].strip()
        
            # 간단한 후처리 (빈 줄 제거 등)
            extracted_text = "\\n".join([line for line in extracted_text.splitlines() if line.strip()])
            visual_analysis = "\\n".join([line for line in visual_analysis.splitlines() if line.strip()])
        
            return extracted_text, visual_analysis
        
        ```
        
    2. **`extract_text_and_analyze_node` 함수 구현 (`src/agents/webtoon_agent.py`):**
        
        ```python
        # ... (imports, WebtoonSequenceAgentState) ...
        
        def extract_text_and_analyze_node(state: WebtoonSequenceAgentState) -> Dict:
            """(Node) 이미지 리스트에서 OCR 및 시각적 분석 동시 수행"""
            print(f"--- Node: extract_text_and_analyze ({len(state.get('images',[]))} images) ---")
            images = state.get("images", [])
            if not images:
                return {"error": "분석할 이미지가 없습니다."}
        
            per_episode_texts = []
            per_episode_analyses = []
            errors = []
        
            try:
                # 모델 초기화 (환경 변수 등에서 모델명 가져오도록 개선 가능)
                # max_tokens를 충분히 크게 설정 (긴 이미지 분석 + OCR 결과)
                model = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=2048, request_timeout=120)
        
                analysis_requests = []
                for i, img in enumerate(images):
                    base64_image = encode_image(img)
                    if not base64_image:
                        errors.append(f"에피소드 {i+1}: 이미지 인코딩 실패")
                        analysis_requests.append(None) # 실패한 요청 표시
                        continue
        
                    # 프롬프트 정의 (WCAI-P07 이전 답변 내용 참고)
                    prompt = """이 웹툰 이미지 장면을 분석해주세요. 다음 두 가지 작업을 수행하고 명확히 구분해서 응답해주세요:
        
        ```
        
1. **추출된 텍스트:** 이미지 안에 보이는 모든 텍스트(말풍선 안 대사, 효과음, 배경글자 등)를 정확히 추출해주세요. 누락 없이 최대한 모든 텍스트를 포함해주세요.
2. **시각적 분석:** 이미지의 시각적 내용(주요 등장인물, 인물의 표정과 행동, 배경 장소, 전체적인 분위기, 중요한 시각적 단서나 효과)을 상세히 설명해주세요.

응답 형식:
추출된 텍스트:
[여기에 추출된 텍스트 내용]

시각적 분석:
[여기에 시각적 분석 내용]
"""
prompt_content = [
{"type": "text", "text": prompt},
{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
]
analysis_requests.append(HumanMessage(content=prompt_content))

```
                # MLLM 호출 (Batch 처리 시도, 실패 시 순차 처리)
                # 주의: LangChain의 batch가 모든 모델/환경에서 안정적이지 않을 수 있음
                valid_requests = [req for req in analysis_requests if req is not None]
                if valid_requests:
                    try:
                        print(f"Requesting analysis for {len(valid_requests)} images using batch...")
                        results = model.batch(valid_requests, config={"max_concurrency": 5}) # 동시성 제한
                    except Exception as batch_err:
                        print(f"Batch processing failed ({batch_err}), falling back to sequential processing...")
                        results = []
                        for req in valid_requests:
                            try:
                                results.append(model.invoke(req))
                            except Exception as invoke_err:
                                print(f"Sequential invoke failed for one image: {invoke_err}")
                                # 실패한 경우 에러 메시지가 담긴 content 객체 생성
                                from langchain_core.messages import AIMessage
                                results.append(AIMessage(content=f"Error: {invoke_err}"))

                    # 결과 파싱 및 저장
                    result_idx = 0
                    for i in range(len(images)):
                        if analysis_requests[i] is None: # 인코딩 실패 케이스
                            per_episode_texts.append("[오류: 이미지 로드/인코딩 실패]")
                            per_episode_analyses.append("[오류: 이미지 로드/인코딩 실패]")
                        elif result_idx < len(results):
                            response = results[result_idx]
                            extracted, visual = parse_mllm_response(response.content)
                            per_episode_texts.append(extracted)
                            per_episode_analyses.append(visual)
                            print(f"  - Image {i+1} analysis parsed.")
                            result_idx += 1
                        else: # API 호출 실패 케이스 등
                            per_episode_texts.append("[오류: API 호출 실패]")
                            per_episode_analyses.append("[오류: API 호출 실패]")
                            errors.append(f"에피소드 {i+1}: API 호출 실패")

                else: # 모든 이미지 인코딩 실패
                     errors.append("모든 이미지 인코딩 실패")
                     per_episode_texts = ["[오류]"] * len(images)
                     per_episode_analyses = ["[오류]"] * len(images)

            except Exception as e:
                print(f"OCR/이미지 분석 노드 전체 오류: {e}")
                errors.append(f"전체 분석 노드 오류: {str(e)}")
                # 모든 결과에 오류 반영
                per_episode_texts = ["[오류]"] * len(images)
                per_episode_analyses = ["[오류]"] * len(images)

        # 상태 업데이트
        final_error = "; ".join(errors) if errors else None
        combined_text = "\\n\\n".join([f"--- 에피소드 {i+1} 텍스트 ---\\n{text}" for i, text in enumerate(per_episode_texts)])
        combined_analysis = "\\n\\n".join([f"--- 에피소드 {i+1} 시각 분석 ---\\n{analysis}" for i, analysis in enumerate(per_episode_analyses)])

        print(f"--- Node: extract_text_and_analyze completed. Errors: {final_error} ---")
        return {
            "per_episode_extracted_text": per_episode_texts,
            "per_episode_visual_analysis": per_episode_analyses,
            "sequence_combined_text": combined_text,
            "sequence_combined_analysis": combined_analysis,
            "error": final_error
        }
    ```
* **완료 기준:**
    * 다중 이미지 입력 처리, 각 이미지별 OCR/분석 수행 노드 구현 완료.
    * 안정적인 응답 파싱 로직 구현 및 테스트.
    * 결과를 리스트 및 종합 텍스트로 상태에 저장. 오류 처리 포함.

```

### **WCAI-P08: 시퀀스 요약 노드 개발 (LLM)** (예상 소요시간: 75분)

- **설명:** 여러 에피소드에 걸친 추출된 텍스트와 시각 분석 결과를 종합하여 시퀀스 전체의 줄거리, 캐릭터 변화 등을 요약하는 LLM 기반 노드를 구현합니다.
- **수행 단계:**
    1. **`summarize_sequence_node` 함수 구현 (`src/agents/webtoon_agent.py`):**
        
        ```python
        # ... (imports, WebtoonSequenceAgentState) ...
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        
        def summarize_sequence_node(state: WebtoonSequenceAgentState) -> Dict:
            """(Node) LLM을 사용하여 시퀀스 전체 내용을 요약"""
            print(f"--- Node: summarize_sequence ---")
            if state.get("error"):
                print("Skipping summarization due to previous error.")
                return {} # 이전 오류 유지
        
            combined_text = state.get("sequence_combined_text", "")
            combined_analysis = state.get("sequence_combined_analysis", "")
            titles = state.get("episode_titles", [])
            webtoon_title = state.get("episode_title","") # P09에서 webtoon title도 넣어주면 좋음
        
            if not combined_text and not combined_analysis:
                return {"sequence_summary": "요약할 내용 없음", "error": "No content to summarize"}
        
            summary = "요약 생성 실패"
            error_message = None
        
            try:
                # 요약용 LLM (비용 효율적 모델 선택 가능)
                model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1500, request_timeout=90)
        
                # 시스템 메시지 (역할 부여)
                system_prompt = "당신은 웹툰의 여러 에피소드에 걸친 내용을 분석하고 핵심 줄거리를 명확하게 요약하는 전문 웹툰 분석가입니다."
        
                # 사용자 메시지 (상세한 지침)
                user_prompt = f"""다음은 웹툰 '{webtoon_title}'의 연속된 {len(titles)}개 에피소드({', '.join(titles)}) 분량에 대한 분석 결과입니다.
        
                [이미지에서 추출된 전체 텍스트 요약]
                {combined_text[:3000]}... (너무 길 경우 일부만 제공)
        
                [이미지 시각적 분석 요약]
                {combined_analysis[:3000]}... (너무 길 경우 일부만 제공)
        
                위 내용을 바탕으로, 이 **연속된 에피소드 시퀀스 전체**에 걸쳐 일어난 **주요 사건의 흐름(기승전결), 핵심 등장인물의 행동/감정 변화나 관계 발전, 중요한 복선이나 드러난 주제** 등을 포함하여 **하나의 통합된 요약문(5~7문장 내외)**을 작성해주세요.
                OCR 오류 가능성을 감안하여 내용을 종합적으로 판단하고, 결과는 서술형으로 작성해주세요.
                """
        
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
        
                response = model.invoke(messages)
                summary = response.content
                print("Sequence summary generated.")
        
            except Exception as e:
                print(f"Error during sequence summarization: {e}")
                error_message = f"Sequence summarization failed: {str(e)}"
        
            print(f"--- Node: summarize_sequence completed. Errors: {error_message} ---")
            return {
                "sequence_summary": summary,
                "error": error_message or state.get("error") # 기존 오류 유지 또는 새 오류 업데이트
            }
        
        ```
        
- **완료 기준:** 시퀀스 전체 내용을 입력받아 통합 요약을 생성하는 LLM 노드 구현 완료. 효과적인 프롬프트 작성.

### **WCAI-P09: 워크플로우 통합 및 테스트 (다중 에피소드)** (예상 소요시간: 75분)

- **설명:** 개발된 노드들을 LangGraph에 연결하고, 에피소드 시퀀스를 입력하여 전체 워크플로우가 동작하는지 프로그래밍 방식으로 테스트합니다.
- **수행 단계:**
    1. **`create_webtoon_agent_graph` 함수 구현 완료 (`src/agents/webtoon_agent.py`):**
        
        ```python
        from langgraph.graph import StateGraph, END
        # ... (import WebtoonSequenceAgentState, node functions) ...
        
        def create_webtoon_agent_graph() -> StateGraph:
            """웹툰 시퀀스 분석 LangGraph 생성 및 설정 (v3)"""
            graph_builder = StateGraph(WebtoonSequenceAgentState)
        
            # 노드 추가
            graph_builder.add_node("extract_and_analyze", extract_text_and_analyze_node)
            graph_builder.add_node("summarize_sequence", summarize_sequence_node) # 노드 이름 변경됨
        
            # 엣지 정의
            graph_builder.set_entry_point("extract_and_analyze")
            graph_builder.add_edge("extract_and_analyze", "summarize_sequence")
            graph_builder.add_edge("summarize_sequence", END)
        
            return graph_builder # 컴파일 전 빌더 반환
        
        ```
        
    2. **에이전트 실행 함수 구현 (`src/agents/webtoon_agent.py`의 `analyze_webtoon_sequence`):**
        
        ```python
        from src.utils.data_loader import WebtoonDataLoader
        from langgraph.checkpoint.memory import MemorySaver # 메모리 내 체크포인트 (선택적)
        
        # 그래프 빌더 생성 (모듈 로드 시)
        graph_builder = create_webtoon_agent_graph()
        
        def analyze_webtoon_sequence(webtoon_id: str, episode_ids: List[str]) -> Dict:
            """에피소드 시퀀스를 분석하는 메인 함수"""
            print(f"--- Analyzing sequence for {webtoon_id}: {episode_ids} ---")
            loader = WebtoonDataLoader()
            webtoon_meta = loader.get_webtoon_metadata(webtoon_id)
            sequence_info = loader.get_episode_sequence_info(webtoon_id) # 제목 가져오기 위해
        
            if not webtoon_meta or not sequence_info:
                return {"error": f"Metadata not found for {webtoon_id}"}
        
            # 시퀀스 ID와 제목 추출 (요청된 ID 기준)
            req_episode_titles = [title for ep_id, title in zip(sequence_info[0], sequence_info[1]) if ep_id in episode_ids]
        
            # 이미지 로드
            images = loader.load_episode_sequence_images(webtoon_id, episode_ids)
            if len(images) != len(episode_ids):
                # 이미지 로드 실패 시 오류 처리 강화
                return {"error": f"Failed to load all images for sequence: {episode_ids}"}
        
            # 초기 상태 구성
            initial_state = WebtoonSequenceAgentState(
                webtoon_id=webtoon_id,
                episode_ids=episode_ids,
                episode_titles=req_episode_titles,
                images=images,
                per_episode_extracted_text=[], # 초기화
                per_episode_visual_analysis=[], # 초기화
                sequence_combined_text="",
                sequence_combined_analysis="",
                sequence_summary="",
                extracted_features=None, # 초기화
                error=None
            )
        
            try:
                # 메모리 내 체크포인터 사용 (선택적, 디버깅/재시작에 유용)
                memory = MemorySaver()
                webtoon_agent = graph_builder.compile(checkpointer=memory)
        
                # config에 스레드 ID 등 추가 가능
                config = {"configurable": {"thread_id": f"{webtoon_id}-{'_'.join(episode_ids)}"}}
                final_state = webtoon_agent.invoke(initial_state, config=config)
        
                print(f"--- Sequence analysis completed for {webtoon_id} ---")
                return final_state
        
            except Exception as e:
                print(f"Agent execution failed for {webtoon_id}: {e}")
                initial_state['error'] = f"Agent execution error: {str(e)}"
                return initial_state
        
        ```
        
    3. **테스트 스크립트 작성/실행 (`tests/test_agent_workflow.py`):**
        
        ```python
        import sys, os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from src.agents.webtoon_agent import analyze_webtoon_sequence
        from src.utils.data_loader import WebtoonDataLoader
        from dotenv import load_dotenv
        
        load_dotenv() # API 키 로드
        
        print("--- Agent Workflow Test (v3: Sequence Analysis) ---")
        loader = WebtoonDataLoader()
        webtoon_ids = loader.get_all_webtoon_ids()
        
        if not webtoon_ids:
            print("RESULT: FAIL - No webtoons found for testing.")
        else:
            test_webtoon_id = webtoon_ids[0] # 첫 번째 웹툰 테스트
            sequence_info = loader.get_episode_sequence_info(test_webtoon_id)
        
            if not sequence_info:
                print(f"RESULT: FAIL - No sequence info found for {test_webtoon_id}")
            else:
                test_episode_ids = sequence_info[0][:3] # 첫 3개 에피소드 테스트
                if len(test_episode_ids) < 3:
                     print(f"RESULT: WARN - Fewer than 3 episodes found for {test_webtoon_id}, testing with {len(test_episode_ids)} episodes.")
                if not test_episode_ids:
                     print(f"RESULT: FAIL - No episodes to test for {test_webtoon_id}")
                else:
                    print(f"Testing Webtoon ID: {test_webtoon_id}, Episodes: {test_episode_ids}")
                    try:
                        # 분석 실행
                        final_state = analyze_webtoon_sequence(test_webtoon_id, test_episode_ids)
        
                        # 결과 확인
                        print("\\n--- Final State ---")
                        if final_state.get("error"):
                            print(f"Error occurred: {final_state['error']}")
                            print("RESULT: FAIL - Agent execution resulted in error.")
                        else:
                            print(f"Webtoon ID: {final_state.get('webtoon_id')}")
                            print(f"Episode IDs: {final_state.get('episode_ids')}")
                            print(f"\\nSequence Summary:\\n{final_state.get('sequence_summary', 'N/A')}")
                            print(f"\\nExtracted Text Snippet (Ep 1):\\n{final_state.get('per_episode_extracted_text', ['N/A'])[0][:200]}...")
                            print(f"\\nVisual Analysis Snippet (Ep 1):\\n{final_state.get('per_episode_visual_analysis', ['N/A'])[0][:200]}...")
                            print("\\nRESULT: SUCCESS - Agent workflow executed (check content quality manually).")
        
                    except Exception as e:
                        print(f"\\nRESULT: FAIL - Unexpected error during workflow test: {e}")
        
        ```
        
    4. **테스트 실행:** 프로젝트 루트에서 `python tests/test_agent_workflow.py` 실행 및 결과 확인.
- **완료 기준:**
    - 다중 에피소드 시퀀스 분석 워크플로우 통합 완료.
    - `analyze_webtoon_sequence` 함수 구현 및 테스트 완료.
    - 테스트 스크립트로 정상 동작 및 최종 상태에 시퀀스 요약 등 결과 포함 확인.

---

## **Phase 2: 그래프 데이터 구축 (Sequence Analysis 기반)**

### **WCAI-G01: 그래프 데이터 모델 설계 및 관리자 구현** (예상 소요시간: 90분)

- **설명:** 웹툰, 작가, 장르, 태그 등 주요 엔티티와 관계를 표현하는 NetworkX 기반 그래프 모델을 설계하고, 이를 관리하는 클래스를 구현합니다.
- **수행 단계:**
    1. **그래프 스키마 정의 (`docs/graph_schema.md`):**
        - Nodes: `Webtoon`(id, title, description, popularity, art_style, ...), `Episode`(id, title, sequence_order), `Character`(id, name, description), `Creator`(id, name), `Platform`(id, name), `Genre`(id, name), `Tag`(id, name), `Theme`(id, name), `PlotPoint`(id, description).
        - Edges: `HAS_EPISODE`(Webtoon->Episode), `APPEARS_IN`(Character->Episode), `CREATED_BY`(Creator->Webtoon), `PUBLISHED_ON`(Webtoon->Platform), `HAS_GENRE`(Webtoon->Genre), `HAS_TAG`(Webtoon->Tag), `HAS_THEME`(Webtoon->Theme), `CONTAINS_PLOT`(Webtoon->PlotPoint), `INTERACTS_WITH`(Character->Character, properties: relationship_type, episode_id), `SIMILAR_TO`(Webtoon->Webtoon, properties: weight, reason).
    2. **`WebtoonGraphModel` 클래스 구현 (`src/graph/graph_model.py`):**
        
        ```python
        import networkx as nx
        from typing import Dict, Any, Optional
        
        class WebtoonGraphModel:
            def __init__(self):
                # 방향성, 다중 엣지 허용 그래프 사용 고려 (예: 인물 관계)
                self.graph = nx.MultiDiGraph()
                print("Initialized WebtoonGraphModel with MultiDiGraph.")
        
            def add_node(self, node_id: str, node_type: str, attributes: Optional[Dict[str, Any]] = None):
                """노드 추가 또는 업데이트 (속성 포함)"""
                if attributes is None: attributes = {}
                # type은 필수 속성으로 추가
                if 'type' not in attributes: attributes['type'] = node_type
                if self.graph.has_node(node_id):
                    # 기존 노드 속성 업데이트
                    self.graph.nodes[node_id].update(attributes)
                else:
                    self.graph.add_node(node_id, **attributes)
                # print(f"Added/Updated node: {node_id} (Type: {node_type})")
        
            def add_edge(self, source_id: str, target_id: str, edge_type: str, attributes: Optional[Dict[str, Any]] = None):
                """엣지 추가 (속성 포함)"""
                if attributes is None: attributes = {}
                if 'type' not in attributes: attributes['type'] = edge_type
                # MultiDiGraph는 여러 엣지 허용, key로 구분 가능 (여기선 단순 추가)
                self.graph.add_edge(source_id, target_id, **attributes)
                # print(f"Added edge: {source_id} -[{edge_type}]-> {target_id}")
        
            def get_graph(self) -> nx.MultiDiGraph:
                """NetworkX 그래프 객체 반환"""
                return self.graph
        
            def save_graph(self, file_path: str = "data/graph_exports/webtoon_graph.graphml"):
                """그래프를 GraphML 형식으로 저장"""
                try:
                    nx.write_graphml(self.graph, file_path)
                    print(f"Graph saved successfully to {file_path}")
                except Exception as e:
                    print(f"Error saving graph: {e}")
        
            def load_graph(self, file_path: str = "data/graph_exports/webtoon_graph.graphml"):
                """GraphML 형식에서 그래프 로드"""
                if os.path.exists(file_path):
                    try:
                        self.graph = nx.read_graphml(file_path)
                        print(f"Graph loaded successfully from {file_path}")
                        return True
                    except Exception as e:
                        print(f"Error loading graph: {e}")
                        # 로드 실패 시 빈 그래프로 초기화
                        self.graph = nx.MultiDiGraph()
                        return False
                else:
                    print(f"Graph file not found at {file_path}, initializing empty graph.")
                    self.graph = nx.MultiDiGraph()
                    return False
        
        ```
        
    3. **`WebtoonGraphManager` 클래스 구현 (`src/graph/graph_manager.py`):**
        
        ```python
        import os
        from src.utils.data_loader import WebtoonDataLoader
        from src.graph.graph_model import WebtoonGraphModel
        from typing import Dict, Any
        
        class WebtoonGraphManager:
            def __init__(self, data_dir: str = "data", graph_file: str = "data/graph_exports/webtoon_graph.graphml"):
                self.data_loader = WebtoonDataLoader(data_dir)
                self.graph_model = WebtoonGraphModel()
                self.graph_file = graph_file
                # 시작 시 기존 그래프 로드 시도
                self.graph_model.load_graph(self.graph_file)
                self.graph = self.graph_model.get_graph() # 내부 그래프 참조
        
            def populate_from_metadata(self):
                """메타데이터 기반으로 초기 노드 및 관계 추가"""
                print("--- Populating graph from metadata ---")
                webtoon_metadata = self.data_loader.webtoon_metadata
        
                for wt in webtoon_metadata:
                    webtoon_id = wt['webtoon_id']
                    # 웹툰 노드 추가/업데이트
                    wt_attrs = {k: v for k, v in wt.items() if k not in ['genre', 'tags', 'creator', 'platform', 'analysis_episode_sequence']}
                    self.graph_model.add_node(webtoon_id, node_type="Webtoon", attributes=wt_attrs)
        
                    # 장르 노드 및 관계
                    for genre_name in wt.get('genre', []):
                        genre_id = f"genre_{genre_name.lower().replace(' ','_')}"
                        self.graph_model.add_node(genre_id, node_type="Genre", attributes={"name": genre_name})
                        self.graph_model.add_edge(webtoon_id, genre_id, edge_type="HAS_GENRE")
        
                    # 작가 노드 및 관계
                    creator_name = wt.get('creator')
                    if creator_name:
                        creator_id = f"creator_{creator_name.lower().replace(' ','_')}"
                        self.graph_model.add_node(creator_id, node_type="Creator", attributes={"name": creator_name})
                        self.graph_model.add_edge(creator_id, webtoon_id, edge_type="CREATED_BY") # 작가 -> 웹툰
        
                    # 플랫폼 노드 및 관계 (선택적)
                    platform_name = wt.get('platform')
                    if platform_name:
                        platform_id = f"platform_{platform_name.lower().replace(' ','_')}"
                        self.graph_model.add_node(platform_id, node_type="Platform", attributes={"name": platform_name})
                        self.graph_model.add_edge(webtoon_id, platform_id, edge_type="PUBLISHED_ON")
        
                print(f"Graph populated from metadata. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
        
            def update_graph_with_features(self, webtoon_id: str, features: Dict[str, Any]):
                """추출된 특징으로 그래프 업데이트"""
                if not self.graph.has_node(webtoon_id):
                    print(f"Warning: Webtoon node {webtoon_id} not found in graph. Cannot update features.")
                    return
        
                # 웹툰 노드 속성 업데이트 (예: 분위기)
                if 'atmosphere' in features:
                    self.graph.nodes[webtoon_id]['atmosphere'] = features['atmosphere']
        
                # 태그 노드 및 관계 추가
                for tag_name in features.get('tags', []):
                    tag_id = f"tag_{tag_name.lower().replace(' ','_')}"
                    self.graph_model.add_node(tag_id, node_type="Tag", attributes={"name": tag_name})
                    self.graph_model.add_edge(webtoon_id, tag_id, edge_type="HAS_TAG")
        
                # 주제 노드 및 관계 추가
                for theme_name in features.get('themes', []):
                    theme_id = f"theme_{theme_name.lower().replace(' ','_')}"
                    self.graph_model.add_node(theme_id, node_type="Theme", attributes={"name": theme_name})
                    self.graph_model.add_edge(webtoon_id, theme_id, edge_type="HAS_THEME")
        
                # 플롯 포인트 노드 및 관계 추가 (선택적)
                for plot_point in features.get('plot_points', []):
                     # PlotPoint 노드 ID 생성 방식 필요 (내용 기반 해시 등)
                     plot_id = f"plot_{hash(plot_point[:50])}" # 예시
                     self.graph_model.add_node(plot_id, node_type="PlotPoint", attributes={"description": plot_point})
                     self.graph_model.add_edge(webtoon_id, plot_id, edge_type="CONTAINS_PLOT")
        
                # 캐릭터 정보 업데이트 (고급)
                # ... 캐릭터 노드 생성 및 관계 업데이트 로직 ...
        
                print(f"Updated graph for {webtoon_id} with extracted features.")
        
            def add_similarity_edges(self, similarity_threshold=0.7):
                """(예시) 공통 태그 기반 유사도 엣지 추가"""
                print("--- Adding similarity edges based on common tags ---")
                webtoon_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'Webtoon']
                processed_pairs = set()
                added_edges = 0
        
                for i in range(len(webtoon_nodes)):
                    for j in range(i + 1, len(webtoon_nodes)):
                        wt1_id = webtoon_nodes[i]
                        wt2_id = webtoon_nodes[j]
                        pair = tuple(sorted((wt1_id, wt2_id)))
                        if pair in processed_pairs: continue
        
                        # 공통 태그 찾기
                        try:
                            tags1 = set(neighbor for neighbor, data in self.graph[wt1_id].items() if self.graph.nodes[neighbor].get('type') == 'Tag')
                            tags2 = set(neighbor for neighbor, data in self.graph[wt2_id].items() if self.graph.nodes[neighbor].get('type') == 'Tag')
                            common_tags = tags1.intersection(tags2)
                            jaccard_sim = len(common_tags) / len(tags1.union(tags2)) if len(tags1.union(tags2)) > 0 else 0
        
                            if jaccard_sim >= similarity_threshold:
                                 self.graph_model.add_edge(wt1_id, wt2_id, edge_type="SIMILAR_TAGS", attributes={"weight": jaccard_sim, "common": len(common_tags)})
                                 added_edges += 1
                        except KeyError: # 이웃이 없는 경우 등
                             continue
                        finally:
                             processed_pairs.add(pair)
        
                print(f"Added {added_edges} SIMILAR_TAGS edges (threshold={similarity_threshold}).")
        
            def save(self):
                 """현재 그래프 저장"""
                 self.graph_model.save_graph(self.graph_file)
        
            # 시각화 함수는 G04에서 구체화
            def visualize_graph(self, output_file: Optional[str] = None):
                 print(f"Graph visualization function called. Target: {output_file or 'default HTML'}")
                 # WCAI-G04에서 PyVis 로직 구현
                 pass
        
        ```
        
- **완료 기준:**
    - 상세 그래프 스키마 정의 완료.
    - `WebtoonGraphModel` 클래스 구현 (NetworkX 기반, 저장/로드 기능 포함).
    - `WebtoonGraphManager` 클래스 구현 (메타데이터 기반 초기 그래프 구축 기능 포함).

### **WCAI-G02: 웹툰 특징 추출 기능 구현 (Sequence 기반)** (예상 소요시간: 120분)

- **설명:** 시퀀스 분석 결과를 바탕으로 그래프에 추가할 정제된 특징(캐릭터 관계 변화, 플롯 포인트, 테마 변화 등)을 추출합니다.
- **수행 단계:**
    1. **`FeatureExtractor` 클래스/모듈 구현 (`src/analysis/feature_extractor.py`):**
        
        ```python
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        from typing import Dict, List, Any, Optional
        import json
        
        class FeatureExtractor:
            def __init__(self, model_name="gpt-4o-mini", temperature=0.1, request_timeout=60):
                self.llm = ChatOpenAI(model=model_name, temperature=temperature, request_timeout=request_timeout)
                print(f"Initialized FeatureExtractor with model: {model_name}")
        
            def extract_features_from_sequence(self,
                                               sequence_summary: str,
                                               webtoon_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                """시퀀스 요약과 메타데이터로부터 구조화된 특징 추출"""
                print(f"--- Extracting features for: {webtoon_metadata.get('title', 'N/A')} ---")
                if not sequence_summary:
                    print("Warning: Empty sequence summary provided.")
                    return None
        
                system_prompt = """당신은 웹툰 분석 전문가입니다. 주어진 웹툰 시퀀스 요약과 메타데이터를 바탕으로, 다음 항목들을 **JSON 형식**으로 추출해주세요. 각 항목에 해당하는 내용이 없으면 빈 리스트([]) 또는 null 값을 사용하세요:
                1.  `main_characters`: 시퀀스에 중요하게 등장하는 인물 이름 목록 (List[str]).
                2.  `character_relationships`: 주요 인물 간의 관계 변화나 특징적 상호작용에 대한 간략한 설명 (str).
                3.  `plot_points`: 시퀀스 내 핵심 사건 또는 전환점 요약 목록 (List[str]).
                4.  `themes`: 시퀀스에서 드러나는 주요 주제나 테마 목록 (List[str]).
                5.  `atmosphere`: 시퀀스의 전체적인 분위기 (예: 긴장감 넘침, 코믹함, 감성적 등) (str).
                6.  `content_tags`: 내용을 잘 나타내는 추가적인 키워드 태그 목록 (List[str]). 기존 장르/태그 외 내용 기반 태그.
                """
        
                user_prompt = f"""분석 대상 웹툰 시퀀스 정보:
                - 제목: {webtoon_metadata.get('title', 'N/A')}
                - 장르: {webtoon_metadata.get('genre', [])}
                - 기존 태그: {webtoon_metadata.get('tags', [])}
                - 시퀀스 요약:
                {sequence_summary}
        
                위 정보를 바탕으로 JSON 형식의 특징 분석 결과를 생성해주세요.
                """
        
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
        
                try:
                    response = self.llm.invoke(messages)
                    content = response.content.strip()
                    print("LLM response received for feature extraction.")
        
                    # LLM 응답에서 JSON 부분 추출 시도 (마크다운 코드 블록 등 제거)
                    json_match = re.search(r"```json\\s*([\\s\\S]*?)\\s*```", content)
                    if json_match:
                        json_str = json_match.group(1)
                    else: # JSON 코드 블록이 없을 경우, 전체 응답이 JSON이라고 가정
                         json_str = content
        
                    # JSON 파싱
                    extracted_data = json.loads(json_str)
                    print("Successfully parsed features from LLM response.")
                    # 간단한 유효성 검사 (필요한 키 존재 여부 등)
                    required_keys = ['main_characters', 'character_relationships', 'plot_points', 'themes', 'atmosphere', 'content_tags']
                    if all(key in extracted_data for key in required_keys):
                        return extracted_data
                    else:
                         print(f"Warning: Missing required keys in parsed JSON. Parsed: {extracted_data}")
                         return None # 불완전한 데이터 처리
        
                except json.JSONDecodeError:
                    print(f"Error: Failed to decode JSON from LLM response.\\nResponse:\\n{content}")
                    return None
                except Exception as e:
                    print(f"Error during feature extraction LLM call: {e}")
                    return None
        
        ```
        
    2. **테스트 스크립트 작성 (`tests/test_feature_extractor.py`):**
        - 샘플 시퀀스 요약 및 메타데이터 준비.
        - `FeatureExtractor` 인스턴스 생성.
        - `extract_features_from_sequence` 호출.
        - 반환된 딕셔너리 내용 확인 (JSON 구조, 예상 키 존재 여부 등).
- **완료 기준:**
    - 시퀀스 분석 결과로부터 구조화된 특징 추출 함수 구현 완료.
    - LLM 응답(JSON) 파싱 및 오류 처리 로직 구현.
    - 테스트 스크립트로 정상 동작 및 유효한 JSON 출력 확인.

### **WCAI-G03: 그래프 자동 구축 파이프라인 구현 (Sequence 기반)** (예상 소요시간: 150분)

- **설명:** 전체 웹툰 데이터에 대해 시퀀스 분석 -> 특징 추출 -> 그래프 업데이트 파이프라인을 구현합니다. 관계(엣지) 생성 로직이 핵심입니다.
- **수행 단계:**
    1. **파이프라인 스크립트 작성 (`scripts/build_graph_pipeline.py`):**
        
        ```python
        import sys, os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from src.utils.data_loader import WebtoonDataLoader
        from src.agents.webtoon_agent import analyze_webtoon_sequence # 시퀀스 분석 함수
        from src.analysis.feature_extractor import FeatureExtractor # 특징 추출기
        from src.graph.graph_manager import WebtoonGraphManager # 그래프 관리자
        from dotenv import load_dotenv
        import json
        import time
        
        # 결과 저장을 위한 디렉토리 (캐싱 또는 중간 결과 저장용)
        RESULTS_DIR = "data/analysis_results"
        FEATURES_DIR = os.path.join(RESULTS_DIR, "features")
        GRAPH_FILE = "data/graph_exports/webtoon_knowledge_graph.graphml"
        
        os.makedirs(FEATURES_DIR, exist_ok=True)
        
        def run_pipeline():
            load_dotenv() # API 키 로드
            start_time = time.time()
        
            print("--- Starting Graph Build Pipeline ---")
            loader = WebtoonDataLoader()
            graph_manager = WebtoonGraphManager(graph_file=GRAPH_FILE)
            feature_extractor = FeatureExtractor() # LLM 기반 특징 추출기 초기화
        
            # 1. 메타데이터 기반 초기 그래프 생성
            graph_manager.populate_from_metadata()
        
            webtoon_ids = loader.get_all_webtoon_ids()
            print(f"Processing {len(webtoon_ids)} webtoons...")
        
            for i, webtoon_id in enumerate(webtoon_ids):
                print(f"\\n[{i+1}/{len(webtoon_ids)}] Processing Webtoon: {webtoon_id}")
                webtoon_meta = loader.get_webtoon_metadata(webtoon_id)
                sequence_info = loader.get_episode_sequence_info(webtoon_id)
        
                if not webtoon_meta or not sequence_info or not sequence_info[0]:
                    print(f"Skipping {webtoon_id}: Missing metadata or sequence info.")
                    continue
        
                episode_ids = sequence_info[0] # 분석할 에피소드 ID 리스트
        
                # 2. 시퀀스 분석 실행 (결과 캐싱 로직 추가 권장)
                # 예: 이미 분석 결과 파일이 있으면 로드, 없으면 실행 후 저장
                # sequence_result_file = os.path.join(RESULTS_DIR, f"{webtoon_id}_sequence_analysis.json")
                # if os.path.exists(sequence_result_file):
                #     with open(sequence_result_file, 'r', encoding='utf-8') as f: sequence_result = json.load(f)
                # else:
                sequence_result = analyze_webtoon_sequence(webtoon_id, episode_ids)
                #     # 분석 결과 저장 (선택적)
                #     # with open(sequence_result_file, 'w', encoding='utf-8') as f: json.dump(sequence_result, f, ensure_ascii=False, indent=2)
        
                if sequence_result.get("error"):
                    print(f"Skipping {webtoon_id}: Sequence analysis failed - {sequence_result['error']}")
                    continue
        
                # 3. 특징 추출 실행
                features_file = os.path.join(FEATURES_DIR, f"{webtoon_id}_features.json")
                if os.path.exists(features_file):
                     with open(features_file, 'r', encoding='utf-8') as f: features = json.load(f)
                else:
                     features = feature_extractor.extract_features_from_sequence(
                         sequence_summary=sequence_result.get("sequence_summary", ""),
                         webtoon_metadata=webtoon_meta
                     )
                     if features:
                         with open(features_file, 'w', encoding='utf-8') as f: json.dump(features, f, ensure_ascii=False, indent=2)
                     else:
                         print(f"Skipping feature update for {webtoon_id}: Feature extraction failed.")
                         features = {} # 빈 값으로 처리
        
                # 4. 그래프 업데이트 (노드 속성 및 특징 기반 엣지)
                if features:
                    graph_manager.update_graph_with_features(webtoon_id, features)
                else: # 특징 추출 실패 시 메타데이터 기반 태그라도 추가
                     tags_from_meta = webtoon_meta.get('tags', [])
                     if tags_from_meta:
                          graph_manager.update_graph_with_features(webtoon_id, {"tags": tags_from_meta})
        
            # 5. 모든 웹툰 처리 후, 관계(유사도) 엣지 생성
            graph_manager.add_similarity_edges(similarity_threshold=0.3) # 예시: 공통 태그 30% 이상 유사
        
            # 6. 최종 그래프 저장
            graph_manager.save()
        
            end_time = time.time()
            print(f"--- Graph Build Pipeline Finished ---")
            print(f"Total time: {end_time - start_time:.2f} seconds")
            print(f"Final Graph: Nodes={graph_manager.graph.number_of_nodes()}, Edges={graph_manager.graph.number_of_edges()}")
            print(f"Graph saved to: {GRAPH_FILE}")
        
        if __name__ == "__main__":
            run_pipeline()
        
        ```
        
- **완료 기준:**
    - 전체 샘플 웹툰에 대한 시퀀스 분석->특징 추출->그래프 업데이트 자동화 파이프라인 구현 완료.
    - 특징 기반 관계(유사도) 생성 로직 구현 (최소 1가지 방식).
    - 파이프라인 실행 후 최종 그래프 파일(`.graphml`) 생성 확인.

### **WCAI-G04: 그래프 시각화 기능 구현** (예상 소요시간: 60분)

- **설명:** 구축된 웹툰 지식 그래프를 시각화하여 관계를 직관적으로 파악할 수 있도록 합니다.
- **수행 단계:**
    1. **시각화 함수 구현 (`src/graph/graph_manager.py` 내 `visualize_graph` 메서드):**
        
        ```python
        from pyvis.network import Network
        import networkx as nx
        import os
        from typing import Optional
        
        # ... (WebtoonGraphManager 클래스 내) ...
        def visualize_graph(self, output_file: str = "data/graph_exports/webtoon_graph_visualization.html", physics: bool = True):
            """그래프를 PyVis를 사용하여 HTML 파일로 시각화"""
            print(f"--- Generating graph visualization to {output_file} ---")
            if not self.graph or self.graph.number_of_nodes() == 0:
                print("Graph is empty, cannot visualize.")
                return
        
            # PyVis 네트워크 객체 생성
            net = Network(notebook=False, height="800px", width="100%", directed=isinstance(self.graph, nx.DiGraph))
        
            # 물리 엔진 설정 (노드가 많을 경우 False 고려)
            if physics:
                net.show_buttons(filter_=['physics']) # 물리 효과 조절 버튼 표시
            else:
                net.force_atlas_2based() # 물리 효과 비활성화 시 레이아웃 알고리즘
        
            # 노드 추가 및 스타일링
            node_colors = {"Webtoon": "#1f77b4", "Genre": "#ff7f0e", "Tag": "#2ca02c", "Creator": "#d62728", "Theme": "#9467bd", "PlotPoint": "#8c564b", "Platform": "#e377c2"}
            node_sizes = {"Webtoon": 25, "Creator": 15, "Genre": 10, "Tag": 8, "Theme": 8, "PlotPoint": 8, "Platform": 10}
        
            for node, data in self.graph.nodes(data=True):
                node_type = data.get('type', 'Unknown')
                label = data.get('title') or data.get('name') or data.get('value') or str(node) # 표시될 라벨
                size = node_sizes.get(node_type, 5)
                color = node_colors.get(node_type, "#7f7f7f")
                title_attr = f"ID: {node}\\nType: {node_type}\\n" + "\\n".join([f"{k}: {v}" for k, v in data.items() if k not in ['type']]) # 마우스 오버 시 정보
        
                net.add_node(str(node), label=label, title=title_attr, size=size, color=color)
        
            # 엣지 추가 및 스타일링
            for u, v, data in self.graph.edges(data=True):
                edge_type = data.get('type', '')
                weight = data.get('weight', 1.0)
                title_attr = f"Type: {edge_type}\\nWeight: {weight:.2f}"
                # 엣지 두께/색상 조절 가능
                edge_width = max(0.5, weight * 2) if edge_type.startswith("SIMILAR") else 1.0
        
                net.add_edge(str(u), str(v), title=title_attr, width=edge_width)
        
            # HTML 파일 저장
            try:
                # 파일 경로의 디렉토리 확인 및 생성
                output_dir = os.path.dirname(output_file)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                net.save_graph(output_file)
                print(f"Graph visualization saved to {output_file}")
            except Exception as e:
                print(f"Error saving graph visualization: {e}")
        
        ```
        
    2. **파이프라인(G03) 완료 후 또는 별도 스크립트에서 호출:** `graph_manager.visualize_graph()` 실행.
- **완료 기준:**
    - 구축된 그래프를 PyVis로 시각화하는 기능 구현 완료.
    - 노드/엣지 스타일링 적용.
    - 상호작용 가능한 HTML 시각화 파일 정상 생성 확인.

## **Phase 3: PathRAG 통합 및 분석/추천 기능 구현 (Sequence 기반)**

### **WCAI-PR01: PathRAG 설정 및 데이터 인덱싱 (Sequence 기반)** (예상 소요시간: 120분)

- **설명:** PathRAG 라이브러리를 설정하고, 웹툰 메타데이터와 **시퀀스 분석 기반의 상세 정보**를 텍스트로 변환하여 인덱싱(삽입)합니다.
- **수행 단계:**
    1. **`WebtoonPathRAG` 클래스 구현 (`src/graph/pathrag_integration.py`):**
        
        ```python
        import os
        import json
        from typing import Dict, List, Any, Optional
        # PathRAG 라이브러리 import 확인 (설치 경로에 따라 다를 수 있음)
        try:
             from PathRAG import PathRAG, QueryParam
             from PathRAG.llm import gpt_4o_mini_complete # 또는 다른 LLM 함수
        except ImportError as e:
             print(f"Error importing PathRAG: {e}. Make sure PathRAG is installed correctly.")
             # 필요한 경우 sys.path 조정 또는 예외 처리
             sys.exit(1) # PathRAG 없으면 실행 불가
        from src.utils.data_loader import WebtoonDataLoader # 데이터 로딩용
        # 특징, 요약 로딩 경로 정의
        FEATURES_DIR = "data/analysis_results/features"
        # SUMMARY_DIR = "data/analysis_results" # 필요시 요약도 별도 저장
        
        class WebtoonPathRAG:
            def __init__(self,
                         working_dir: str = "./pathrag_working_dir",
                         model_func = gpt_4o_mini_complete, # PathRAG 내부 LLM 설정
                         api_key: Optional[str] = None):
        
                if api_key: os.environ["OPENAI_API_KEY"] = api_key
                if not os.getenv("OPENAI_API_KEY"): raise ValueError("OPENAI_API_KEY must be set.")
        
                self.working_dir = working_dir
                if not os.path.exists(working_dir): os.makedirs(working_dir)
        
                print(f"Initializing PathRAG with working_dir: {working_dir}")
                self.rag = PathRAG(working_dir=working_dir, llm_model_func=model_func)
                self.loader = WebtoonDataLoader() # 메타데이터 로딩용
        
            def _generate_webtoon_text(self, webtoon_id: str) -> Optional[str]:
                """PathRAG 인덱싱을 위한 상세 텍스트 생성"""
                metadata = self.loader.get_webtoon_metadata(webtoon_id)
                if not metadata: return None
        
                # 특징 데이터 로드
                features = {}
                feature_file = os.path.join(FEATURES_DIR, f"{webtoon_id}_features.json")
                if os.path.exists(feature_file):
                    try:
                        with open(feature_file, 'r', encoding='utf-8') as f: features = json.load(f)
                    except json.JSONDecodeError: print(f"Warning: Could not decode feature file {feature_file}")
        
                # 시퀀스 요약 로드 (Agent 실행 결과에서 가져오거나, 별도 저장된 파일 로드)
                # 여기서는 요약이 특징에 포함되었다고 가정하거나, 별도 로직 필요
                # sequence_summary = features.get("sequence_summary", "") # 예시
                # 실제로는 G03 파이프라인에서 저장/관리된 요약 로드 필요
                sequence_summary = "시퀀스 요약 로딩 로직 필요" # Placeholder
        
                # 상세 텍스트 생성
                text = f"웹툰 정보: {webtoon_id}\\n"
                text += f"제목: {metadata.get('title', 'N/A')}\\n"
                text += f"작가: {metadata.get('creator', 'N/A')}\\n"
                text += f"플랫폼: {metadata.get('platform', 'N/A')}\\n"
                text += f"장르: {', '.join(metadata.get('genre', []))}\\n"
                text += f"기본 태그: {', '.join(metadata.get('tags', []))}\\n"
                text += f"아트 스타일: {metadata.get('art_style', 'N/A')}\\n"
                text += f"인기도: {metadata.get('popularity_score', 'N/A')}\\n"
                text += f"로맨스 포함 여부: {'예' if metadata.get('is_romance') else '아니오'}\\n"
                text += f"설명: {metadata.get('description', 'N/A')}\\n"
                text += f"\\n분석된 시퀀스 요약:\\n{sequence_summary}\\n" # 시퀀스 요약 포함
                text += f"\\n추출된 주요 특징:\\n"
                text += f"- 주요 등장인물: {', '.join(features.get('main_characters', []))}\\n"
                text += f"- 캐릭터 관계/변화: {features.get('character_relationships', 'N/A')}\\n"
                text += f"- 핵심 플롯 포인트: {'; '.join(features.get('plot_points', []))}\\n"
                text += f"- 주요 주제/테마: {', '.join(features.get('themes', []))}\\n"
                text += f"- 전체 분위기: {features.get('atmosphere', 'N/A')}\\n"
                text += f"- 내용 기반 태그: {', '.join(features.get('content_tags', []))}\\n"
        
                return text.strip()
        
            def build_index_from_metadata(self, force_rebuild: bool = False):
                """모든 웹툰 데이터 인덱싱 (기존 인덱스 삭제 옵션)"""
                print("--- Building PathRAG index ---")
                if force_rebuild and os.path.exists(self.working_dir):
                    import shutil
                    print(f"Removing existing index at {self.working_dir}")
                    shutil.rmtree(self.working_dir)
                    os.makedirs(self.working_dir)
                    # PathRAG 재초기화 필요할 수 있음
                    self.rag = PathRAG(working_dir=self.working_dir, llm_model_func=gpt_4o_mini_complete)
        
                webtoon_ids = self.loader.get_all_webtoon_ids()
                indexed_count = 0
                for webtoon_id in webtoon_ids:
                    print(f"Generating text for {webtoon_id}...")
                    webtoon_text = self._generate_webtoon_text(webtoon_id)
                    if webtoon_text:
                        try:
                            print(f"Inserting data for {webtoon_id} into PathRAG...")
                            self.rag.insert(webtoon_text) # PathRAG에 텍스트 삽입
                            indexed_count += 1
                        except Exception as e:
                            print(f"Error inserting {webtoon_id} into PathRAG: {e}")
                    else:
                        print(f"Skipping {webtoon_id}: Could not generate text.")
                print(f"--- PathRAG indexing complete. Indexed {indexed_count}/{len(webtoon_ids)} webtoons. ---")
        
            def query(self, question: str, mode: str = "hybrid", max_paths: int = 5, k: int = 5) -> Any:
                """PathRAG 쿼리 실행"""
                print(f"--- PathRAG Query ---")
                print(f"Question: {question}")
                print(f"Mode: {mode}, Max Paths: {max_paths}, K: {k}")
                try:
                    # QueryParam 설정 확인 (라이브러리 버전 따라 다를 수 있음)
                    param = QueryParam(mode=mode, k=k, max_path_num=max_paths)
                    result = self.rag.query(question, param=param)
                    print("PathRAG Query successful.")
                    return result
                except Exception as e:
                    print(f"Error during PathRAG query: {e}")
                    return f"Error: {str(e)}"
        
        ```
        
    2. **인덱싱 스크립트 작성 및 실행 (`scripts/build_pathrag_index.py`):**
        
        ```python
        import sys, os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.graph.pathrag_integration import WebtoonPathRAG
        from dotenv import load_dotenv
        
        if __name__ == "__main__":
            load_dotenv()
            # API 키 직접 전달 또는 환경변수 확인
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                 print("Error: OPENAI_API_KEY not set.")
            else:
                 rag_builder = WebtoonPathRAG(api_key=api_key)
                 # force_rebuild=True로 설정하면 기존 인덱스 삭제 후 재생성
                 rag_builder.build_index_from_metadata(force_rebuild=False)
        
        ```
        
    3. **실행:** 프로젝트 루트에서 `python scripts/build_pathrag_index.py` 실행. `pathrag_working_dir`에 파일 생성 확인.
- **완료 기준:**
    - `WebtoonPathRAG` 클래스 구현 완료 (초기화, 텍스트 생성, 인덱싱, 쿼리 메서드).
    - 전체 웹툰 데이터에 대한 상세 텍스트 생성 및 PathRAG 인덱싱 기능 구현 완료.
    - 인덱싱 스크립트 실행 성공 확인.

### **WCAI-PR02: PathRAG 기반 분석/추천 기능 구현 (Sequence 기반)** (예상 소요시간: 150분)

- **설명:** PathRAG 쿼리 기능을 활용하여 시퀀스 내용을 반영한 분석/추천 시나리오를 구현합니다.
- **수행 단계:**
    1. **쿼리 처리 모듈/클래스 구현 (`src/analysis/pathrag_analyzer.py`):**
        
        ```python
        from src.graph.pathrag_integration import WebtoonPathRAG
        from typing import Dict, List, Any, Optional
        
        class PathRAGAnalyzer:
            def __init__(self, pathrag_instance: WebtoonPathRAG):
                self.pathrag = pathrag_instance
                print("Initialized PathRAGAnalyzer.")
        
            def _post_process_result(self, result: Any) -> str:
                """PathRAG 결과를 사용자 친화적 텍스트로 후처리"""
                # PathRAG 결과 형식에 따라 파싱/정리 필요
                # 예시: result가 텍스트 문자열이라고 가정
                if isinstance(result, str):
                    # 간단히 앞뒤 공백 제거 및 정리
                    return result.strip()
                # TODO: PathRAG 라이브러리의 실제 반환 형식 확인 후 구현
                return str(result) # 기본 변환
        
            def find_popular_pattern(self, genre: Optional[str] = None, top_n: int = 3) -> str:
                """인기 웹툰 패턴 분석"""
                query = f"상위 {top_n}개의 인기 웹툰들의 공통적인 특징이나 성공 요인을 분석해줘."
                if genre:
                    query = f"'{genre}' 장르 " + query
                print(f"Executing popular pattern query: {query}")
                # 'graph' 모드가 관계 분석에 더 적합할 수 있음
                result = self.pathrag.query(query, mode="graph", max_paths=10, k=top_n * 2)
                return self._post_process_result(result)
        
            def analyze_creator_strategy(self, creator_name: str) -> str:
                """크리에이터 전략 분석"""
                query = f"웹툰 작가 '{creator_name}'의 작품 스타일, 주요 주제, 성공 전략 등을 분석해줘."
                print(f"Executing creator strategy query for: {creator_name}")
                result = self.pathrag.query(query, mode="hybrid", max_paths=8, k=5)
                return self._post_process_result(result)
        
            def recommend_similar_webtoons(self, webtoon_title: str, num_recs: int = 5) -> str:
                """유사 웹툰 추천"""
                # 상세 정보 활용 쿼리
                query = f"웹툰 '{webtoon_title}'과 줄거리, 분위기, 그림체, 주제 면에서 가장 유사한 다른 웹툰 {num_recs}개를 추천해주고, 각 추천의 이유를 설명해줘."
                print(f"Executing similar webtoon query for: {webtoon_title}")
                result = self.pathrag.query(query, mode="hybrid", max_paths=10, k=num_recs * 2)
                # TODO: 결과에서 웹툰 목록만 추출하는 후처리 필요 가능성
                return self._post_process_result(result)
        
            def recommend_by_criteria(self, criteria_text: str, num_recs: int = 5) -> str:
                """조건 기반 웹툰 추천"""
                query = f"다음 조건을 만족하는 웹툰 {num_recs}개를 추천해줘: {criteria_text}. 각 추천 이유도 설명해줘."
                print(f"Executing criteria-based recommendation query: {criteria_text}")
                result = self.pathrag.query(query, mode="hybrid", max_paths=10, k=num_recs * 2)
                # TODO: 결과 후처리
                return self._post_process_result(result)
        
        ```
        
    2. **테스트 스크립트 작성 (`tests/test_pathrag_analyzer.py`):**
        - `WebtoonPathRAG` 인스턴스 생성 (인덱싱 완료 상태 가정).
        - `PathRAGAnalyzer` 인스턴스 생성.
        - 각 분석/추천 함수 호출 및 결과 확인 (텍스트 출력).
- **완료 기준:**
    - 주요 분석/추천 시나리오별 쿼리 처리 함수 구현 완료.
    - PathRAG 결과 후처리 로직 기본 구현.
    - 테스트 스크립트로 각 기능 호출 및 응답 확인.

### **WCAI-PR03: PathRAG 성능 최적화 및 평가** (예상 소요시간: 60분)

- **설명:** PathRAG 쿼리 성능(응답 속도, 품질, 토큰 사용량)을 평가하고 개선합니다.
- **수행 단계:**
    1. **쿼리 파라미터 튜닝 (`QueryParam`):**
        - `recommend_by_criteria` 등 주요 함수에 대해 `mode` ('hybrid', 'graph', 'vector'), `k`, `max_paths`, `max_path_length` 변경하며 결과 비교.
        - 예: `mode='graph'`가 관계 기반 질문에 더 좋은지, `k` 값을 늘리면 추천 다양성이 증가하는지 확인.
    2. **데이터 인덱싱 텍스트 최적화:** (WCAI-PR01 관련) `_generate_webtoon_text` 함수 결과물의 구조나 상세 수준을 변경했을 때 PathRAG 응답 품질 변화 관찰.
    3. **LLM 프롬프트 검토:** WCAI-PR02의 쿼리 생성 로직 또는 PathRAG 내부 LLM 프롬프트(수정 가능하다면) 검토.
    4. **토큰 사용량 측정:** LangChain Callback Handler 등을 사용하여 `WebtoonPathRAG.query` 호출 시 LLM 토큰 사용량 측정 및 기록. 파라미터 변경에 따른 변화 확인.
    5. 간단한 성능 평가 결과 요약 (`docs/pathrag_performance.md`).
- **완료 기준:**
    - 주요 쿼리에 대한 PathRAG 파라미터 튜닝 시도 및 결과 기록.
    - 토큰 사용량 측정 방법 확인 및 일부 쿼리에 대해 측정.
    - 성능 평가 결과 간략히 문서화.

## **Phase 4: UI 통합 및 최종화 (다중 에피소드 + PathRAG)**

### **WCAI-P10: Streamlit UI - 입력 방식 수정 (다중 이미지/시퀀스 선택)** (예상 소요시간: 60분)

- **설명:** 사용자가 여러 개의 에피소드 이미지 파일을 한 번에 업로드하거나, 샘플 웹툰의 분석 대상 시퀀스를 선택할 수 있도록 UI를 수정합니다.
- **수행 단계:**
    1. **`app.py` 사이드바 UI 구현:**
        
        ```python
        import streamlit as st
        from src.utils.data_loader import WebtoonDataLoader
        # ... other imports ...
        
        st.set_page_config(page_title="Webtoon Analysis + PathRAG", layout="wide")
        st.sidebar.title("웹툰 분석 & 추천 Agent")
        st.sidebar.markdown("---")
        
        # --- 입력 선택 ---
        input_method = st.sidebar.radio("분석 대상 선택", ["샘플 웹툰 사용", "이미지 파일 업로드"], index=0)
        
        uploaded_files = None
        selected_webtoon_id = None
        selected_episode_ids = None
        selected_episode_titles = None
        
        try:
            loader = WebtoonDataLoader()
            webtoon_ids = loader.get_all_webtoon_ids()
        except Exception as e:
             st.sidebar.error(f"데이터 로딩 실패: {e}")
             webtoon_ids = []
        
        if input_method == "샘플 웹툰 사용":
            if not webtoon_ids:
                st.sidebar.warning("사용 가능한 샘플 웹툰이 없습니다.")
            else:
                selected_webtoon_id_from_sample = st.sidebar.selectbox(
                    "분석할 웹툰 선택", webtoon_ids, format_func=lambda x: loader.get_webtoon_metadata(x).get('title', x)
                )
                if selected_webtoon_id_from_sample:
                     sequence_info = loader.get_episode_sequence_info(selected_webtoon_id_from_sample)
                     if sequence_info and sequence_info[0]:
                         selected_webtoon_id = selected_webtoon_id_from_sample
                         selected_episode_ids = sequence_info[0] # 정의된 시퀀스 사용
                         selected_episode_titles = sequence_info[1]
                         st.sidebar.caption(f"선택됨: {loader.get_webtoon_metadata(selected_webtoon_id)['title']} ({len(selected_episode_ids)} 에피소드)")
                     else:
                         st.sidebar.warning(f"{selected_webtoon_id_from_sample}의 에피소드 시퀀스 정보 없음.")
        
        elif input_method == "이미지 파일 업로드":
            uploaded_files = st.sidebar.file_uploader(
                "연속된 에피소드 이미지 파일들을 순서대로 업로드하세요.",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True
            )
            if uploaded_files:
                st.sidebar.caption(f"{len(uploaded_files)}개 이미지 업로드됨.")
                # 업로드 시 ID/Title 임의 지정
                selected_webtoon_id = "uploaded_webtoon"
                selected_episode_ids = [f"ep{i+1}" for i in range(len(uploaded_files))]
                selected_episode_titles = [f"Uploaded Episode {i+1}" for i in range(len(uploaded_files))]
        
        # 분석 실행 버튼 (상태에 따라 활성화)
        can_analyze = (selected_webtoon_id and selected_episode_ids) or (uploaded_files)
        analyze_button_placeholder = st.sidebar.empty()
        
        # --- 메인 영역 (다음 티켓에서 채움) ---
        st.title("✨ Webtoon Sequence Analysis + PathRAG")
        st.markdown("---")
        main_area = st.container()
        # ...
        
        # 버튼 생성
        if analyze_button_placeholder.button("웹툰 시퀀스 분석 실행", type="primary", key="analyze_sequence_btn", disabled=not can_analyze):
            # TODO: WCAI-P12에서 로직 연결
            with main_area: st.info("분석 실행 로직 연결 예정...")
        
        ```
        
- **완료 기준:** 다중 이미지 업로드 및 샘플 시퀀스 선택 UI 구현 완료. 입력 상태 관리 및 버튼 활성화 로직 구현.

### **WCAI-P11: Streamlit UI - 결과 표시 수정 (시퀀스 + PathRAG)** (예상 소요시간: 90분)

- **설명:** WCAI Agent의 시퀀스 분석 결과와 PathRAG의 분석/추천 결과를 통합적으로 보여주도록 UI를 구성합니다.
- **수행 단계:**
    1. **결과 표시 함수 구현 (`app.py` 또는 `src/app/ui_display.py`):**
        
        ```python
        import streamlit as st
        from typing import Dict, List, Any
        
        def display_sequence_analysis(results_data: Dict):
            """WCAI Agent 시퀀스 분석 결과 표시"""
            st.subheader(f"웹툰 시퀀스 분석 결과: {results_data.get('episode_title', results_data.get('webtoon_id', 'N/A'))}")
            if results_data.get("error"):
                st.error(f"분석 중 오류: {results_data['error']}")
                return
        
            st.markdown("**📜 시퀀스 전체 요약**")
            st.markdown(results_data.get('sequence_summary', '요약 없음'))
            st.markdown("---")
        
            with st.expander("📚 에피소드별 상세 분석 보기", expanded=False):
                texts = results_data.get('per_episode_extracted_text', [])
                analyses = results_data.get('per_episode_visual_analysis', [])
                titles = results_data.get('episode_titles', [''] * len(texts))
        
                if texts or analyses:
                     for i in range(max(len(texts), len(analyses))):
                         st.markdown(f"**Episode {i+1}: {titles[i] if i < len(titles) else ''}**")
                         cols = st.columns(2)
                         with cols[0]:
                             st.caption("📄 추출된 텍스트")
                             st.text_area(f"txt_{i}", texts[i] if i < len(texts) else "N/A", height=200, key=f"text_ep_{i}")
                         with cols[1]:
                             st.caption("🖼️ 시각적 분석")
                             st.text_area(f"vis_{i}", analyses[i] if i < len(analyses) else "N/A", height=200, key=f"analysis_ep_{i}")
                         st.markdown("---")
                else:
                     st.info("상세 분석 결과가 없습니다.")
        
        def display_pathrag_results(query: str, result_text: str):
             """PathRAG 쿼리 결과 표시"""
             st.subheader("💬 PathRAG 분석/추천 결과")
             st.markdown(f"**질문:** {query}")
             st.markdown("**답변:**")
             st.markdown(result_text) # 결과 텍스트 표시
             st.markdown("---")
        
        def display_graph_visualization(html_file: str = "data/graph_exports/webtoon_graph_visualization.html"):
            """그래프 시각화 HTML 표시"""
            st.subheader("🕸️ 웹툰 지식 그래프 시각화")
            if os.path.exists(html_file):
                 try:
                     with open(html_file, 'r', encoding='utf-8') as f:
                         html_content = f.read()
                     st.components.v1.html(html_content, height=600, scrolling=True)
                 except Exception as e:
                     st.error(f"그래프 시각화 파일 로드 실패: {e}")
            else:
                 st.warning("그래프 시각화 파일이 생성되지 않았습니다. 그래프 구축 파이프라인을 실행하세요.")
        
        ```
        
    2. **메인 영역 레이아웃 구성 (`app.py`):** 탭 또는 섹션으로 결과 구분.
        
        ```python
        # ... (사이드바 코드 이후) ...
        # 메인 영역
        st.title("✨ Webtoon Sequence Analysis + PathRAG")
        st.markdown("---")
        
        # 분석 결과 표시 영역 (Session State 활용 권장)
        if 'analysis_result' not in st.session_state: st.session_state['analysis_result'] = None
        if 'pathrag_query' not in st.session_state: st.session_state['pathrag_query'] = ""
        if 'pathrag_result' not in st.session_state: st.session_state['pathrag_result'] = None
        
        # --- 시퀀스 분석 결과 표시 ---
        if st.session_state['analysis_result']:
            display_sequence_analysis(st.session_state['analysis_result'])
        
        st.markdown("---")
        
        # --- PathRAG 분석/추천 ---
        st.subheader("🔍 PathRAG 기반 분석 및 추천")
        pathrag_query_input = st.text_area("질문 입력:", height=100, key="pathrag_query_input", placeholder="예: '로맨스 없고 액션 스릴러 웹툰 추천해줘', '인기 판타지 웹툰들의 공통점은?'")
        pathrag_query_button = st.button("PathRAG에게 질문하기", key="pathrag_query_btn")
        
        if pathrag_query_button and pathrag_query_input:
            # TODO: WCAI-P12에서 PathRAG 쿼리 로직 호출
            st.session_state['pathrag_query'] = pathrag_query_input
            st.info("PathRAG 쿼리 실행 로직 연결 예정...")
            # 임시 결과 표시
            st.session_state['pathrag_result'] = "PathRAG로부터 답변을 받아오는 중입니다... (연결 예정)"
        
        if st.session_state['pathrag_result']:
             display_pathrag_results(st.session_state['pathrag_query'], st.session_state['pathrag_result'])
        
        st.markdown("---")
        
        # --- 그래프 시각화 ---
        # TODO: WCAI-P12에서 연동
        display_graph_visualization()
        
        # 초기 안내 메시지 (결과 없을 때)
        if not st.session_state['analysis_result'] and not st.session_state['pathrag_result']:
             st.info("👈 사이드바에서 이미지를 업로드하거나 샘플을 선택 후 '분석 실행', 또는 아래에 PathRAG 질문을 입력하세요.")
        
        ```
        
- **완료 기준:**
    - 시퀀스 분석 결과(요약+상세) 표시 함수 및 UI 구현 완료.
    - PathRAG 쿼리 입력 및 결과 표시 UI 구현 완료.
    - 그래프 시각화 표시 UI 구현 완료.

### **WCAI-P12: Streamlit - Agent/PathRAG 연동 (최종)** (예상 소요시간: 75분)

- **설명:** UI 입력과 백엔드 로직(시퀀스 분석 Agent, PathRAG 쿼리 함수)을 최종 연동합니다.
- **수행 단계:**
    1. **필요 모듈 import (`app.py`):**
        
        ```python
        from src.agents.webtoon_agent import analyze_webtoon_sequence
        from src.analysis.pathrag_analyzer import PathRAGAnalyzer # PathRAG 분석기
        from src.graph.pathrag_integration import WebtoonPathRAG # PathRAG 인스턴스 생성용
        from PIL import Image
        import os
        
        ```
        
    2. **백엔드 객체 초기화 (캐싱 활용):** `@st.cache_resource`를 사용하여 PathRAG 인스턴스 등 무거운 객체 로드.
        
        ```python
        # app.py 상단 또는 별도 모듈
        @st.cache_resource
        def get_pathrag_analyzer():
            print("Initializing WebtoonPathRAG and PathRAGAnalyzer...")
            # API 키 로드 확인
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OPENAI_API_KEY가 설정되지 않았습니다! .env 파일을 확인하세요.")
                st.stop()
            try:
                # PathRAG 인스턴스 생성 (인덱스 빌드는 별도 스크립트로 수행 가정)
                # 실제로는 인덱스가 존재하는지 확인하는 로직 필요
                pathrag_instance = WebtoonPathRAG(api_key=api_key)
                analyzer = PathRAGAnalyzer(pathrag_instance)
                print("PathRAGAnalyzer initialized.")
                return analyzer
            except Exception as e:
                 st.error(f"PathRAG 초기화 실패: {e}")
                 st.stop()
        
        analyzer = get_pathrag_analyzer()
        
        ```
        
    3. **'웹툰 시퀀스 분석 실행' 버튼 로직 구현 (`app.py`):**
        
        ```python
        # 버튼 로직 수정
        if analyze_button_placeholder.button("웹툰 시퀀스 분석 실행", type="primary", key="analyze_sequence_btn", disabled=not can_analyze):
            with main_area:
                st.empty() # 이전 내용 지우기
                st.session_state['analysis_result'] = None # 결과 초기화
                st.session_state['pathrag_result'] = None # 다른 결과도 초기화
        
                images_to_analyze = []
                analyze_webtoon_id = "custom_upload"
                analyze_episode_ids = []
                analyze_episode_titles = []
        
                # 입력 소스 결정
                if uploaded_files:
                    try:
                        images_to_analyze = [Image.open(img) for img in uploaded_files]
                        # RGBA -> RGB 변환 추가
                        images_to_analyze = [img.convert('RGB') if img.mode == 'RGBA' else img for img in images_to_analyze]
                        analyze_episode_ids = [f"ep{i+1}" for i in range(len(uploaded_files))]
                        analyze_episode_titles = [f"Uploaded Ep {i+1}" for i in range(len(uploaded_files))]
                        print(f"Processing {len(images_to_analyze)} uploaded images.")
                    except Exception as e:
                        st.error(f"이미지 파일 처리 오류: {e}")
                        st.stop()
                elif use_sample and selected_webtoon_id and selected_episode_ids:
                    try:
                        images_to_analyze = loader.load_episode_sequence_images(selected_webtoon_id, selected_episode_ids)
                        analyze_webtoon_id = selected_webtoon_id
                        analyze_episode_ids = selected_episode_ids # 이미 로드됨
                        analyze_episode_titles = selected_episode_titles # 이미 로드됨
                        print(f"Processing sample sequence: {analyze_webtoon_id} - {analyze_episode_ids}")
                    except Exception as e:
                        st.error(f"샘플 데이터 로드 오류: {e}")
                        st.stop()
        
                # 분석 실행
                if images_to_analyze:
                    with st.spinner('웹툰 시퀀스 분석 중... (OCR, 분석, 요약)'):
                        try:
                            # 에이전트 함수 호출 (데이터 추가 전달)
                            analysis_result_data = analyze_webtoon_sequence(
                                webtoon_id=analyze_webtoon_id,
                                episode_ids=analyze_episode_ids
                                # analyze_webtoon_sequence 함수 내부에서 이미지 로딩하도록 수정하거나,
                                # 이미지를 직접 전달받도록 수정 필요. 여기선 직접 전달 가정.
                                # 만약 analyze_webtoon_sequence가 ID로 로딩한다면 images 전달 불필요.
                                # 아래 코드는 함수가 ID로 로딩한다고 가정.
                            )
                            # 결과를 Session State에 저장
                            st.session_state['analysis_result'] = analysis_result_data
                            print("Sequence analysis complete.")
                            # 결과 표시를 위해 리로드 (Streamlit 작동 방식)
                            st.rerun()
        
                        except Exception as e:
                            st.error(f"웹툰 시퀀스 분석 중 오류 발생: {e}")
                else:
                    st.warning("분석할 이미지가 없습니다.")
        
        # --- Session State에 결과 있으면 표시 ---
        if st.session_state['analysis_result']:
             display_sequence_analysis(st.session_state['analysis_result'])
        # ... (PathRAG 및 그래프 표시 로직) ...
        
        ```
        
    4. **'PathRAG에게 질문하기' 버튼 로직 구현 (`app.py`):**
        
        ```python
        # PathRAG 쿼리 버튼 로직
        if pathrag_query_button and pathrag_query_input:
            st.session_state['pathrag_query'] = pathrag_query_input
            st.session_state['pathrag_result'] = None # 결과 초기화
            st.session_state['analysis_result'] = None # 다른 결과 초기화
        
            with st.spinner("PathRAG 분석/추천 진행 중..."):
                try:
                    # TODO: 쿼리 유형(분석, 추천 등)에 따라 analyzer의 다른 함수 호출
                    # 여기서는 간단히 일반 쿼리 함수 호출 예시
                    # 실제로는 selectbox 등으로 유형 선택 후 분기 처리 필요
                    if "추천" in pathrag_query_input:
                        # 간단한 키워드 기반 추천 예시 (개선 필요)
                         result_text = analyzer.recommend_by_criteria(criteria_text=pathrag_query_input)
                    elif "분석" in pathrag_query_input or "특징" in pathrag_query_input:
                         # 패턴 분석 함수 호출 (예시)
                         result_text = analyzer.find_popular_pattern(genre=None) # 쿼리에서 장르 추출 필요
                    else: # 일반 쿼리
                         result_text = analyzer.pathrag.query(pathrag_query_input) # 기본 query 직접 호출
                         result_text = analyzer._post_process_result(result_text) # 후처리
        
                    st.session_state['pathrag_result'] = result_text
                    print("PathRAG query complete.")
                    st.rerun()
        
                except Exception as e:
                    st.error(f"PathRAG 쿼리 중 오류 발생: {e}")
                    st.session_state['pathrag_result'] = f"오류 발생: {e}" # 오류 메시지 표시
        # --- PathRAG 결과 표시 ---
        if st.session_state['pathrag_result']:
             display_pathrag_results(st.session_state['pathrag_query'], st.session_state['pathrag_result'])
        
        ```
        
    5. **그래프 시각화 연동:** `display_graph_visualization()` 함수 호출 부분을 활성화. 해당 HTML 파일이 생성되어 있는지 확인.
- **완료 기준:**
    - UI 입력 -> 백엔드 로직(Agent/PathRAG) 호출 -> 결과 표시 전체 흐름 연동 완료.
    - Streamlit Session State를 활용한 결과 유지 및 표시.
    - 백엔드 호출 시 로딩 스피너 표시 및 오류 처리 구현.

### **WCAI-P13: End-to-End 통합 테스트 (다중 에피소드 + PathRAG)** (예상 소요시간: 90분)

- **설명:** 전체 시스템의 통합 테스트를 수행하고 디버깅합니다. (시퀀스 분석 및 PathRAG 기능 포함)
- **수행 단계:**
    1. **통합 테스트 시나리오 상세화 (`tests/test_scenarios.md`):**
        - 샘플 시퀀스 선택 -> 분석 실행 -> 시퀀스 요약, 에피소드별 OCR/분석 결과 확인.
        - 다중 이미지 업로드 -> 분석 실행 -> 결과 확인.
        - PathRAG 쿼리 입력 (유사 웹툰, 특정 조건 추천, 패턴 분석 등) -> 반환된 텍스트 결과의 관련성, 논리성 확인.
        - 그래프 시각화 탭/섹션 확인.
        - 경계 조건 테스트 (이미지 0개/1개 업로드, 빈 쿼리 입력 등).
        - 오류 처리 테스트 (잘못된 API 키, 분석 중 타임아웃 등).
    2. **시나리오 기반 테스트 수행:** Streamlit 앱 직접 사용하며 테스트. 브라우저 개발자 도구 콘솔 및 서버 로그 확인.
    3. **버그 식별 및 수정:** 데이터 흐름, 상태 관리(Session State), API 호출 오류, UI 표시 오류, PathRAG 쿼리/응답 문제 등 전 영역 디버깅. 특히 긴 이미지 처리 시간 및 메모리 사용량 관찰.
    4. **결과 기록 (`tests/test_results.md`):** 각 시나리오별 성공/실패, 발견된 이슈, 해결 내용, 성능(응답 시간) 등 기록.
- **완료 기준:** 정의된 통합 테스트 시나리오 실행 완료. 식별된 주요 버그 수정 완료. 시스템 안정성 확보.

### **WCAI-P14: 최종 README 및 데모 준비 (다중 에피소드 + PathRAG)** (예상 소요시간: 60분)

- **설명:** 최종 기능을 모두 반영하여 문서를 업데이트하고 데모를 준비합니다.
- **수행 단계:**
    1. [**README.md](http://readme.md/) 최종 완성:**
        - 프로젝트 목표, 최종 구현 기능 (시퀀스 분석, PathRAG 기반 분석/추천 상세 설명) 명확화.
        - 설치 (PathRAG 포함), API 키 설정, **데이터 준비 가이드** (샘플 구조, 이미지 형식/캡처 가이드), **그래프/인덱스 빌드 스크립트 실행 방법** 명시.
        - **UI 사용법 최종 안내** (시퀀스 분석 실행, PathRAG 쿼리 방법 등 스크린샷 포함).
        - 프로젝트 아키텍처(데이터 흐름도 포함) 최종본 요약.
        - **결과 예시:** 실제 분석 결과 스크린샷 (시퀀스 요약, PathRAG 추천/분석 답변 예시).
        - **제한 사항:** OCR 정확도 한계, PathRAG 쿼리 이해도, 분석/쿼리 시간, 비용 문제, 샘플 데이터 규모 한계 등 솔직하게 명시.
        - 향후 개선 방향 구체화 (URL 입력 지원, 모델 파인튜닝, 그래프 관계 강화 등).
        - 라이선스 명시.
    2. **데모 시나리오 최종 확정:** 짧은 시간 내에 프로젝트의 핵심 가치(시퀀스 분석 능력, PathRAG를 통한 인사이트 도출)를 효과적으로 보여줄 수 있는 시나리오 구성.
    3. **데모용 스크린샷/GIF 최종 준비:** 완성된 UI 및 인상적인 결과 화면 자료 준비.
- **완료 기준:** 모든 기능과 사용법이 상세히 기술된 최종 README 완성. 데모 시연 자료 준비 완료.

### **WCAI-P15: 최종 코드 정리 및 제출 준비** (예상 소요시간: 30분)

- **설명:** 코드의 가독성을 높이고 최종 버전을 GitHub에 푸시하여 제출 준비를 완료합니다.
- **수행 단계:**
    1. **코드 주석 추가/검토:** 모든 주요 함수, 클래스, 복잡한 로직에 docstring 및 설명 주석 추가/최종 검토.
    2. **코드 포맷팅:** Black, Flake8 등 사용하여 코드 스타일 일관성 유지.
    3. **최종 리뷰:** 불필요한 코드, print문, 임시 파일, 하드코딩된 값 등 제거 및 정리. 설정값 분리 확인 (config 등).
    4. **최종 테스트:** 마지막으로 전체 애플리케이션이 README의 지침대로 오류 없이 실행되는지 최종 확인.
    5. **최종 커밋 및 푸시:** 모든 변경사항을 GitHub 저장소에 커밋하고 푸시. 버전 태그(e.g., `v1.0-final-poc`) 생성 고려.
        
        ```bash
        git status # 변경사항 확인
        git add .
        git commit -m "Finalize project: Complete sequence analysis, PathRAG integration, UI, and documentation"
        git tag v1.0-final-poc # 버전 태그 생성
        git push origin main --tags # main 브랜치와 태그 푸시
        
        ```
        
- **완료 기준:**
    - 코드 정리 및 주석 완료.
    - 최종 버전 GitHub 푸시 완료.
    - 포트폴리오로 제출 가능한 상태.

---