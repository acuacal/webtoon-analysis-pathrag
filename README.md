# Webtoon Analysis & Recommendation Agent with PathRAG (PoC)

## 프로젝트 개요

이 프로젝트는 웹툰 콘텐츠(이미지+텍스트)를 자동으로 이해하고 분석하는 AI 에이전트의 Proof-of-Concept(PoC)입니다. LangGraph 기반 에이전트를 사용하여 개별 웹툰의 내용을 분석하고, **PathRAG 기술을 통합하여 웹툰 간의 복잡한 관계성을 분석하고 개인화된 추천 기능을 제공하는 것**을 목표로 합니다.

### PoC 범위

* **개별 웹툰 분석 (WCAI Agent):**
    * 입력: 사용자가 업로드한 웹툰 에피소드의 이미지 파일들.
    * 처리: MLLM을 이용한 이미지 내 텍스트 추출(OCR) 및 시각적 내용 동시 분석.
    * 출력: 추출된 텍스트, 이미지별 시각 분석 결과, LLM 기반 에피소드 요약.
* **관계성 분석 및 추천 (PathRAG 통합):**
    * **그래프 구축:** 분석된 웹툰 정보(메타데이터, 추출된 특징)를 기반으로 지식 그래프 생성 (웹툰, 작가, 장르, 태그 등 노드 및 관계 엣지).
    * **PathRAG 인덱싱:** 웹툰 정보를 PathRAG에 인덱싱.
    * **분석/추천 쿼리 처리:** PathRAG를 활용하여 자연어 쿼리에 대한 답변 생성.
        * 유사 웹툰 추천 (예: "이 웹툰과 비슷한 그림체의 다른 작품 찾아줘").
        * 인기 웹툰 패턴 분석 (예: "액션 판타지 장르 인기 웹툰들의 공통점은?").
        * 크리에이터 전략 분석 (예: "작가 A의 작품 스타일 변화는?").
        * 조건 기반 추천 (예: "로맨스 없고 그림체 좋은 웹툰 추천").
* **UI:** Streamlit 기반 웹 인터페이스 제공 (개별 분석 + PathRAG 쿼리 기능).

### 주요 기능 목록

* 웹툰 이미지 기반 자동 텍스트 추출 (OCR) 및 시각 분석.
* 에피소드 내용 자동 요약.
* 웹툰 지식 그래프 자동 구축 및 시각화.
* PathRAG를 이용한 문맥 기반 웹툰 정보 검색.
* 유사 웹툰 추천 (내용, 스타일, 장르 등 기반).
* 웹툰 트렌드 및 패턴 분석 지원.

### 기술 스택

* **AI Workflow:** LangChain & LangGraph
* **Core Models:** OpenAI GPT-4o (MLLM/LLM) 또는 Gemini Pro Vision/1.5 Pro
* **Graph RAG:** **PathRAG (BUPT-GAMMA)**
* **Graph Handling:** **NetworkX**
* **Web Interface:** Streamlit
* **Development:** Python
* **(선택) Graph Visualization:** PyVis

---
*(README의 나머지 부분: 설치 방법, 사용 방법, 프로젝트 구조 등은 이후 단계에서 구체화됩니다.)*