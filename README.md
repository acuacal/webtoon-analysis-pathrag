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

