import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# .env 파일 로드
load_dotenv()

# Openai API 키 확인
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# 간단한 OpenAI API 호출 테스트
llm = ChatOpenAI(model="gpt-3.5-turbo")
response = llm.invoke("Hello, are you working properly?")
print("OpenAI API Test Response:", response.content)
print("OpenAI API connection test successful!")


# Gemini API 키 확인
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# 간단한 Gemini 초기화
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash")

# 간단한 테스트 실행
response = llm_gemini.invoke("Hello, are you working properly?")

print("Gemini API Test Response:", response.content)
print("OpenAI API connection test successful!")