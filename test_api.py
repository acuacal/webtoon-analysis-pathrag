import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from google import genai
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
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Gemini 클라이언트 초기화
client = genai.Client(api_key=api_key)  # New client initialization

# 간단한 Gemini API 호출 테스트
response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents="Hello, are you working properly?"
)

print("Gemini API Test Response:", response.text)
print("OpenAI API connection test successful!")