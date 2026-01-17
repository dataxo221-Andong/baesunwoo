import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAHEiUACkvXTxT-I_r60dmLIDHWyWP5wmI")  # 보안 처리 권장

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat_session = model.start_chat()
    READY = True
except Exception as e:
    print("[❌ Gemini 초기화 실패]", e)
    chat_session = None
    READY = False

def get_gemini_response(prompt: str) -> str:
    if not READY or chat_session is None:
        return "[오류] Gemini가 준비되지 않았습니다."
    try:
        response = chat_session.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[오류] Gemini 응답 실패: {e}"
