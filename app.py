import streamlit as st
from huggingface_hub import InferenceClient
from gtts import gTTS
from io import BytesIO
import base64
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    st.error("Missing Hugging Face API key")
    st.stop()


@st.cache_resource
def load_client():
    return InferenceClient(
        model="HuggingFaceH4/zephyr-7b-beta",  
        token=hf_token
    )

client = load_client()


st.set_page_config(page_title="100x Interview Voicebot", layout="centered")
st.title("100x Voicebot")

st.markdown("""
**Example questions:**
- What should we know about your life story?
- What's your #1 superpower?
- What are the top 3 areas you'd like to grow in?
- What misconception do your coworkers have about you?
- How do you push your boundaries and limits?
""")

#llm response
def get_response(user_input: str):
    """Generate a clean, short, and natural interview-style response."""
    import re
    try:
        messages = [
            {
                "role": "system",
                "content": (
                     "You are a friendly, thoughtful person having a normal conversation with another human. "
            "Answer their question in first person as if you are sharing your own thoughts or experiences. "
            "Do NOT mention being an AI, chatbot, or program. "
            "Do NOT ask the user any questions. "
            "Keep the answer under 100 words, self-contained, and end with a complete sentence. "
            "Avoid phrases like 'I am not capable' or 'as an AI'. "
            "End naturally with correct punctuation.\n\n"
            "Example:\n"
            "Q: What's your #1 superpower?\n"
            "A: My superpower is adaptability. I stay calm under pressure, learn quickly, and find creative ways to solve challenges."
        )
            },
            {"role": "user", "content": user_input},
        ]

        response = client.chat_completion(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=messages,
            temperature=0.2,     
            max_tokens=120       
        )

        reply = response.choices[0].message["content"]
        reply = re.sub(r"\[/?(ASSISTANT|ASS|INST|USER|HR).*?\]", "", reply)
        reply = re.sub(r"\s{2,}", " ", reply).strip()


       
        return reply

    except Exception as e:
        return f"Error contacting Hugging Face Inference API: {e}"

#gtts
def text_to_speech(text: str):
    """Convert reply to speech using gTTS."""
    tts = gTTS(text)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio_bytes = fp.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'


user_input = st.text_input("Type your interview question:")

if st.button("Ask Bot"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            reply = get_response(user_input)
            st.markdown(f"**You asked:** {user_input}")
            st.markdown(text_to_speech(reply), unsafe_allow_html=True)
    else:
        st.warning("Please type a question first.")

st.caption("Built for 100x")
