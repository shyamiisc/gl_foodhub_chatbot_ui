import os
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from dataclasses import dataclass
import time
import re
from typing import Generator, List, Optional
from pathlib import Path


@dataclass
class BotResponse:
    text: str


class RuleBasedBot:
    def reply(self, message: str) -> BotResponse:
        text = self._generate_response(message)
        return BotResponse(text=text)

    def _generate_response(self, message: str) -> str:
        normalized = (message or "").strip().lower()

        if any(greet in normalized for greet in ["hello", "hi", "hey"]):
            return "Hello! How can I help you today?"
        if "help" in normalized:
            return (
                "I can answer a few built-in prompts: try 'time', 'date', or 'joke'. "
                "You can also ask me general questions if OpenAI is enabled."
            )
        if "time" in normalized:
            return time.strftime("The current time is %H:%M:%S")
        if "date" in normalized:
            return time.strftime("Today's date is %Y-%m-%d")
        if "joke" in normalized:
            return "Why do programmers prefer dark mode? Because light attracts bugs."

        return (
            "I'm a simple rule-based bot right now. Type 'help' to see what I can do, "
            "or switch to the OpenAI provider from the sidebar if you have an API key."
        )


class OpenAIBot:
    def __init__(
        self,
        client: "OpenAI",
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature

    def stream(self, user_message: str) -> Generator[str, None, None]:
        messages: List[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_message})

        # Stream chat completion tokens
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )

        accumulated = ""
        for chunk in response:
            try:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
            except Exception:
                content = None

            if content:
                accumulated += content
                yield content

        # Final yield not necessary; Streamlit's write_stream returns accumulated text


def get_openai_bot(
    api_key: Optional[str],
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
) -> OpenAIBot:
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. Set it to enable OpenAI."
        )

    try:
        from openai import OpenAI
    except Exception as import_error:  # pragma: no cover
        raise RuntimeError(
            "The 'openai' package is not installed. Add it to requirements to enable OpenAI."
        ) from import_error

    client = OpenAI(api_key=api_key)
    return OpenAIBot(client=client, model=model, system_prompt=system_prompt, temperature=temperature)



load_dotenv()


st.set_page_config(page_title="Chat Bot", page_icon="💬", initial_sidebar_state="expanded")


def get_or_init_session_messages() -> List[Dict[str, str]]:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    return st.session_state["messages"]


def load_default_hero_bytes() -> Optional[bytes]:
    """Load hero image bytes from chatbot/assets, preferring hero.jpg then fallbacks."""
    assets_dir = Path(__file__).parent / "assets"
    candidate_names = [
        "hero.jpg",
        "hero.jpeg",
        "hero.png",
        "hero.webp",
    ]
    try:
        for name in candidate_names:
            path = assets_dir / name
            if path.exists():
                return path.read_bytes()
    except Exception:
        # Ignore image load failures; app should continue to function
        return None
    return None


def render_sidebar() -> Dict[str, Optional[str]]:
    # Minimal sidebar with Logout button and default provider settings
    with st.sidebar:
        st.header("Settings")
        if st.button("Logout", key="logout_sidebar"):
            st.session_state.pop("user_profile", None)
            st.session_state["messages"] = []
            st.rerun()

        return {
            "provider": "Rule-based",
            "model": None,
            "system_prompt": None,
            "temperature": str(0.7),
        }


def render_message(role: str, content: str) -> None:
    """Render a single message aligned: user on right, assistant on left."""
    left_col, right_col = st.columns([1, 1])
    if role == "user":
        with right_col:
            with st.chat_message("user"):
                st.markdown(content)
    else:
        with left_col:
            with st.chat_message("assistant"):
                st.markdown(content)


def render_stream(role: str, stream: Generator[str, None, None]) -> str:
    """Render a streamed response aligned by role and return the accumulated text."""
    left_col, right_col = st.columns([1, 1])
    target = right_col if role == "user" else left_col
    with target:
        with st.chat_message(role):
            full_text = st.write_stream(stream)
    return full_text


def _status_bar_html(label: str, state: str) -> str:
    icon = "…" if state == "running" else ("✅" if state == "complete" else "⚠️")
    return f"""
<div id=\"chat-status-bar\" role=\"status\">{icon} {label}</div>
"""


def get_user_profile() -> Optional[Dict[str, str]]:
    user = st.session_state.get("user_profile")
    if isinstance(user, dict) and user.get("name") and user.get("user_id"):
        return {"name": str(user["name"]).strip(), "user_id": str(user["user_id"]).strip()}
    return None


def validate_user_details(name: str, user_id: str) -> Optional[str]:
    """Validate user's display name and customer ID.

    Returns an error message string if invalid, otherwise None.
    Rules:
    - Name: 2-80 chars; letters, spaces, hyphens, apostrophes; starts with a letter
    - Customer ID: 3-32 chars; alphanumeric plus . _ -; must start alphanumeric
    """
    name_val = (name or "").strip()
    id_val = (user_id or "").strip()

    if not name_val or not id_val:
        return "Please provide both name and Customer ID."

    if len(name_val) < 2 or len(name_val) > 80:
        return "Name must be between 2 and 80 characters."

    if not re.fullmatch(r"[A-Za-z][A-Za-z\s\-']{1,79}", name_val):
        return "Name can include letters, spaces, hyphens, apostrophes, and must start with a letter."

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,31}", id_val):
        return "Customer ID must be 3-32 chars, alphanumeric plus . _ -, starting with a letter/number."

    return None


def render_onboarding() -> None:
    st.title("Food Hub ")
    hero_bytes = load_default_hero_bytes()
    if hero_bytes is not None:
        st.image(hero_bytes, use_column_width=True)
    st.subheader("Welcome! Please enter your details to continue")
    with st.form("onboarding"):
        name = st.text_input("Your name")
        user_id = st.text_input("Customer ID")
        submitted = st.form_submit_button("Sign In")
        if submitted:
            error = validate_user_details(name, user_id)
            if error:
                st.error(error)
            else:
                name_val = (name or "").strip()
                id_val = (user_id or "").strip()
                st.session_state["user_profile"] = {"name": name_val, "user_id": id_val}
                st.session_state["welcome_message"] = f"Welcome, {name_val}!"
                st.rerun()


def main() -> None:
    # If user not onboarded, show onboarding and exit early
    user = get_user_profile()
    if user is None:
        render_onboarding()
        return

    # Once onboarded, render hero centered, caption, settings, and chat
    hero_bytes = load_default_hero_bytes()
    if hero_bytes is not None:
        _left, _center, _right = st.columns([1, 2, 1])
        with _center:
            st.image(hero_bytes, use_column_width=True)

    st.title("Food Hub")

    # Show one-time welcome message inside the chat window area
    welcome_msg = st.session_state.pop("welcome_message", None)
    if welcome_msg:
        st.session_state.setdefault("messages", []).append({
            "role": "assistant",
            "content": welcome_msg,
        })

    settings = render_sidebar()

    messages = get_or_init_session_messages()

    # Render history (user right, assistant left)
    for msg in messages:
        render_message(msg["role"], msg["content"])

    # Chat input
    user_input = st.chat_input("Message the assistant…")
    # Progress status placeholder anchored under the input prompt UI using HTML
    status_placeholder = st.container()
    status_placeholder.markdown(
        """
<style>
  #chat-status-bar { margin-top: 6px; color: #6b7280; font-size: 0.9rem; }
</style>
""",
        unsafe_allow_html=True,
    )
    if user_input:
        messages.append({"role": "user", "content": user_input})
        render_message("user", user_input)

        provider = settings.get("provider") or "Rule-based"
        status_area = status_placeholder.empty()
        status_area.markdown(
            _status_bar_html("… working on your query", state="running"),
            unsafe_allow_html=True,
        )
        if provider == "Rule-based":
            bot = RuleBasedBot()
            bot_response: BotResponse = bot.reply(user_input)
            render_message("assistant", bot_response.text)
            messages.append({"role": "assistant", "content": bot_response.text})
            try:
                status_area.markdown(
                    _status_bar_html("Response ready", state="complete"),
                    unsafe_allow_html=True,
                )
            except Exception:
                pass
        else:
            try:
                openai_bot = get_openai_bot(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model=settings.get("model") or "gpt-4o-mini",
                    system_prompt=settings.get("system_prompt") or "You are a helpful assistant.",
                    temperature=float(settings.get("temperature") or 0.7),
                )
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"OpenAI is not configured: {e}")
                messages.append({"role": "assistant", "content": f"OpenAI is not configured: {e}"})
                try:
                    status_area.markdown(
                        _status_bar_html("Failed", state="error"),
                        unsafe_allow_html=True,
                    )
                except Exception:
                    pass
            else:
                stream = openai_bot.stream(user_input)
                full_text = render_stream("assistant", stream)
                messages.append({"role": "assistant", "content": full_text})
                try:
                    status_area.markdown(
                        _status_bar_html("Response ready", state="complete"),
                        unsafe_allow_html=True,
                    )
                except Exception:
                    pass

    # Removed bottom Logout button; now placed at top-right


if __name__ == "__main__":
    main()


