import streamlit as st

from rag.ingest import ingest_pdf
from rag.embeddings import get_embedder
from rag.vector_store import build_vector_store
from agent.agent import build_agent
from agent.tools import initialize_tools, set_retrieval_k
from llm.groq_llm import get_llm
from typing import List
import json

st.set_page_config(page_title="Agentic Research Analyst", layout="wide")
st.title("📄 Agentic AI Research & Insight Agent")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    # 1️⃣ Ingest PDF → chunks
    chunks = ingest_pdf(uploaded_file)
    st.success(f"Document ingested. Total chunks: {len(chunks)}")

    # 2️⃣ Build embeddings
    embedder = get_embedder()

    # 3️⃣ Build LangChain vector store (FAISS)
    vector_store = build_vector_store(chunks, embedder)

    # 4️⃣ Initialize tools with vector store + LLM
    llm = get_llm()
    initialize_tools(vector_store, llm)

    # allow user to tune retrieval depth (helps quality vs cost)
    retrieval_k = st.slider("Chunks to retrieve for context", 1, 8, 4, help="Increase to pass more context to the agent (higher token use)")

    # 5️⃣ Build agent
    agent = build_agent()

    # tell tools the desired retrieval_k
    set_retrieval_k(retrieval_k)

    # 6️⃣ User input
    user_goal = st.text_input(
        "Enter your goal",
        placeholder="e.g. Summarize the document and give risks and recommendations"
    )

    if st.button("Run Agent"):
        with st.spinner("Agent reasoning..."):
            response = agent.invoke({
                "messages": [
                    {"role": "user", "content": user_goal}
                ]
            })
        def _get_msg_fields(msg):
            # Normalize message-like objects/dicts to (name, content)
            name = None
            content = None
            if isinstance(msg, dict):
                name = msg.get("name") or msg.get("tool") or msg.get("type")
                content = msg.get("content") or msg.get("text") or msg.get("message")
            else:
                name = getattr(msg, "name", None)
                content = getattr(msg, "content", None) or getattr(msg, "text", None) or getattr(msg, "message", None)
            if callable(content):
                try:
                    content = content()
                except Exception:
                    content = str(content)
            return name, content

        def format_agent_response(resp):
            # Attempt to build structured sections from agent/tool messages
            summary = None
            action_items = None
            retrieved = []
            final_answer = None

            # Messages may appear in resp['messages'] or resp.messages
            msgs = None
            if isinstance(resp, dict) and "messages" in resp:
                msgs = resp["messages"]
            elif hasattr(resp, "messages"):
                msgs = getattr(resp, "messages")
            elif isinstance(resp, dict):
                # fallback: include top-level tool messages if present
                msgs = []
                for v in resp.values():
                    if isinstance(v, list):
                        msgs.extend(v)

            if not msgs:
                return str(resp)

            missing_context_flag = False

            for m in msgs:
                name, content = _get_msg_fields(m)
                if not content:
                    continue

                lc = content.lower() if isinstance(content, str) else ""

                # retrieve_context outputs usually contain chunk markers
                if (name and "retrieve" in str(name).lower()) or "[chunk" in lc:
                    retrieved.append(content)
                    continue

                # summarize_context likely includes 'summary' or is a paragraph
                if "no document context" in lc or "you haven't provided" in lc or "please run retrieve_context" in lc:
                    missing_context_flag = True

                if (name and "summarize" in str(name).lower()) or "summary" in lc or "here's a clear" in lc:
                    summary = content
                    continue

                # extract_action_items usually contains bullets
                if (name and "extract" in str(name).lower()) or any(tok in content for tok in ["•", "- ", "1.", "• "]):
                    action_items = content
                    continue

                # AI final answer may include business opportunities / risks
                if (name and "ai" in str(name).lower()) or any(k in lc for k in ["business opportunities", "business risks", "opportunities", "risks"]):
                    final_answer = content

            # If the model produced a placeholder-like summary (e.g. describing
            # that context was retrieved but not providing it), treat it as
            # missing and synthesize from retrieved chunks instead.
            placeholder_indicators = [
                "provided context appears",
                "you haven't provided",
                "no document context",
                "document context retrieved",
                "please provide the document",
                "please run retrieve_context",
            ]
            if summary and any(p in summary.lower() for p in placeholder_indicators):
                missing_context_flag = True
                summary = None

            # If model/tool failed to summarize but we have retrieved chunks,
            # synthesize a summary and/or action items by calling the LLM directly.
            if missing_context_flag and not summary and retrieved:
                try:
                    # Use top-N retrieved chunks (based on slider) and truncate each to limit prompt size
                    def _short(s, n=1200):
                        if not s:
                            return s
                        return s[:n] + ("..." if len(s) > n else "")

                    retrieved_text = "\n\n---\n\n".join(_short(r) for r in retrieved[:retrieval_k])
                    prompt = f"""
Summarize the following document context clearly and concisely:

{retrieved_text}
"""
                    synth = llm.invoke(prompt)
                    summary = getattr(synth, "content", None) or getattr(synth, "text", None) or str(synth)
                except Exception:
                    summary = None

            if missing_context_flag and not action_items and retrieved:
                try:
                    retrieved_text = "\n\n---\n\n".join(_short(r) for r in retrieved[:2])
                    prompt = f"""
From the following document context, extract 5–7 clear, actionable insights.
Present them as bullet points. Only use the provided content.

{retrieved_text}
"""
                    synth_ai = llm.invoke(prompt)
                    action_items = getattr(synth_ai, "content", None) or getattr(synth_ai, "text", None) or str(synth_ai)
                except Exception:
                    action_items = None

            # Build formatted output
            parts = []
            if summary:
                parts.append("**Summary**\n" + summary)

            if retrieved and not summary:
                # if no summary but we have retrieved chunks, show a short preview
                parts.append("**Retrieved Context (preview)**\n" + "\n\n---\n\n".join(retrieved[:3]))

            if final_answer:
                # try to split final answer into Opportunities and Risks
                parts.append("**Agent Analysis**\n" + final_answer)

            if action_items:
                parts.append("**Actionable Insights**\n" + action_items)

            if not parts:
                return str(resp)

            # join with spacing for Streamlit display
            return "\n\n".join(parts)

        formatted = format_agent_response(response)

        # If the user's goal requests specific deliverables (e.g. "top 3 risks",
        # "mitigation strategies", "6 months plan") and the agent output does
        # not contain them, synthesize a strict RAG answer directly from the
        # vector store + LLM to guarantee the required structure.
        def _needs_strict_synthesis(goal: str, formatted_text: str) -> bool:
            goal_l = (goal or "").lower()
            triggers = ["top 3 risk", "top 3 risks", "mitigation", "mitigation strategy", "6 months", "next 6 months"]
            if not any(t in goal_l for t in triggers):
                return False
            # if formatted already contains clear structured sections, skip
            has_risks = any(k in formatted_text.lower() for k in ["risk", "risks", "mitigation", "mitigat"]) 
            return not has_risks

        def synthesize_strict(goal: str, vector_store, llm, k: int) -> str:
            # retrieve top-k chunks and truncate
            docs = vector_store.similarity_search(goal, k=k)
            def _short(s: str, n=1200):
                if not s:
                    return s
                return s[:n] + ("..." if len(s) > n else "")
            context = "\n\n---\n\n".join(_short(getattr(d, "page_content", str(d))) for d in docs)

            # Request JSON output to make validation and formatting deterministic.
            prompt = f"""
You are a Chief Data Officer. Using ONLY the CONTEXT below, do NOT hallucinate.

Respond ONLY in JSON with the following keys: "summary" (string),
"top_risks" (list of objects with "risk" and "evidence"), and
"mitigations" (list of objects with "risk" and "actions", where "actions" is a list of {"action","owner","timeline"}).

If an item is not present in the context, set its value to the string "Not found in document" or an empty list as appropriate.

CONTEXT:
{context}
"""

            resp = llm.invoke(prompt)
            text = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)

            def _parse_and_validate(text: str):
                try:
                    payload = json.loads(text)
                except Exception:
                    return None, "invalid_json"

                if not isinstance(payload, dict):
                    return None, "not_object"
                if "summary" not in payload or "top_risks" not in payload or "mitigations" not in payload:
                    return None, "missing_keys"

                risks = payload.get("top_risks") or []
                if not isinstance(risks, list) or len(risks) == 0:
                    return None, "no_risks"
                for r in risks:
                    if not isinstance(r, dict) or not r.get("evidence"):
                        return None, "no_evidence"

                return payload, "ok"

            parsed, status = _parse_and_validate(text)
            if parsed is None:
                # Retry with stricter instruction
                prompt2 = f"""
You previously returned invalid or unrelated output. Using ONLY the CONTEXT below, produce valid JSON with keys: summary, top_risks, mitigations.
Do NOT invent items not present in the context. Each risk must include a short "evidence" string quoting the context.

CONTEXT:
{context}
"""
                resp2 = llm.invoke(prompt2)
                text2 = getattr(resp2, "content", None) or getattr(resp2, "text", None) or str(resp2)
                parsed, status = _parse_and_validate(text2)
                if parsed is None:
                    return text2

            # Format into markdown for display
            out_lines = []
            out_lines.append("**Summary**\n" + parsed["summary"]) 
            out_lines.append("**Top 3 Risks (with evidence)**")
            for i, r in enumerate(parsed.get("top_risks", []), 1):
                out_lines.append(f"{i}. {r.get('risk')} — Evidence: {r.get('evidence')}")

            out_lines.append("**Mitigation Strategies**")
            for m in parsed.get("mitigations", []):
                risk = m.get("risk")
                out_lines.append(f"- For: {risk}")
                for a in m.get("actions", []):
                    out_lines.append(f"  - {a.get('action')} (Owner: {a.get('owner')}, Timeline: {a.get('timeline')})")

            return "\n\n".join(out_lines)

        if _needs_strict_synthesis(user_goal, formatted):
            try:
                strict_out = synthesize_strict(user_goal, vector_store, llm, retrieval_k)
                st.subheader("📌 Agent Output (strict RAG synthesis)")
                st.markdown(strict_out)
            except Exception as e:
                st.subheader("📌 Agent Output")
                st.markdown(formatted)
                st.error(f"Synthesis failed: {e}")
        else:
            st.subheader("📌 Agent Output")
            # display as markdown for better readability
            st.markdown(formatted)
