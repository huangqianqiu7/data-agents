# RAG Memory HTML Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a self-contained interactive HTML learning page explaining the current `data_agent_langchain` architecture and the recommended Memory/RAG evolution path.

**Architecture:** The output is a static HTML page with embedded CSS and JavaScript. It does not modify runtime code, does not require a build step, and uses only documented project architecture from `src/data_agent_langchain` and the existing memory proposal.

**Tech Stack:** HTML5, CSS3, vanilla JavaScript.

---

## File Structure

- Create: `src/计划/RAG记忆架构演示/architecture-rag-memory-demo.html`
  - Self-contained interactive architecture demo.
- Create: `src/计划/RAG记忆架构演示/2026-05-12-rag-memory-html-demo-design.md`
  - Approved design summary.
- Create: `src/计划/RAG记忆架构演示/2026-05-12-rag-memory-html-demo-plan.md`
  - This implementation plan.

## Task 1: Create the Design Record

- [x] **Step 1: Write the design summary**

Create `src/计划/RAG记忆架构演示/2026-05-12-rag-memory-html-demo-design.md` with the project context, page goals, interaction model, and recommended Memory/RAG strategy.

- [x] **Step 2: Confirm the file is written**

Expected result: the design file exists under `src/计划/RAG记忆架构演示/`.

## Task 2: Create the Interactive HTML Demo

- [ ] **Step 1: Write the static HTML file**

Create `src/计划/RAG记忆架构演示/architecture-rag-memory-demo.html` with:

- Stage selector: Current / Memory MVP / RAG Full.
- Clickable architecture cards.
- Detailed side panel for selected module.
- Data flow diagrams built with HTML/CSS.
- Testing and risk checklist.

- [ ] **Step 2: Verify the HTML file exists**

Expected result: `architecture-rag-memory-demo.html` exists and is non-empty.

- [ ] **Step 3: Verify important text markers**

Expected markers:

- `data_agent_langchain`
- `RunState`
- `execution_subgraph`
- `Memory MVP`
- `Corpus RAG`
- `tool_node`
- `planner_node`
- `model_node`
- `prompts.py`

## Task 3: Provide Opening Instructions

- [ ] **Step 1: Tell the user the file path**

Provide the exact path:

```text
c:\Users\18155\learn python\Agent\kddcup2026-data-agents-starter-kit-master\src\计划\RAG记忆架构演示\architecture-rag-memory-demo.html
```

- [ ] **Step 2: Offer browser preview if a server is requested**

If the user wants a browser preview, start a local static file server from the project root and provide a browser preview URL. Do not start a server unless requested.

## Self-Review

- Spec coverage: The plan creates the requested folder, stores the design record, and creates the interactive HTML demo.
- Placeholder scan: No placeholder content is required for implementation; all expected sections are specified.
- Type consistency: The plan uses exact file paths and existing project module names.
