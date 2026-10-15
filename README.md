# llm-failure-github-collection

Data collection for the study *How LLM users explain factual errors, memory lapses and instruction-following failures: Expressed mental models and expectation violations*.

The code collects public GitHub Issues and Discussions from 30 repositories, saves them as full-text datasets (Steps 1 to 7), and writes cleaned, coding-ready copies (Step 8).

## Files

| Path | Purpose |
| --- | --- |
| `LLM_Failure_GitHub_Data_Collection_v3.ipynb` | Google Colab notebook (embeds the module below) |
| `llm_failures_collect.py` | Collection and cleaning module |
| `tests/` | Offline tests against a simulated GitHub API |
| `requirements.txt` | Python dependencies |

## Scope

**Observation window.** Threads created from 1 September 2023 to 30 September 2026, inclusive. Messages posted after 30 September 2026, 23:59:59 UTC are kept and flagged `outside_window`.

**Repositories.**

| Category | Repositories |
| --- | --- |
| General chat interfaces and personal assistants | `open-webui/open-webui`, `LibreChat-AI/LibreChat`, `lobehub/lobehub`, `janhq/jan`, `CherryHQ/cherry-studio` |
| Document, knowledge and search assistants | `Mintplex-Labs/anything-llm`, `nomic-ai/gpt4all`, `khoj-ai/khoj`, `infiniflow/ragflow`, `onyx-dot-app/onyx` |
| Coding assistants and action-taking agents | `openai/codex`, `cline/cline`, `Aider-AI/aider`, `continuedev/continue`, `OpenHands/OpenHands` |
| Role-play and local-model environments | `SillyTavern/SillyTavern`, `kwaroran/Risuai`, `LostRuins/koboldcpp`, `oobabooga/textgen`, `ollama/ollama` |
| Agent orchestration and LLM application frameworks | `microsoft/autogen`, `crewAIInc/crewAI`, `langchain-ai/langgraph`, `FlowiseAI/Flowise`, `langflow-ai/langflow` |
| Agent memory and model-access infrastructure | `letta-ai/letta`, `mem0ai/mem0`, `getzep/graphiti`, `BerriAI/litellm`, `mostlygeek/llama-swap` |

## Method

- **Lexicon.** 187 search terms (79 core, 108 exploratory) covering factual errors, memory lapses, instruction-following failures and folk explanatory idioms.
- **Issues.** Terms are packed into OR-bundles within GitHub's search limits (five boolean operators, 256 characters) and searched in titles, bodies and comments (1,020 base queries). Date windows over the 1,000-result cap are split. Every unique Issue is retrieved with all comments.
- **Discussions.** One core term per query in repositories with Discussions enabled. Every matching Discussion is retrieved with all comments and nested replies.
- **Retention.** No retrieved thread is discarded during collection.
- **Reliability.** Retries, rate-limit handling, request logging, deduplication and checkpointing, so an interrupted run resumes where it stopped.

## Running in Google Colab

1. Upload the notebook at colab.research.google.com (File, Upload notebook).
2. Add a GitHub personal access token as the Colab secret `GITHUB_TOKEN` (key icon, notebook access on).
3. Run the setup cells 1 to 4, approving the Google Drive pop-up.
4. Run Step 1, then Steps 3 to 8 in order (Step 2 is optional).
5. If Colab disconnects, re-run cells 1 to 4 and Step 1, then the interrupted step. Completed work is skipped.

Data are saved to `MyDrive/llm_failures_data_v3`.

## Outputs

**Collected datasets (Steps 6 and 7)**

| File | Content |
| --- | --- |
| `corpus_threads.csv` / `.jsonl` | One row per Issue or Discussion, with metadata and retrieval status |
| `corpus_messages.csv` / `.jsonl` | One row per opening post, comment or reply, with full text, author, timestamps and reply links |
| `repository_metadata.csv` | Resolved repository paths and features |
| `search_log.csv` | Every executed query |
| `collection_summary.json`, `collection_by_repository.csv` | Counts of threads, messages and retrieval status |
| `corpus_threads_preview.xlsx` | Excel preview (opening posts shortened) |

**Coding-ready files (Step 8, in `coding_ready/`)**

Step 8 removes bot-opened threads, bot messages, messages after the cutoff and empty comments. It replaces long code or log blocks, collapsed sections and images with placeholders, and flags very short "+1"-style comments in `low_content`. The collected datasets are not modified.

| File | One row per | Text column |
| --- | --- | --- |
| `coding_threads.csv` / `.jsonl` | thread | `thread_text`: title and all messages in order, each labelled with type, author role and date |
| `coding_opening_posts.csv` / `.jsonl` | thread | `opening_post_text` |
| `coding_messages_clean.csv` / `.jsonl` | message | `body_clean` |
| `cleaning_log.json` | | Counts removed at each step |

Use the `.jsonl` files in analysis scripts. Saving the CSVs from Excel can truncate long text.

## Tests

```
pip install -r requirements.txt
python -m pytest tests -q
python tests/run_notebook_offline.py LLM_Failure_GitHub_Data_Collection_v3.ipynb ./test_run
```

## License

MIT. See `LICENSE`.
