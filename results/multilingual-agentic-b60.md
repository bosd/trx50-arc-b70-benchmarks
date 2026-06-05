# Mistral-Nemo (Dutch) & Hermes-4-14B (agentic) on the B60

Two models picked for *capability*, not speed — so each gets a capability check, not just tg/s.
Single B60, Q4_K_M, SYCL (both dense → SYCL per the backend rule), xe active-power.

| Model | arch | size GiB | pp512 | tg t/s (SYCL) | active W | t/J | license |
|---|---|---|---|---|---|---|---|
| Mistral-Nemo-Instruct-2407 (12B) | llama/mistral | 7.0 | 1178 | **45.0** | 89 | 0.51 | Apache-2.0 |
| Hermes-4-14B | qwen3 (Qwen3-14B finetune) | 8.4 | 992 | **37.5** | 90 | 0.42 | Apache-2.0 |

(Vulkan: Mistral-Nemo 23.8 tg/s, Hermes-4 19.7 tg/s — SYCL wins, as for every dense model.)

## Mistral-Nemo — Dutch language quality ✅

Recommended for its European-language coverage; the box's context is partly Dutch, so tested with a
Dutch accounting prompt (*balans vs winst-en-verliesrekening*). Output was **fluent, native-like Dutch
with correct domain terminology** — *activa, passiva, eigen vermogen, inkomsten, uitgaven* — and an
accurate explanation (balance = position on a date; P&L = results over a period, with examples).
One minor article-gender slip (*"een financieel rapportage dat"* → should be *"die"*), otherwise clean.
**It answers directly in Dutch.**

**vs Gemma 4 12B (same prompt):** Gemma first emitted an **English** `<|channel>thought` planning block
(snapshot-vs-flow reasoning) before the Dutch answer — i.e. it reasons in English on a Dutch task,
spending tokens before replying. Concepts were right, but for a Dutch-first workflow Mistral-Nemo's
**direct-Dutch** behaviour is the better fit. (Mistral-Nemo is also the lighter/faster of the two: 45
vs 36 tg/s.) **Verdict: Mistral-Nemo-12B is a good Dutch daily-driver on the B60.**

## Hermes-4-14B — agentic tool-calling ✅

Recommended as a B60 agentic daily-driver. Tested with a two-tool task (`get_invoice`, `send_email`)
requiring correct ordering and argument extraction. It produced a **correct, well-formed JSON sequence**:

```json
[ {"tool_call":"get_invoice","arguments":{"invoice_id":"INV-2026-0042"}},
  {"tool_call":"send_email","arguments":{"to":"klant@example.nl","subject":"Payment reminder ...","body":"..."}} ]
```

Correct dependency order (fetch → email), right args pulled from the prompt, sensible reminder body.
(Nit: the email body came back in English despite the Dutch recipient — worth a Dutch system prompt for
NL-facing use.) **Verdict: Hermes-4-14B handles multi-step tool sequencing cleanly — a viable agentic
daily-driver on the B60 at 37 tg/s.** For Dutch-heavy agentic work, pairing Hermes' tool logic with a
Dutch system prompt (or Mistral-Nemo's Dutch fluency) is the combination to try.
