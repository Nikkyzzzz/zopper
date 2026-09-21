# OpPal — Code Review

**Reviewed:** 21 September 2026
**Scope:** whole repository (`oppal_stage`), backend + frontend
**Method:** read the code, ran the full test suite, and executed the
suspicious paths to confirm behaviour rather than guessing at it.

---

## How to read this document

There are two halves, because two different things were asked for.

**Part 1** answers the four "how does this actually work without an LLM?"
questions. It is a guided tour with file names and line numbers. If you
only want to understand the machine, read Part 1 and stop.

**Part 2** is the actual review against the six requested metrics, with a
grade and evidence for each. **Part 3** is the prioritised fix list.
**Part 4** is a file map for whoever picks up the work.

Every claim in this document points at a specific file and line, like
`main.py:1493`. Nothing here is an impression — if it is stated, it was
read, and the important ones were executed.

---

## The one-paragraph summary

OpPal is a genuinely well-engineered system built on an unusual and good
idea: the business logic lives in Markdown files that domain experts can
edit, not in Python that engineers must redeploy. That idea is really
followed through, not just claimed. The test suite is large and passes
clean. The security *thinking* is well above average for a project this
size.

But there is one endpoint that hands every user's password hash to anyone
who logs in, a live API key sitting in the source tree, and no tests at
the HTTP layer — which is exactly why nobody caught the first two. None of
these are hard to fix. They are hours of work, not weeks. They are simply
blocking, and they need doing before this goes anywhere near real patient
data.

### Scorecard

| # | Area | Grade | One-line reason |
|---|---|---|---|
| 1 | Business / Functionality | **B+** | Does what it claims, end to end, offline. One modelling gap in the simulator. |
| 2 | Architecture | **A−** | The "logic lives in Markdown" idea is real and well executed. One module cheats on it. |
| 3 | Code quality | **B** | Outstanding comments and zero TODOs, undermined by a 1,868-line god-file. |
| 4 | Security | **C** | Strong design, one critical hole, and credentials in the tree. |
| 5 | Testing | **B−** | 1,627 checks that all pass, but structurally blind to the layer where the bugs are. |
| 6 | Production readiness | **C+** | Good container, no logging, no CI, no migrations, single-instance only. |

### The five things to fix first

| | Issue | Where | Effort |
|---|---|---|---|
| 1 | Any logged-in user can read **every user's password hash and email** | `backend/app/main.py:1493` | ~1 hour |
| 2 | A **live API key** is sitting in the source tree | `backend/app/data/settings.json` | Rotate today |
| 3 | The **token-signing secret** ships with the source folder | `backend/app/data/.auth_secret` | Rotate today |
| 4 | **No `.gitignore`** — items 2 and 3 get committed the moment this becomes a git repo | repo root | 5 minutes |
| 5 | **16 of 95 API routes** never check who is calling | `backend/app/main.py` | ~1 day |

---
---

# Part 1 — How it all works with no LLM

## The single idea that explains everything

Before the four questions, one thing needs saying, because it makes all
four answers obvious.

**OpPal is not an AI system that falls back to rules when the AI is off.
It is a rules system that can optionally borrow an AI to go faster.**

That is the opposite of how most "AI products" are built, and it is the
key to the whole codebase. Look at what happens when the provider is set
to `mock`:

```python
# backend/app/main.py:55-57
    if isinstance(provider, MockProvider):
        llm = None
        provider_name = "mock"
```

The LLM object is set to **`None`**. Not a stub, not a fake — nothing.
And every agent in the system is written in this shape:

```python
# backend/app/agents/sql_agent.py:65-93  (shortened)
    def generate(self, question, intent, grounding, ...):
        if self.llm:
            ...ask the model...
            if out and "sql" in out:
                return out
        return self._template(intent, grounding, ...)   # <-- line 93
```

So with no LLM, line 93 runs. That deterministic path is not a degraded
mode — **it is the product**. The LLM is a shortcut that the system is
always free to refuse.

Even the mock provider is honest about this. It does not return canned
fake answers:

```python
# backend/app/llm/base.py:244-249
class MockProvider(LLMProvider):
    """Returns None so agents use their deterministic fallback paths."""
    name = "mock"

    def _complete(self, system, user, json_mode=False):
        return None
```

It returns `None`. There is no pretend intelligence anywhere. Everything
you see working offline is genuinely working.

**Why this matters for the business:** the compliance story depends on it.
If a regulator asks "why did the system say this?", the answer is a regex
that matched and a SQL template that was chosen — both of which can be
printed. You cannot do that with a model's reasoning.

---

## Question 1 — When we upload data, how do pillars create cases?

### First, two corrections to the question

The question assumes upload → pillars → cases happens automatically. It
does not. Two things are worth knowing before the mechanism:

**(a) Uploading data does not run anything.** The import endpoint only
lands the tables and updates the Markdown:

```python
# backend/app/main.py:1507-1518  (shortened)
@app.post("/api/data/import")
async def data_import(request: Request, files: list[UploadFile] = File(...)):
    _require(_user(request), "data_import")
    from .data.importer import import_files
    payload = [(f.filename, await f.read()) for f in files]
    out = import_files(payload)
    POOL.reset()
    CACHE.invalidate_all()
```

That is it. No pillars, no agents. **There is no scheduler anywhere in
this codebase** — no cron, no background jobs, no timers. Everything is
pull-based.

Pillars actually run **when somebody opens the Pillars tab in the UI**:

```python
# backend/app/main.py:381-387
@app.get("/api/pillars")
def get_pillars(request: Request):
    from .mcp import scope_from_user
    from .analytics import rules_engine
    scope = scope_from_user(_user(request))
    return rules_engine.pillar_summary(scope)
```

Triggered from the frontend at `frontend/src/components/PillarsTab.jsx:12`.

**(b) Pillars do not create cases.** Pillars show *breaches*. Only the
**swarm** creates tracked cases, and only through one agent (that is
Question 3). So the real picture is two separate chains:

```
  upload  ->  tables land in SQLite + Brain Markdown updated
                 |
                 +-- (someone opens Pillars tab) --> rules run --> breaches displayed
                 |
                 +-- (someone clicks Run Swarm)  --> agents run --> findings --> CASES
```

### Now the mechanism: what a "pillar" actually is

A **pillar** is just a named bucket of checks — `regulatory`,
`claims_triage`, `fwa`. And a **check** is, in the important case, just a
SQL query that somebody wrote in a Markdown file.

Here is a real rule, straight out of the Medicare Sales domain
(`backend/app/semantic/domains/medicare_sales/policy_rules.md:14-23`,
abbreviated):

```markdown
### CMS-AGENT-CERT-2274
- citation: 42 CFR §422.2274(b) — licensure, appointment and annual training
- check: agent_certification_currency
- pillar: regulatory
- max_breach_count: 0
- breach_action: fail
- headline: Agent {agent_id} — {applications} application(s) written after
            the {cert_type} expired on {expiry_date} ({lob}, {market})
- query: SELECT a.lob, a.market, a.agent_id, c.cert_type, c.expiry_date,
         COUNT(DISTINCT a.application_id) AS applications
         FROM applications a JOIN agent_certifications c ...
- remediation: Suspend the agent from writing new business until reinstated
```

**That is a compliance officer's work product, not an engineer's.** The
regulation citation, the SQL that detects the breach, the sentence shown
to the user, and the recommended fix all live in one Markdown block.

### The one function that runs all of them

There is no Python code specific to that rule. **One** generic handler
executes every rule in every domain:

```python
# backend/app/analytics/rules_engine.py:342-383  (the core loop)
@register_check("brain_query", "regulatory")
def _brain_query_pillar(scope: ScopeContext) -> dict:
    rows_out, evaluated = [], []
    for r in get_brain().get("policy_rules", []):
        q = str(r.get("query") or "").strip()
        if not q:
            continue                               # no SQL -> not our job
        evaluated.append(r.get("rule_id") or r.get("check"))
        res = _rows(scope, " ".join(q.split()))    # run it, scope-enforced
        if not res["ok"]:
            rows_out.append({"rule": ..., "error": "query refused by the governance gate"})
            continue
        cols = res["columns"]
        tpl = str(r.get("headline") or "{0}")
        for row in res["rows"]:
            d = dict(zip(cols, row))
            try:
                head = tpl.format(**d)             # fill in {agent_id} etc.
            except (KeyError, IndexError, ValueError):
                head = tpl
            rows_out.append({"rule": ..., "citation": ..., "headline": head,
                             "breach_action": ..., "remediation": ...,
                             "evidence": d})
```

In plain English, four steps:

1. **Loop every rule that has a `query:`.** Rules without one are skipped.
2. **Run that SQL** through the scoped gate (`_rows`, line 43-45), so a
   user only ever sees their own line of business.
3. **Every row that comes back is one breach.** The queries are written to
   return *only* offending rows — so `breach_count` (line 382) is simply
   the number of rows.
4. **Fill in the headline.** `tpl.format(**d)` is ordinary Python string
   formatting: `{agent_id}` in the Markdown gets replaced by the
   `agent_id` column from the row. That is all the "natural language" is.

No LLM. No inference. A human wrote the SQL and a human wrote the
sentence; the code just joins them to the data.

### How a finding becomes a tracked case

Only the swarm does this, in one place:

```python
# backend/app/agents/departmental.py:510-517
            case = case_store.upsert_case(
                case_type=spec["case_type"], lob=lob,
                subject=str(spec["subject"]), pattern=spec["pattern"],
                detail=spec.get("detail", ""),
                severity=spec.get("severity", "medium"),
                trust_score=round(run_trust, 3),
                estimated_impact=float(spec.get("estimated_impact", 0) or 0),
                evidence=f.get("evidence", {}), created_by=f["agent"])
```

Cases are deduplicated by a hash of `type|lob|subject|pattern`
(`case_store.py:63-65`), so running the swarm twice bumps the existing
case instead of creating a duplicate.

### Files to open, in order

| # | File | What to look at |
|---|---|---|
| 1 | `backend/app/semantic/domains/medicare_sales/policy_rules.md` | What a rule looks like. Start at line 14. |
| 2 | `backend/app/semantic/brain.py:210-215` | How that Markdown becomes a Python dict |
| 3 | `backend/app/analytics/rules_engine.py:342-383` | **The one handler that runs them all** |
| 4 | `backend/app/analytics/rules_engine.py:386-430` | `pillar_summary` — the dashboard entry point |
| 5 | `backend/app/main.py:381-387` | The `/api/pillars` endpoint |
| 6 | `backend/app/agents/case_store.py:78-121` | `upsert_case` + deduplication |
| 7 | `backend/app/main.py:1507-1518` | The upload endpoint (proves it runs no rules) |

---

## Question 2 — I type in the chat box. Where do the SQL and the graph come from?

This is the most interesting path in the system, so it gets the most
space. Short answers to your four sub-questions first:

| You asked | Short answer |
|---|---|
| How does SQL get written with no LLM? | From about a dozen **hand-written SQL templates** in `sql_agent.py`, filled in with fragments that come from Markdown. |
| Does it use regex to find the table name? | **No.** Regex finds *filters and groupings*. The **table comes from the metric's declaration** in `metrics.md`. |
| How does it know x and y? | The **backend decides both** by looking at the shape of the result, and sends `"x"` and `"series"` to the browser. The browser just obeys. |
| How does it know what SQL to write for my sentence? | It classifies your sentence into one of ~9 **intents** using keyword lists, then picks the template for that intent. |

The pipeline has four deterministic stages. Let us walk a real question
through all four:

> **"Why did medical cost spike in Medicaid in 2025 Q3?"**

### Stage 1 — Grounding: which business concepts did you mention?

`backend/app/semantic/graph.py:148-219`, function `ground()`.

This is called "graph-RAG" in the docs, but with no LLM it is much
simpler than that sounds: **it is dictionary lookup with a plural-tolerant
regex.**

```python
# backend/app/semantic/graph.py:152-159
    for term, target in SYNONYMS.items():
        # plural-tolerant: 'member count' matches 'member counts' etc.
        pattern = rf"\b{re.escape(term)}s?\b"
        if re.search(pattern, q):
            matched_terms.append(term)
            seeds.add(target)
            extraction.append({"entity": target, "regex": pattern,
                               "matched_text": term})
```

`SYNONYMS` is built from `vocabulary.md`, which is a list of lines like
`medical cost => metric:medical_cost`. So every business phrase anyone
might type has been written down by a human. The code turns each one into
`\bterm s?\b` and checks whether it appears in your sentence. A term
either matches or it does not — no fuzzy matching, no embeddings, no
ranking.

It then walks up to 3 hops through a graph of tables/columns/metrics
(lines 161-174) to pull in related context, and produces a confidence
number:

```python
# backend/app/semantic/graph.py:191-196
    tokens = [t for t in re.findall(r"[a-z0-9]{2,}", q)
              if t not in STOPWORDS]
    coverage = (len(matched_terms) / max(len(tokens), 1)) if tokens else 0.0
    coverage = min(coverage * 1.8, 1.0)
    if metrics or drugs:     # resolved hop -> floor for the trust gate
        coverage = max(coverage, 0.8)
```

In words: *"what fraction of the meaningful words in your question did I
recognise?"*, multiplied by 1.8 and capped at 1.0. (I have a criticism of
that last `max(coverage, 0.8)` line — see the Architecture section.)

For our example, this matches `medical cost` and `medicaid`.

### Stage 2 — Intent routing: what kind of question is this?

`backend/app/agents/intent_router.py:218-384`, function `_rule_based()`.

**This is where the regex lives, and there is a lot of it.** This function
is 166 lines of pattern matching, and it is worth your manager seeing the
actual code, because it removes all mystery.

**Finding the filters** — it loops over the *known values* of each
dimension (read from Markdown) and checks whether you said any of them:

```python
# backend/app/agents/intent_router.py:262-286  (shortened)
        for v in _vals("lob"):
            if match(rf"\b{re.escape(v.lower())}\b"):
                filters["lob"] = v
        for v in _vals("region"):
            if match(rf"\b{re.escape(v.lower())}\b"):
                filters["region"] = v
        ...
        m = match(r"\breject code\s*(70|75|76)\b") or match(r"\bcode\s*(70|75|76)\b")
        if m:
            filters["reject_code"] = m.group(1)
        m = match(r"\btier\s*([1-5])\b")
        if m:
            filters["formulary_tier"] = m.group(1)
        m = match(r"\b(pr-\d{3})\b")
        if m:
            filters["provider_id"] = m.group(1).upper()
```

**Finding the time period** — a straightforward date regex:

```python
# backend/app/agents/intent_router.py:288-296
        qpat = r"(20\d\d)\s?q\s?([1-4])"
        mr = match(rf"(?:from|between|starting from)?\s*{qpat}\s*(?:to|and|through|-|until)\s*{qpat}")
        if mr:
            filters["quarter_range"] = [f"{mr.group(1)}Q{mr.group(2)}",
                                        f"{mr.group(3)}Q{mr.group(4)}"]
        else:
            m = match(r"\b(20\d\d)\s?q([1-4])\b")
            if m:
                filters["service_quarter"] = f"{m.group(1)}Q{m.group(2)}"
```

**Finding the grouping** — three passes, in order. First a hardcoded
phrase table:

```python
# backend/app/agents/intent_router.py:18-31
GROUP_BY_KEYWORDS = {
    "lob": ["by lob", "by line of business", "per lob", "by plan",
            "for each plan", "each plan", "per plan", "for each lob",
            "across lobs", "by contract"],
    "region": ["by region", "across regions", "per region"],
    "claim_type": ["by claim type", "by type", "by place of service"],
    ...
}
```

Then any dimension the Markdown declares (lines 318-326), then bare nouns
like "which reject codes..." (lines 332-339).

**Deciding the intent** — a plain if/elif ladder of keyword checks:

```python
# backend/app/agents/intent_router.py:341-364  (shortened)
        if COMPLIANCE_HINTS.search(q):
            intent = "compliance_audit"
        elif grounding.get("drugs") and re.search(r"\b(tier|cover|formulary|prior auth|pa\b|why)", q):
            intent = "formulary_lookup"
        elif any(w in q for w in ["scatter", "correlation", "volume vs"]):
            intent = "correlation"
        elif any(w in q for w in ["why", "driver", "driven", "cause",
                                  "explain", "root", "reason"]):
            intent = "root_cause"
        elif any(w in q for w in ["top", "worst", "best", "highest", "lowest",
                                  "rank", "share", "mix", "distribution",
                                  "spike", "spiked", "jump", "surge", "which"]):
            intent = "top_n"
        elif group_by:
            intent = "breakdown"
        elif any(w in q for w in ["list", "detail", "sample", "show me claims"]):
            intent = "detail_lookup"
        else:
            intent = "metric_trend"
```

That is genuinely it. **"Why" in your sentence means `root_cause`. "Top"
or "worst" means `top_n`.** Nothing cleverer is happening.

For our example: `"why"` matches, so intent = **`root_cause`**, with
`filters = {lob: "Medicaid", service_quarter: "2025Q3"}` and
`metric = "medical_cost"`.

Every single regex that fired is recorded with the text it matched
(lines 222-227) and shipped back to the UI as the "extraction trace" —
which is how the journey strip can show a regulator exactly why the
system understood the question the way it did.

### Stage 3 — SQL generation: no, it does not regex for table names

`backend/app/agents/sql_agent.py:95-327`, function `_template()`.

**Your question was: "Does it apply any kind of Regex to find table
name?" The answer is no.** The table is not extracted from your sentence
at all. It is declared alongside the metric in Markdown, and simply looked
up:

```python
# backend/app/agents/sql_agent.py:97-103
        METRIC_SQL = _metric_sql(brain)      # from metrics.md
        DIM_SQL = _dim_sql(brain)            # from entities.md
        default_metric = next(iter(METRIC_SQL), "medical_cost")
        metric = intent.get("metric") or default_metric
        if metric not in METRIC_SQL:
            metric = default_metric
        expr, base, needs_members, is_avg = METRIC_SQL[metric]
```

That last line is the whole answer. The metric `medical_cost` carries its
own `sql:` fragment and its own `base:` table, both read from
`metrics.md`. The chain is:

```
  your words  ->  vocabulary.md  ->  metric id  ->  metrics.md  ->  base table
   "medical cost"     lookup        medical_cost     lookup          claims
```

The table name is then chosen by a one-line mapping, not a search:

```python
# backend/app/agents/sql_agent.py:218-220
        alias = "rx" if base == "rx" else "c"
        table = "pharmacy_claims rx" if base == "rx" else "claims c"
        dims_map = DIM_SQL[base]
```

Joins are added from *flags on the metric*, never from your text:

```python
# backend/app/agents/sql_agent.py:222-227
        if needs_members or (cohort and cohort.get("needs_members")):
            joins.append(f"JOIN members m ON {alias}.member_id = m.member_id")
        if base == "claims" and (group_by == "provider"
                                 or "provider_id" in filters
                                 or intent["intent"] == "correlation"):
            joins.append("JOIN providers pr ON c.provider_id = pr.provider_id")
```

**The intent picks the shape of the query.** This is the whole "how does
it know what SQL to write" answer — there is one template per intent:

| Intent | Line | Shape of SQL produced |
|---|---|---|
| `formulary_lookup` | `121-128` | `SELECT ... FROM formulary WHERE rxnorm_code IN (...)` |
| *(table-declared metric)* | `143-187` | `FROM {declared_table}`, with scope predicate |
| `base == members` | `203-215` | `FROM members m2 GROUP BY {dim}` |
| `correlation` | `276-283` | provider volume vs severity, `HAVING COUNT >= 20` |
| `metric_trend` / `root_cause` | `284-286` | `GROUP BY service_quarter ORDER BY quarter` |
| `top_n` | `287-292` | `GROUP BY {dim} ORDER BY 2 DESC LIMIT 10` |
| `breakdown` | `293-302` | dimension × quarter matrix |
| `detail_lookup` | `303-317` | flat row listing, `LIMIT 100` |

And the final assembly is ordinary string building:

```python
# backend/app/agents/sql_agent.py:322-327
        sql = (f"SELECT {', '.join(dims)}, {expr} AS {metric}{count_col} "
               f"{' '.join(joins)} {where} {grp} {order}")
        return {"sql": " ".join(sql.split()),
                "explanation": f"Governed template for {metric} "
                               f"({intent['intent']}, base={base})",
                "source": "template"}
```

Note `"source": "template"` — the answer carries a label saying it was
machine-generated from a governed template rather than written by a model.
That label is what the UI's provenance receipt displays.

Security is stapled on here rather than trusted to the caller — the
user's permitted lines of business are appended as a `WHERE` clause on
every single query:

```python
# backend/app/agents/sql_agent.py:329-335
    @staticmethod
    def _where(conds, alias, filters, allowed_lobs, lob_col):
        conds = list(conds)
        if allowed_lobs is not None:
            lobs = ", ".join(f"'{l}'" for l in allowed_lobs)
            conds.append(f"{lob_col} IN ({lobs})")   # RBAC scope
        return ("WHERE " + " AND ".join(conds)) if conds else ""
```

### Stage 4 — The chart: how x and y get chosen

`backend/app/agents/viz_agent.py` — the whole file is 115 lines and
contains no LLM code at all. Charts are **always** rule-based, even when
an LLM is connected.

**Step A: pick the chart type from the shape of the data.** Not from your
question — from how many rows and columns came back:

```python
# backend/app/agents/viz_agent.py:10-34
def recommend_chart(intent, cols, rows):
    """Analyze the result shape and pick the most readable chart.
    Returns (chart_type, reason)."""
    has_quarter = "quarter" in cols
    n = len(rows)
    metric = intent.get("metric", "")
    multi_dim = has_quarter and len(cols) >= 3 and cols[0] != "quarter"
    if intent["intent"] == "correlation":
        return "scatter", "two numeric measures — correlation reads best as a scatter"
    if multi_dim:
        n_series = len({r[0] for r in rows})
        n_quarters = len({r[cols.index("quarter")] for r in rows})
        if n_quarters <= 6 and n_series <= 5:
            return "bar", (...)
        return "multiline", (...)
    if has_quarter:
        return ("area" if metric.startswith(("medical_cost", "pharmacy_cost",
                                             "total_")) else "line"), "..."
    if n <= 6 and intent.get("wants_pie"):
        return "pie", "few categories as shares — pie works"
    if n <= 12:
        return "bar", f"{n} categories — bars compare magnitudes best"
    return "bar", "many categories — sorted bars remain scannable"
```

Read that as a set of readability rules a designer would give you: *time
on the axis → line; money over time → area; ≤ 12 categories → bars; a
handful of shares → pie.* It also returns the **reason**, which is shown
in the UI.

**Step B: set the axes explicitly.** There are four cases, and each one
writes `"x"` and the y-series into the spec by name:

```python
# backend/app/agents/viz_agent.py:68-109  (the four cases, abbreviated)
        if intent["intent"] == "correlation":
            artifacts.append(spec({
                "chart_type": "scatter",
                "x": "claim_count", "y": "claim_severity",     # <-- explicit x and y
                "label": cols[0], "format": "currency"}))
        elif "quarter" in cols and len(cols) >= 3 and cols[0] != "quarter":
            artifacts.append(spec({
                "x": "quarter", "series_dim": cols[0], "value": metric,
                "pivot": True}))
        elif intent["intent"] in ("metric_trend", "root_cause") and "quarter" in cols:
            artifacts.append(spec({
                "x": "quarter", "series": [metric], ...}))
        elif intent["intent"] in ("top_n", "breakdown") and cols:
            dim = cols[0]                                       # <-- x is the first column
            value_col = metric if metric in cols else next(
                (c for c in cols[1:] if c != "claim_count"), cols[-1])
            artifacts.append(spec({
                "x": dim, "series": [value_col], "colorful": True, ...}))
```

The rule in plain English:

- **x-axis** = `"quarter"` if the result has a time column, otherwise the
  **first column** of the result (which is the thing you grouped by).
- **y-axis** = the metric you asked about — *if* the SQL actually selected
  it. Line 100-101 is a nice touch: if the query returned a different
  value column (e.g. a count instead of a rate), it plots the real column
  rather than mislabelling the chart.

**Step C: the browser just obeys.** Nothing is decided in the frontend:

```jsx
// frontend/src/components/ChartArtifact.jsx:161
    <XAxis dataKey={spec.x} {...axis}
// frontend/src/components/ChartArtifact.jsx:235
        <Bar key={k} dataKey={k} radius={barRadius}
```

`spec.x` is the string the Python code put there. Recharts is handed a
column name and draws it.

### The whole trip, end to end

```
 "Why did medical cost spike in Medicaid in 2025 Q3?"
   |
   |  graph.py:152      regex "\bmedical cost s?\b" hits vocabulary.md
   v                    -> metric:medical_cost
 GROUNDING              coverage score = 0.8+
   |
   |  intent_router.py:349   "why" in question  -> intent = root_cause
   |  intent_router.py:262   "medicaid" matches a known lob value
   |  intent_router.py:294   "2025 q3" matches the quarter regex
   v
 INTENT  {intent: root_cause, metric: medical_cost,
          filters: {lob: Medicaid, service_quarter: 2025Q3}}
   |
   |  sql_agent.py:103   metrics.md says medical_cost lives on base "claims"
   |  sql_agent.py:219   base "claims" -> FROM claims c
   |  sql_agent.py:284   intent root_cause -> GROUP BY service_quarter
   |  sql_agent.py:334   append  WHERE c.lob IN ('Medicaid')
   v
 SQL     SELECT c.service_quarter AS quarter, SUM(...) AS medical_cost
         FROM claims c WHERE c.lob IN ('Medicaid') GROUP BY ... ORDER BY quarter
   |
   |  mcp/base.py:83     read-only check, scope check, row cap, execute
   v
 ROWS    [["2025Q1", 41200], ["2025Q2", 43100], ["2025Q3", 58800], ...]
   |
   |  viz_agent.py:28    has a "quarter" column + cost metric -> "area"
   |  viz_agent.py:83    x = "quarter", series = ["medical_cost"]
   v
 CHART SPEC  {chart_type: "area", x: "quarter", series: ["medical_cost"]}
   |
   |  ChartArtifact.jsx:161   <XAxis dataKey="quarter">
   v
 PICTURE
```

Not one step of that needed a model.

### Files to open, in order

| # | File | What to look at |
|---|---|---|
| 1 | `backend/app/main.py:218-257` | The `/api/ask` endpoint |
| 2 | `backend/app/agents/orchestrator.py:78-312` | `_handle` — the 9-step pipeline, with the audit trail |
| 3 | `backend/app/semantic/graph.py:148-219` | `ground()` — the dictionary lookup |
| 4 | **`backend/app/agents/intent_router.py:218-384`** | **`_rule_based` — all the regex** |
| 5 | `backend/app/agents/intent_router.py:18-54` | The keyword tables at the top of the file |
| 6 | **`backend/app/agents/sql_agent.py:95-327`** | **`_template` — all the SQL templates** |
| 7 | `backend/app/semantic/domains/healthcare_payer/metrics.md` | Where `sql:` and `base:` are declared |
| 8 | **`backend/app/agents/viz_agent.py:10-115`** | **The entire chart decision, x and y included** |
| 9 | `frontend/src/components/ChartArtifact.jsx:161,235` | Proof the browser only obeys |
| 10 | `backend/app/llm/base.py:244-249` | `MockProvider` returning `None` |

---

## Question 3 — How do swarms work without an LLM? How does a sub-agent produce a finding?

### The short answer

A "sub-agent" is **a Markdown block containing a SQL query**. A "finding"
is **one row that query returned**. The severity is decided by comparing
one named column against a number. That is the entire mechanism.

### An agent is a Markdown block

Here is a real one, from
`backend/app/semantic/domains/medicare_sales/agents.md:126-147`
(abbreviated):

```markdown
## compensation_siu
- department: Finance & SIU
- pillars: fwa
- detector: brain_query
- finding_kind: compensation_overpayment
- severity_column: overpaid_usd
- severity_threshold: 2000
- impact_column: overpaid_usd
- min_impact: 250
- headline: Agent {agent_id} — {overpaid_usd} USD paid on {payments}
            enrolment(s) that never effectuated ({lob})
- case_type: CompensationCase
- case_subject_column: agent_id
- case_pattern: compensation_overpayment
- query: SELECT c.lob, c.agent_id, COUNT(*) AS payments,
         ROUND(SUM(c.commission_paid + c.bonus_paid + c.override_paid), 2)
           AS overpaid_usd
         FROM agent_commissions c JOIN enrollments e ON ...
         WHERE e.effectuated_flag = 0 AND (...) > 0
         GROUP BY c.lob, c.agent_id ORDER BY overpaid_usd DESC
```

Read that as a job description written by a fraud investigator: *find
agents who were paid commission on enrolments that never took effect;
anything over $2,000 is high severity; ignore anything under $250; open a
CompensationCase against the agent.*

### `detector: brain_query` is a sentinel, not a function

This is the neat trick, and it is worth understanding.

```python
# backend/app/agents/departmental.py:66-104  (abbreviated)
    def act(self, ctx: AgentContext, **kw):
        name = self.persona.get("detector", "") or ""
        detector = getattr(self, name, None)
        if detector is None:
            ...
            if self.persona.get("query"):
                return self._declarative(ctx)
            ...
        out = detector(ctx) or []
        for f in out:
            ctx.findings.append(f)
```

There is **no method called `brain_query` anywhere in the Python**. So
`getattr(self, "brain_query")` returns `None`, the code notices the
persona has a `query:`, and falls through to `_declarative`. The string
`brain_query` is a convention meaning *"I am declared in Markdown, not
coded in Python."*

The other nine detector names **are** real Python methods, for checks too
gnarly to express as one query:

| `detector:` value | Python class | Lines |
|---|---|---|
| `brain_query` | *(none — falls through to `_declarative`)* | `107-199` |
| `triage` | `SentryAgent` | `205-253` |
| `cost_drivers` | `MedicalEconomicsAgent` | `256-291` |
| `pbm_rejects` | `PBMForensicsAgent` | `294-323` |
| `pa_triage` | `ClinicalUMAgent` | `326-344` |
| `fwa_scan` | `FWASIUAgent` | `347-364` |
| `regulatory_timeliness` | `ComplianceAgent` | `367-387` |
| `network_exposure` | `NetworkAgent` | `390-412` |
| `hedis_gaps` | `QualityHEDISAgent` | `415-432` |
| `risk_profile` | `RiskHCCAgent` | `435-467` |
| `propose_actions` | `PrescriptiveActionAgent` | `470-533` |

Worth noting: `medicare_sales` and `pharma_supply_chain` use
`brain_query` for **every** investigator. Only the original payer domain
uses the Python detectors. That is the declare-don't-code principle
winning over time.

### How a row becomes a finding

`backend/app/agents/departmental.py:107-199`, `_declarative()`:

1. **Run the persona's SQL** through the scoped gate (line 130).
2. **Read the persona's settings** — `severity_column`,
   `severity_threshold`, `impact_column`, `min_impact`, `headline`,
   `case_type`... (lines 139-155).
3. **For each row**: extract the impact, skip it if it is below
   `min_impact` (lines 168-175).
4. **Decide severity** — this is the entire "judgement" the agent makes:

   ```python
   # backend/app/agents/departmental.py:176-182  (in essence)
   #   static severity if declared, otherwise:
   #   value >= severity_threshold  ->  "critical" / "high"
   ```

5. **Build a case spec**, but only if the persona declared a `case_type`:

   ```python
   # backend/app/agents/departmental.py:183-190
               case = None
               if case_type:
                   case = {"case_type": case_type,
                           "subject": str(row.get(case_subj) or "unknown"),
                           "pattern": case_pat,
                           "estimated_impact": impact,
                           "detail": fmt(det_t, row) if det_t else "",
                           "severity": sev}
   ```

6. **Emit the finding** with its headline filled in from the row
   (lines 191-194).

So: **one row in, one finding out.** There is no reasoning step, no
summarisation, no model call. The `headline` you see in the UI is
`str.format` over a database row — same mechanism as the pillars.

### How the swarm runs several agents

`backend/app/agents/swarm.py:115-224`, `run_swarm()`.

Which agents wake up is also keyword matching
(`select_agents:45-111`): each persona declares trigger keywords, they
are matched against the question, scored with weights (line 73), bumped
if an anomaly was detected (lines 89-93), and cut off at a budget
(lines 101-109, `max_active_agents: 4` in the Markdown).

Then they simply run in a loop:

```python
# backend/app/agents/swarm.py:169-175
    def _run_departments(c: AgentContext):
        for persona in activated:
            if persona["name"] == "prescriptive_action":
                continue                      # runs last, after all findings
            build_agent(persona, llm).run(c)
    graph.add("departmental_agents", _run_departments)
```

They share a blackboard — `AgentContext.findings`
(`framework.py:60`) — that they all append to. That is the only
"collaboration" in the swarm: a shared Python list. The word *swarm* is
doing a lot of marketing work for a `for` loop, and that is fine; the
loop is honest and auditable, which is the point.

Each agent is wrapped so one failing agent cannot sink the run
(`framework.py:84-105`).

Finally, `PrescriptiveActionAgent` runs last, reads everything on the
blackboard, and is the **only** thing in the system that opens cases
(lines 470-533).

### Files to open, in order

| # | File | What to look at |
|---|---|---|
| 1 | `backend/app/semantic/domains/medicare_sales/agents.md` | What a persona looks like. Start at line 126. |
| 2 | `backend/app/semantic/brain.py:287-337` | Markdown → persona dict (line 305 is the load-bearing one) |
| 3 | `backend/app/agents/departmental.py:553-555` | `build_agent` — three lines |
| 4 | `backend/app/agents/departmental.py:66-104` | `act` — the `brain_query` sentinel trick |
| 5 | **`backend/app/agents/departmental.py:107-199`** | **`_declarative` — row → finding** |
| 6 | `backend/app/agents/departmental.py:470-533` | The only code that creates cases |
| 7 | `backend/app/agents/swarm.py:45-111` | `select_agents` — keyword activation |
| 8 | `backend/app/agents/swarm.py:115-224` | `run_swarm` — the loop at 169-175 |
| 9 | `backend/app/agents/framework.py:60,84-105` | The shared blackboard and the failure wrapper |

---

## Question 4 — How does simulation work?

### The short answer

Two SQL queries' worth of "before" numbers, then **multiplication**.

That is not a criticism — for a what-if tool it is the right call, and it
makes every number reproducible. But it is worth being precise about,
because "simulation" implies something heavier than what runs.

### Step 1: copy the Brain so nothing can leak

```python
# backend/app/simulation/isolation.py:57-68
def fork_brain(domain: str | None = None) -> dict:
    live = get_brain()
    fork = copy.deepcopy(live)
    assert fork is not live, "fork_brain must not return the live singleton"
    for key in _FORK_CHECK_KEYS:
        if key in live:
            assert fork.get(key) is not live.get(key), (
                f"fork_brain aliases the live singleton's '{key}' section")
    return fork
```

A deep copy, with **assertions that the copy does not secretly share
memory with the original**. This is good defensive engineering: the worst
possible bug here would be a sandbox experiment quietly changing the live
production configuration, and these two asserts make that impossible
rather than merely unlikely.

The module documents five isolation invariants at lines 3-27 (brain,
scope, data, write, concurrency).

### Step 2: measure the baseline with real SQL

`backend/app/simulation/engine.py:499-528` runs **five real aggregate
queries** against the warehouse: a ratio (MLR), two cost figures, a
volume figure, and a population count.

### Step 3: apply the user's changes — to the copy only

```python
# backend/app/simulation/engine.py:532
        _apply_overrides_to_fork(brain_fork, request.parameter_overrides)
```

An override addresses a single numeric leaf by dotted path. It is
explicitly forbidden from touching anything that contains SQL —
`sql`, `formula`, `value_sql`, `table`, `time_column` — and that is
enforced twice, independently (a Pydantic denylist *and*
`_assert_safe_override_target:282-289`). **A user can change a number.
A user can never change a query.**

### Step 4: the "simulation" itself

Here is the part worth knowing:

```python
# backend/app/simulation/engine.py:534-546
        def _stressed(name, base):
            pct = _stress(brain_fork, name)      # a plain fraction: -0.05 == -5%
            return (base * (Decimal(1) + pct)).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP)

        simulated_mlr = _stressed(bind["ratio_stress_key"], baseline_mlr)
        simulated_medical = _stressed(bind["cost_a_stress_key"], baseline_medical)
        simulated_pharmacy = _stressed(bind["cost_b_stress_key"], baseline_pharmacy)
        simulated_rejects = max(Decimal(0),
                                _stressed(bind["volume_stress_key"], baseline_rejects))
```

**There is no second query.** The simulated numbers are
`baseline × (1 + percentage)`, computed in `Decimal` for exact money
arithmetic. The percentages are read back off the forked Brain.

The dollar impact is then a subtraction:

```python
# backend/app/simulation/engine.py:554-557
        net_dollar_impact = ((baseline_medical - simulated_medical) +
                             (baseline_pharmacy - simulated_pharmacy)
                             ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
```

### ⚠ A modelling gap worth raising with the product owner

Look carefully at line 539 versus lines 554-557. **The four levers are
completely independent of each other.**

`simulated_mlr` is stressed off `baseline_mlr` on its own. It is **not**
derived from the simulated cost figures. So:

> If a user drops medical cost by 10%, the medical loss ratio **does not
> move** — unless they separately drag the MLR lever too.

A user can therefore build a scenario that says *"costs fell 10% and the
loss ratio was unchanged"*, and the tool will report it without
complaint. For a finance or actuarial audience that is a credibility
risk, because MLR is by definition a function of cost.

This is a **modelling limitation, not a bug** — the code does exactly
what it says. But "simulator" sets an expectation of a causal model, and
what exists is four independent sliders. Either derive the ratio from the
stressed costs, or rename the feature to something like "sensitivity
sliders" so nobody is surprised in a board meeting.

### Step 5: the same four levers, renamed per industry

This is the elegant part. The engine only knows about "ratio", "cost A",
"cost B" and "volume". Each domain says what those mean in its own
language, in `simulation.md` under `### simulator_bindings`:

| Lever | Healthcare payer | Medicare sales |
|---|---|---|
| ratio | `mlr` (medical loss ratio) | `lead_loss_ratio` |
| cost A | `medical_cost` | `acquisition_cost` |
| cost B | `pharmacy_cost` | `commission_cost` |
| volume | `reject_count` | `cancellation_count` |
| population | members | leads |

Parsed generically at `brain.py:343-347`, consumed at
`engine.py:130-146`. Adding a new industry means writing six lines of
Markdown, not touching the engine.

### Step 6: promotion never writes anything

```python
# backend/app/simulation/engine.py:766-770  (docstring)
def promote(run_id: str, note: str, user: dict, scope):
    """REQ-5.5 — promotion NEVER executes anything; it proposes a Phase-4
    ActionRequest into the SAME dual-control queue as any other action
```

"Promoting" a simulation just files a proposal into the normal
maker-checker approval queue (`action_engine.propose`, lines 790-808),
targeted at the `dry_run` adapter. Someone still has to approve it.
Severity is derived from a threshold in the Brain, not hardcoded
(lines 779-788).

### Files to open, in order

| # | File | What to look at |
|---|---|---|
| 1 | `backend/app/simulation/isolation.py:3-27` | The five invariants, in the docstring |
| 2 | **`backend/app/simulation/isolation.py:57-68`** | **`fork_brain` — deep copy + asserts** |
| 3 | `backend/app/simulation/engine.py:11-28` | The override contract |
| 4 | `backend/app/simulation/engine.py:282-289` | What an override may *not* touch |
| 5 | `backend/app/simulation/engine.py:499-528` | The five baseline SQL queries |
| 6 | **`backend/app/simulation/engine.py:534-557`** | **The actual arithmetic — and the modelling gap** |
| 7 | `backend/app/simulation/engine.py:130-146` | `bindings()` — per-domain lever renaming |
| 8 | `backend/app/semantic/domains/medicare_sales/simulation.md:8-34` | A real `simulator_bindings` block |
| 9 | `backend/app/simulation/engine.py:766-822` | `promote()` — proposes, never writes |

---

## Part 1 summary — assumption vs reality

| Capability | What people assume | What actually runs | Where |
|---|---|---|---|
| "Understands my question" | Language model | Dictionary of phrases + `\bterm s?\b` regex | `graph.py:152-159` |
| "Knows what I'm asking for" | Language model | if/elif ladder over keyword lists | `intent_router.py:341-364` |
| "Writes SQL" | Language model | ~9 hand-written templates, one per intent | `sql_agent.py:95-327` |
| "Finds the right table" | Regex / NER on your text | Looked up from the metric's `base:` in Markdown | `sql_agent.py:103,218-220` |
| "Picks the best chart" | Language model | Row/column count heuristics | `viz_agent.py:10-34` |
| "Chooses x and y" | Frontend figures it out | Backend writes `"x"` and `"series"` into the spec | `viz_agent.py:68-109` |
| "AI agents investigate" | Autonomous reasoning | Each agent = one SQL query in Markdown | `departmental.py:107-199` |
| "Agents collaborate" | Negotiation / delegation | A `for` loop appending to a shared list | `swarm.py:169-175` |
| "Simulates outcomes" | Predictive model | `baseline × (1 + pct)` in `Decimal` | `engine.py:534-546` |
| "Explains its answer" | Model narration | Sentence templates filled from rows | `insight_agent.py:37-121` |

**This is a strength, not an exposure.** A regulated buyer should *prefer*
every one of these. A regex you can print beats a model you cannot
interrogate, and the LLM path can be switched on later for the
convenience of analysts without any of these guarantees changing.

---
---

# Part 2 — The review

## 1. Business / Functionality — **B+**

### What works

**The product does what the pitch says, end to end, with nothing plugged
in.** That is rarer than it sounds. I ran the full regression suite for
this review and it passed clean:

```
PASS  1627 phase checks + the full business scenario
```

Every suite green: 7 phase packs, 8 demo pre-flights, 26 standalone
suites, 5 frontend suites. The `medicare_sales` all-tabs pre-flight alone
executes 119 checks covering every documented keystroke in the demo.

**Three complete industries ship**, not one plus a promise:
`healthcare_payer`, `medicare_sales` and `pharma_supply_chain` each have
11 Brain files, plus `retail_banking` as a 7-file template. A domain
expert can add a compliance rule by writing Markdown — no engineer, no
deploy, no release.

**The governance chain is real.** Maker-checker is enforced by shipping
*two* compliance officers (`u-compliance`, `u-compliance2`) so one person
physically cannot complete a dual-control approval. Simulation promotion
files a proposal rather than executing (`engine.py:766-770`). Write-backs
route through a dry-run adapter.

### What is weak

**The simulator's levers are independent** (detailed in Part 1,
Question 4). Costs and the loss ratio do not move together, so a user can
produce an internally inconsistent scenario. `engine.py:539` vs
`engine.py:554-557`. This is the one finding in this section that could
embarrass someone in front of a customer.

**Nothing runs on a schedule.** There is no cron, no background worker, no
queue anywhere in the codebase. Compliance breaches are only discovered
when a human opens the Pillars tab, and cases only exist if somebody
clicks Run Swarm. For a *monitoring* product that is a real functional
gap — the current design is "a dashboard you must remember to look at".
Worth deciding deliberately whether that is the intended product.

**Grade: B+.** Functionally complete and genuinely demonstrable offline.
Held back by the simulator modelling gap and the absence of any
scheduled/continuous monitoring.

---

## 2. Architecture — **A−**

### The central idea is good, and it is actually followed

The design rule is that domain knowledge lives in editable Markdown, and
**one generic handler interprets all of it**. Most codebases claim
something like this and then quietly grow a bespoke function per rule.
This one largely did not:

- 12 of 16 policy rules in `medicare_sales` are query-bearing and run
  through a **single** 40-line function (`rules_engine.py:342-383`).
- Every investigator agent in two of the three domains is declared, not
  coded (`departmental.py:107-199`).
- A new industry renames the simulator's levers in six lines of Markdown
  (`engine.py:130-146`).

The layering is clean and the security boundaries are drawn in the right
places. The strongest single piece of design in the repo:

```python
# backend/app/mcp/base.py:41-50
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Resolve execute() across the ENTIRE MRO (not just cls.__dict__): a
        # mixin placed before BaseWarehouseConnector could otherwise supply a
        # rogue execute and slip past a __dict__-only check.
        if cls.execute is not BaseWarehouseConnector.execute:
            raise TypeError(
                f"{cls.__name__} (or a base/mixin) overrides execute(); this "
                "is forbidden. Implement _run(). Scope enforcement lives in "
                "BaseWarehouseConnector.execute() by design.")
```

A data connector that tries to bypass scope enforcement **fails to
import**. That is a policy turned into a language-level guarantee, and it
is checked across the whole inheritance chain rather than just the class
body. Excellent.

### Where it cheats on its own rule

The project's own `CLAUDE.md` states the rule: *"a threshold, formula,
rule or persona that appears as a literal in Python is a bug."*
`intent_router.py` breaks it, extensively:

| What is hardcoded | Lines |
|---|---|
| `GROUP_BY_KEYWORDS` — English phrases per dimension | `18-31` |
| `DIMENSION_NOUNS` — payer vocabulary | `37-47` |
| `PHARMACY_HINTS`, `COMPLIANCE_HINTS` — payer regexes | `49-54` |
| NCPDP reject codes `70\|75\|76` | `277-278` |
| Fallback metrics `"pharmacy_cost"` / `"medical_cost"` | `254` |

`brain_dims()` (lines 92-126) is a partial retrofit — it lets
Brain-declared dimensions work, but only for `group_by`. Everything else
still routes through payer vocabulary. **The practical consequence: the
`retail_banking` domain is being understood through a health-insurance
dictionary.** This is the single largest piece of architectural debt.

### The trust gate is less discriminating than it looks

The system advertises a hard trust gate at 0.85, below which answers are
withheld. Grounding is the largest component at 0.35
(`trust/scoring.py:15-17`). But:

```python
# backend/app/semantic/graph.py:194-196
    if metrics or drugs:     # resolved hop -> floor for the trust gate
        coverage = max(coverage, 0.8)
```

Whenever *any* metric matches at all, grounding is floored at 0.8 — and
the comment says so out loud. So the largest input to the gate is close
to constant in the common case. The gate is real, but it is doing less
work than the number implies. Worth either removing the floor or being
clear in the docs about what the score actually measures.

### Accepted trade-offs (not defects)

- Two audit chains in one file is deliberate and documented; "unifying"
  them would produce false tamper positives.
- `PROTECTED_KEYS` in `brain_governance.py` is deliberately *not*
  Brain-editable, which is correct — otherwise one officer could lower
  the dual-control threshold and then act alone.

**Grade: A−.** A genuinely good idea, executed with discipline, with one
module that has not caught up.

---

## 3. Code quality — **B**

### The best thing about this codebase

**The comments.** They explain *why*, and very often name the specific bug
they prevent. This one is typical:

```python
# backend/app/agents/intent_router.py:96-104  (excerpt)
    # Symptom this fixes: a HEDIS metric declaring
    # ``dims: region, age_band, measure_name`` answered "CBP gap rate by age
    # band" with the REGION breakdown, because "by age band" matched nothing,
    # group_by stayed None, and the question resolved to the identical intent as
    # "what is the CBP gap rate?" — right down to the same semantic-cache key.
    # The cache was faithfully returning the correct answer for the intent it
    # was handed; the intent was wrong.
```

Someone reading that in two years will not re-break it. There are dozens
like it. This is the codebase's strongest asset and it should be
protected in review.

Also clean: **zero `TODO`/`FIXME`/`HACK`/`XXX`** anywhere, **zero bare
`except:`**, and 75% of functions carry type hints.

### The problems

**`main.py` is a god-file.** 1,868 lines, **95 routes**, 104 top-level
functions, and **141 function-local imports** used to dodge circular
dependencies. Most importantly, `_user(request)` is written out by hand
**81 times** instead of being a FastAPI `Depends()`.

That is not a style complaint — **it is the direct cause of the critical
security bug in the next section.** With 95 hand-written gates, the 16
that were forgotten are invisible to a reviewer. One
`Depends(current_user)` would have made the omission impossible.

**One policy, two divergent implementations.** The read-only SQL rule
exists twice:

| | `mcp/scope.py` | `data/db.py` |
|---|---|---|
| Normalizer | Character scanner, `69-128` | Regex, `58-63` |
| Blocked keywords | 15 | 9 |
| Excluded-table check | Yes, `142-148` | **No** |

Two copies of one rule, and the weaker copy is where the critical bug
lives. Brain path validation is also copy-pasted verbatim three times
(`brain.py:505`, `:524`, `:563`).

**Silent failure is the default.** 136 `except Exception` blocks against
only **7** `log.error`/`log.exception` calls in the entire application;
about 30 are `except: pass`. The intent ("degrade, never crash") is
right, but the execution means an enforcement failure looks exactly like
a normal refusal. (I checked the two in the audit-ledger modules
specifically — both are benign, and `ledger_v2.py:97-99` does log a
warning. The concern is the overall ratio, not the ledger.)

**Two virtualenvs are committed** — `backend/.venv` (52 MB) and
`backend/myenv` (41 MB), 93 MB of dependencies in the source tree.

**Grade: B.** Excellent craft at the function level, structural problems
at the file level.

---

## 4. Security — **C**

This grade is uncomfortable, because the security *design* here is better
than most production systems I see. The grade reflects one critical bug
and credential hygiene, not the thinking.

### 🔴 CRITICAL — any logged-in user can read every password hash

**`backend/app/main.py:1493-1502`**

```python
@app.get("/api/preview")
def preview():
    from .data.importer import data_status
    status = data_status()
    tables = {}
    for t in status["tables"]:
        r = run_query(f"SELECT * FROM {t['name']} LIMIT 20")
```

Three separate facts combine into the bug:

1. **No authorisation check.** The function takes no `Request` parameter,
   so it calls neither `_user()` nor `_require()`. The only gate is the
   middleware's "does this person have *some* valid token".
2. **It enumerates every table**, including the auth tables.
   `data_status()` (`importer.py:204-223`) reads all of `sqlite_master`
   and merely *labels* `users`, `roles`, `role_lob_access` and
   `user_attributes` as `origin: "system"` — it does not exclude them.
3. **It queries with no scope.** `run_query` is called without
   `allowed_lobs`, so `db.py:292` computes `scoped = False` and the query
   runs against the **full** database rather than the LOB sandbox. And
   `validate_sql` (`db.py:180-195`) checks read-only and
   single-statement, but **never consults the excluded-table list**.

**I did not infer this — I ran it.** Executing the exact call the
endpoint makes:

```
gate passed : True
columns     : ['user_id', 'display_name', 'email', 'role', 'password_hash']
row count   : 5
HASH LEAKED : yes -> Qv5IRE52A7Q+H2vohB... (len 69)
```

So an `ma_analyst` restricted to a single contract receives every user's
`user_id`, `email`, `role` and PBKDF2 `password_hash`, plus 20 rows of
every other table in the warehouse regardless of line of business.

The bitter part: **the library layer already blocks this**.
`tests/test_phase0.py:371-378` proves `SELECT password_hash FROM users`
fails *when `allowed_lobs` is passed*. This endpoint is the one path that
forgets to pass it, and no test covers the endpoint.

**Fix:** add `_require(_user(request), "audit_read")`, pass the caller's
`allowed_lobs` into `run_query`, and exclude the four system tables in
`data_status()`. Roughly an hour, including a test.

### 🔴 CRITICAL — credentials in the source tree

Three related items:

1. **A live-format LLM API key** sits in plaintext at
   `backend/app/data/settings.json` under `llm.api_key`. (I have not
   reproduced the value here.) **Rotate it today**, then remove it from
   the tree.
2. **The token-signing secret** `backend/app/data/.auth_secret` is
   present in the delivered folder (`security/auth.py:19-30`). Anyone
   holding this folder can forge a valid session **for any user,
   including a compliance officer** — which defeats maker-checker
   entirely. Rotate it, or set `OPPAL_AUTH_SECRET` from a secret manager.
3. **There is no `.gitignore` anywhere in the repository.** The moment
   someone runs `git init`, both files above are committed.

Credit where it is due: `backend/.dockerignore:12,18` *does* exclude both
files from the Docker image, with a well-reasoned comment explaining
exactly why. The container is protected. The source distribution is not —
and `docker-compose.yml:16` bind-mounts the real directory back in, so
compose still runs with that key.

### 🟠 HIGH — 16 of 95 routes never identify the caller

Beyond `/api/preview`, these never call `_user()` or `_require()`:

| Route | `main.py` | Exposure |
|---|---|---|
| `GET /api/users` | `330` | Full user list **with roles** |
| `GET /api/metrics` | `281` | Org-wide query volume, latency, token spend, cost |
| `GET /api/brain/{domain}/{fname}` | `1637` | All Brain content readable by any role |
| `GET /api/quality/missing\|history\|anomalies` | `1590,1610,1617` | Unscoped data-quality statistics |
| `POST /api/settings/test-llm` | `1835` | Any user can trigger a paid outbound call |

Brain *writes* are correctly gated (`1646`); reads are not. Also ungated:
`GET /api/settings` (`1801`), `GET /api/graph` (`1773`),
`POST /api/privacy/preview` (`1855`).

Being fair about the number: I counted these programmatically — 95 routes,
19 with no `_user()`/`_require()` call — and **three of the 19 are
deliberately public** (`/api/auth/login`, `/api/auth/users`,
`/api/health`). So the genuine gap is 16 routes, not 19.

As noted in the previous section, the root cause is structural — 81
hand-written `_user(request)` calls instead of one dependency.

### 🟠 HIGH — the audit ledger is not thread-safe

This matters more than usual, because tamper-evidence is the product's
central compliance claim.

```python
# backend/app/trust/audit_ledger.py:39-53
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]

def _write(record: dict):
    global _last_hash
    record["prev_hash"] = _last_hash
    line = json.dumps(record, default=str)
    _last_hash = _sha(line.encode())
```

A lock-free read-modify-write of a module global, and the hash is
**truncated to 16 hex characters (64 bits)**. This runs as middleware
(`main.py:41`) on *every* request, and FastAPI executes sync handlers on
a threadpool — so two concurrent requests will interleave and produce a
chain that no longer verifies.

The fix already exists in the repo: `ledger_v2.py:57` declares a
`threading.Lock()` and `:181` uses it. The v1 chain was simply never
brought up to match.

### 🟡 MEDIUM and below

- **Unauthenticated user enumeration.** `GET /api/auth/users`
  (`main.py:192-199`) is outside the auth exemption and lists every
  `user_id`. Combined with one shared password for all five seeded users
  (`data/generate.py:277`), that is a complete account-takeover path.
- **The rate limit is forgeable and leaks memory.** `_RATE`
  (`main.py:127`) is an in-memory dict whose keys are **never pruned**,
  and it is per-process, so it does nothing behind more than one
  instance. Worse, `frontend/nginx.conf:28-30` sets
  `set_real_ip_from 0.0.0.0/0` with `real_ip_header X-Forwarded-For`, so
  any caller picks their own bucket. The repo flags this in a comment
  (`nginx.conf:23-27`) as "only buys extra requests to your own app" —
  but that reasoning stops holding once it is the only throttle on
  `POST /api/auth/login`. There is no account lockout.
- **TLS verification and PHI scrubbing are UI toggles.**
  `network.ssl_verify=false` (`config.py:177` → `llm/base.py:190,195`)
  turns off certificate verification on all outbound model traffic, and
  `privacy.phi_protection` (`main.py:54-55`) turns off PHI scrubbing.
  Both default safe, but in a regulated deployment these should need more
  than one click.
- **CORS is wide open** — `allow_origins=["*"]`, `allow_methods=["*"]`
  (`main.py:39-40`). Being fair: `allow_credentials` is not set and the
  token lives in `sessionStorage` rather than a cookie, so a browser will
  not auto-attach it cross-origin. This is a hardening item, not an
  active hole.
- **Missing headers.** `main.py:175-178` sets `nosniff`,
  `X-Frame-Options`, `Referrer-Policy` and `Cache-Control`, but no
  `Content-Security-Policy` and no `Strict-Transport-Security`.
- **SQL built by string interpolation.** `sql_agent.py:249`
  (`f"{col_sql} = '{val}'"`) and `:333` do not escape quotes, while
  `:159` *does* (`.replace("'", "''")`). The codebase knows the pattern
  and applies it inconsistently. Being precise about the residual risk:
  `scope.py:135-140` blocks DML and multi-statement, and the sandbox is
  physically filtered, so this is scope/logic tampering rather than data
  destruction. It should still be parameterised.
- **No token revocation.** 8-hour TTL (`auth.py:32`), no `jti`, no
  blacklist; logout only clears the client copy. A leaked token is live
  for 8 hours.

### What is genuinely strong — do not regress these

| Control | Where |
|---|---|
| PBKDF2-SHA256, 120k iterations, random salt, constant-time compare | `auth.py:83-97` |
| Warehouse connections opened **read-only** at the driver level (`mode=ro`) | `db.py:77` |
| Connectors physically cannot bypass scope (fails at import) | `mcp/base.py:41-50` |
| Physical `:memory:` sandbox — scoped users query a filtered copy | `db.py:249-278` |
| SQL guard that survives comment/literal splitting (`DEL/**/ETE`) | `scope.py:69-128` |
| Schema-qualified reference to a governed table refused | `scope.py:150-168` |
| Dual-control keys deliberately not Brain-editable | `brain_governance.py` |
| Prompt-injection guard aimed at *stored* injection, with a 60-line rationale | `injection_guard.py:1-56` |
| Clean on: `eval`, `exec`, `pickle`, `os.system`, `shell=True`, md5/sha1 — zero hits | — |

**Grade: C.** The architecture deserves an A. One unauthorised endpoint
and three credential-hygiene failures pull it down. All four are same-day
fixes, and the grade moves to B+ once they are done.

---

## 5. Testing — **B−**

### What is impressive

- **1,627 checks, all passing**, verified by running the suite for this
  review. 33 test files + 19 demo scripts + 5 node tests, about 16,700
  lines of test code, 200 test functions.
- **The isolation setup is the best-engineered part of the test rig.**
  `run_all_tests.sh:12-41` redirects **ten** separate `OPPAL_*` paths into
  a scratch temp dir — warehouse, actions, cases, sims, workspaces,
  compensations, audit log, ledger head, ledger lock, and the
  active-domain pointer — with a `trap … EXIT` cleanup. The comments at
  lines 22-39 document two real cross-run contamination bugs this fixed.
- **The negative security tests are genuinely good.**
  `test_phase1.py:162-175` proves a connector cannot override `execute()`.
  `test_phase1.py:384-400` proves `SELECT password_hash FROM us/**/ers`
  is refused. `test_phase0.py:371-378` proves auth tables are unreachable.
- Zero test framework, by choice — each file is a standalone script with
  a `check()` helper. That is a defensible call for a demo product and it
  keeps the dependency list at six packages.

### The gap that matters

**There is not a single test at the HTTP layer.** No `TestClient`, no
`httpx` against the app. Every test imports Python functions directly.

Which means `security_middleware`, `_user()` and `_require()` are
**never exercised as middleware or dependencies** — and that is exactly
the layer where the critical `/api/preview` bug and the 16 ungated routes
live.

This is the most important sentence in the testing section:

> **1,627 checks pass, and the password hashes still leak.**

The tests prove the library is safe. Nothing proves the routes use the
library correctly. A dozen `TestClient` tests asserting
`401`/`403` per route would have caught both findings.

### Smaller gaps

- **A deleted test disappears silently.** Suites are registered in three
  hardcoded name lists (`run_all_tests.sh:83-86`, `102-112`, `132-133`)
  guarded by `[ -f … ] || continue`. Rename a file and it stops running,
  while the suite still reports PASS.
- **Pass counts are scraped from printed text**, not exit codes
  (lines 57, 89, 115, 136). A suite that exits 0 but prints nothing
  contributes 0 and still renders a green tick.
- **8 of 17 demo suites are never run** by the regression —
  `e2e_case2_upcoding`, `e2e_case3_encounter_feed`,
  `e2e_case4_network_leakage`, `e2e_case5_governance_review`,
  `e2e_case6_hedis_stars`, `demo_day_rehearsal`,
  `verify_analyst_walkthrough`, `verify_functionality_chain`. Note the
  pattern: the `medicare_sales` e2e cases are registered, their
  `healthcare_payer` twins are not.
- **No coverage measurement** of any kind, and **no CI** — no `.github/`,
  no `pytest.ini`, no `pyproject.toml`. Tests run only when someone
  remembers to type the command.
- **Almost no concurrency testing** — three files touch threads. Nothing
  tests SQLite write contention, connection-pool saturation, or the
  unlocked ledger chain.

**Grade: B−.** Strong volume and strong intent, with a structural blind
spot precisely where the real bugs turned out to be.

---

## 6. Production readiness — **C+**

### Better than expected

The **backend Dockerfile is well done** — and the comments explain every
choice:

- Runs as a **non-root** `appuser`, with `chown` rather than `chmod 777`.
- Layer order puts `requirements.txt` and `adduser` before `COPY . .`.
- `HEALTHCHECK` uses stdlib `urllib` rather than pulling in 10 MB of curl.
- `CMD exec uvicorn …` in shell form, so Cloud Run's `$PORT` expands *and*
  uvicorn stays PID 1 to receive `SIGTERM`.

`docker-compose.yml` deliberately does not publish the backend port, and
gates the frontend on `condition: service_healthy` because first boot
builds a 6,000-member warehouse.

### Blockers

**Logging is effectively switched off.** There are 20+ named loggers
(`oppal.mcp`, `oppal.db`, …) and **zero** `logging.basicConfig` or
`dictConfig` anywhere. Uvicorn configures only its own loggers, so the
root logger stays at WARNING and **every `log.info` in the application is
silently discarded in the container** — including the `mcp.execute` audit
lines. This is a one-line fix and it is currently costing all
observability.

**Failures are invisible to monitoring.** `/api/ask` wraps the entire
orchestrator and returns **HTTP 200** with `degraded: True`:

```python
# backend/app/main.py:226-236  (abbreviated)
    except Exception as e:                     # graceful degradation
        item = rq.enqueue(body.question, user.get("user_id", "anonymous"),
                          f"internal error: {type(e).__name__}: {e}", None, [])
        return {..., "summary": "⛔ Something went wrong processing this ...",
                "degraded": True, ...}
```

Good UX, but the 5xx rate is now structurally always zero — there is
nothing for an alert to fire on. There is also no `logging.exception`
here, so the only record of a crash is a review-queue ticket. And
`main.py:229` puts the **raw exception text** into that ticket, so a
database error quoting a row value would write PHI into the queue. (The
user-facing string correctly shows only the exception type.)

**SQLite is not configured for concurrency.** Seven separate database
files. **WAL mode is never enabled** — zero hits for `journal_mode`, so a
writer blocks all readers. `busy_timeout` is set in exactly one place
(`brain_governance.py:113`); the other 16 connection sites take the 5 s
default. The six writable stores each open a fresh connection per call
with `check_same_thread=False` — the classic `database is locked` setup.

**The connection pool is undersized.** `ConnectionPool` is **6**
(`db.py:70`) while uvicorn's threadpool defaults to 40, so
`acquire(timeout=5.0)` will start raising under load. Its `reset()`
(`:99-107`) also zeroes `_created` while connections are still checked
out.

**`/api/metrics` re-reads the entire ledger on every call**
(`main.py:284` → `audit_ledger.read_all:129-133`) with no rotation —
unbounded O(file) work on a dashboard endpoint.

**`/api/health` is unusable by an orchestrator** — it sits behind the auth
middleware and returns 401, which is precisely why the Dockerfile has to
probe `/api/auth/users` instead. No `/livez` vs `/readyz` split.

**Dependencies are unpinned.** All six lines of `requirements.txt` use
`>=` with no upper bound and no lockfile, so `docker build` today and in
six months produce different images. The frontend has a
`package-lock.json` but the Dockerfile runs `npm install`, not `npm ci`,
so it is ignored.

**No migrations, no backup story.** `ensure_db()` builds from scratch only
if the file is missing. There is no schema versioning and no upgrade
path; the compose bind mount is the entire durability plan.

**Single-instance only.** The rate limiter, the semantic cache and the
ledger head are all in-process or file-based. Scaling to two instances
breaks all three. This should be documented as a known limit rather than
discovered.

**Grade: C+.** Containerised competently, but unobservable, not
horizontally scalable, and without a data-lifecycle story.

---
---

# Part 3 — Prioritised fix list

The good news: **every P0 is hours of work, not weeks.** None of them
require re-architecting anything.

## P0 — before this touches real data

| # | Fix | Where | Effort |
|---|---|---|---|
| 1 | **Rotate the LLM API key**, then remove it from the tree and load it from a secret manager | `backend/app/data/settings.json` | 30 min |
| 2 | **Rotate the token-signing secret**; set `OPPAL_AUTH_SECRET` from the environment in any deployed instance | `backend/app/data/.auth_secret`, `security/auth.py:19-30` | 30 min |
| 3 | **Add a `.gitignore`** covering `app/data/settings.json`, `app/data/.auth_secret`, `*.db`, `.venv/`, `myenv/`, `__pycache__/`. The `.dockerignore` already has the right list to copy. | repo root | 5 min |
| 4 | **Fix `/api/preview`**: add `_require(_user(request), "audit_read")`, pass the caller's `allowed_lobs` into `run_query`, and exclude the four system tables in `data_status()` | `main.py:1493`, `importer.py:211` | 1 hr |
| 5 | **Write one `TestClient` test per route** asserting 401 unauthenticated and 403 for the wrong role. This is what stops #4 recurring. | `backend/tests/` (new file) | 1 day |
| 6 | **Delete the two committed virtualenvs** (93 MB) | `backend/.venv`, `backend/myenv` | 2 min |

## P1 — before the next customer deployment

| # | Fix | Where | Effort |
|---|---|---|---|
| 7 | **Put a `threading.Lock()` around the v1 ledger write** and stop truncating the hash to 64 bits. Copy the pattern from `ledger_v2.py:57,181`. | `trust/audit_ledger.py:39-53` | 1 hr |
| 8 | **Replace the 81 hand-written `_user(request)` calls with `Depends(current_user)`** and audit the 16 ungated routes. This is the structural fix for #4. | `main.py` throughout | 2 days |
| 9 | **Configure logging** — one `dictConfig` at startup. Currently every `log.info` is discarded. | `main.py` startup | 1 hr |
| 10 | **Enable WAL and set `busy_timeout`** on every SQLite connection; raise the pool size above 6 or cap the uvicorn threadpool to match | `data/db.py:70,77` + 16 other sites | 4 hrs |
| 11 | **Pin every dependency** with `==` and a lockfile; switch the frontend Dockerfile to `npm ci` | `requirements.txt`, `frontend/Dockerfile:6` | 1 hr |
| 12 | **Add CI** — a GitHub Actions workflow running `run_all_tests.sh` on every push. There is currently no `.github/` at all. | new `.github/workflows/` | 2 hrs |
| 13 | **Stop `/api/auth/users` leaking the user list**; give the healthcheck its own unauthenticated `/livez` instead | `main.py:192`, `Dockerfile:34` | 1 hr |
| 14 | **Make the rate limiter real** — prune old keys, move it out of process, and add a separate stricter budget plus lockout for failed logins | `main.py:127,151-163` | 4 hrs |

## P2 — worth scheduling

| # | Fix | Where |
|---|---|---|
| 15 | **Decide what the simulator's levers mean.** Either derive the ratio from the stressed costs, or rename the feature so nobody expects a causal model. | `simulation/engine.py:539` |
| 16 | **Move the payer vocabulary out of `intent_router.py`** into Brain Markdown — the last significant NFR-1 violation | `intent_router.py:18-54,277-278` |
| 17 | **Collapse the two read-only SQL policies into one.** Keep the `scope.py` implementation; delete the weaker copy in `db.py`. | `scope.py:69-148`, `db.py:29-63` |
| 18 | **Parameterise the SQL string interpolation** at `sql_agent.py:249,333` | `agents/sql_agent.py` |
| 19 | **Register suites by glob**, not by hardcoded name list, so a renamed test cannot silently vanish. Register the 8 orphaned demo suites. | `run_all_tests.sh:83-133` |
| 20 | **Add CSP and HSTS headers** | `main.py:175-178` |
| 21 | **Split `main.py` into `APIRouter` modules** by domain | `main.py` |
| 22 | **Add `logging.exception` to the `/api/ask` handler**, and scrub the raw exception text before it enters the review queue | `main.py:226-241` |
| 23 | **Rotate or cap the ledger file**, and stop `/api/metrics` re-reading it whole on every call | `trust/audit_ledger.py:129-133` |
| 24 | **Consider whether monitoring should be continuous.** Today nothing runs unless a human opens a tab. | product decision |

---
---

# Part 4 — File map

Where to look for what. Line ranges are the parts worth reading, not the
whole file.

## The request pipeline (chat box → answer)

| File | Lines | What it does |
|---|---|---|
| `backend/app/main.py` | `218-257` | `POST /api/ask` — the entry point |
| `backend/app/agents/orchestrator.py` | `78-312` | The 9-step pipeline and the audit trail |
| `backend/app/semantic/graph.py` | `148-219` | Grounding — matches your words to business concepts |
| `backend/app/agents/intent_router.py` | `218-384` | **All the regex.** Classifies the question |
| `backend/app/agents/sql_agent.py` | `95-327` | **All the SQL templates** |
| `backend/app/agents/viz_agent.py` | `10-115` | **Chart type, x-axis and y-axis** |
| `backend/app/agents/insight_agent.py` | `37-121` | The narration sentence templates |
| `backend/app/trust/scoring.py` | `15-23` | Trust weights and the 0.85 hard gate |
| `frontend/src/components/ChartArtifact.jsx` | `161,235` | Where the spec becomes pixels |

## The Context Brain (the editable Markdown)

| File | Lines | What it does |
|---|---|---|
| `backend/app/semantic/brain.py` | `82-367` | Parses all 11 Markdown files per domain |
| `backend/app/semantic/brain.py` | `210-215` | Policy rules parsing |
| `backend/app/semantic/brain.py` | `287-337` | Agent persona parsing |
| `backend/app/semantic/active_domain.txt` | — | **The live domain pointer.** Read this first when a suite fails mysteriously. |
| `.../domains/healthcare_payer/metrics.md` | — | Where `sql:` and `base:` per metric are declared |
| `.../domains/medicare_sales/policy_rules.md` | `14-23` | A worked example of a query-bearing rule |
| `.../domains/medicare_sales/agents.md` | `126-147` | A worked example of a declared agent |
| `.../domains/medicare_sales/simulation.md` | `8-34` | A worked example of `simulator_bindings` |

## Pillars, agents and cases

| File | Lines | What it does |
|---|---|---|
| `backend/app/analytics/rules_engine.py` | `342-383` | **The one handler that runs every declared rule** |
| `backend/app/analytics/rules_engine.py` | `386-430` | `pillar_summary` — the dashboard payload |
| `backend/app/agents/departmental.py` | `66-104` | Detector dispatch (the `brain_query` sentinel) |
| `backend/app/agents/departmental.py` | `107-199` | **`_declarative` — SQL row becomes a finding** |
| `backend/app/agents/departmental.py` | `470-533` | The only code that creates a case |
| `backend/app/agents/swarm.py` | `45-111` | Which agents activate (keyword matching) |
| `backend/app/agents/swarm.py` | `169-175` | The loop that runs them |
| `backend/app/agents/case_store.py` | `78-121` | `upsert_case` + deduplication |

## Simulation

| File | Lines | What it does |
|---|---|---|
| `backend/app/simulation/isolation.py` | `3-27` | The five isolation invariants |
| `backend/app/simulation/isolation.py` | `57-68` | `fork_brain` — deep copy with asserts |
| `backend/app/simulation/engine.py` | `130-146` | Per-domain lever renaming |
| `backend/app/simulation/engine.py` | `282-289` | What an override may never touch |
| `backend/app/simulation/engine.py` | `534-557` | **The arithmetic — and the modelling gap** |
| `backend/app/simulation/engine.py` | `766-822` | `promote()` — proposes, never writes |

## Security

| File | Lines | What it does |
|---|---|---|
| `backend/app/main.py` | `151-180` | Rate limit + deny-by-default middleware |
| `backend/app/main.py` | `91-121` | `_user()` and `_require()` |
| `backend/app/security/auth.py` | `83-128` | Password hashing and token signing |
| `backend/app/security/auth.py` | `35-79` | The permission map |
| `backend/app/mcp/base.py` | `41-50` | **The `execute()` final guard** |
| `backend/app/mcp/base.py` | `83-148` | The single enforcement path |
| `backend/app/mcp/scope.py` | `69-148` | The SQL guards |
| `backend/app/data/db.py` | `249-278` | The physical `:memory:` sandbox |
| `backend/app/privacy/phi_guard.py` | `145-164` | `GuardedProvider` |
| `backend/app/security/injection_guard.py` | `1-56` | Prompt-injection rationale — worth reading |

## Operations

| File | Lines | What it does |
|---|---|---|
| `backend/run_all_tests.sh` | `12-41` | Scratch-store isolation |
| `backend/run_all_tests.sh` | `53-133` | Suite registration (three hardcoded lists) |
| `backend/Dockerfile` | — | Container build |
| `docker-compose.yml` | — | Local stack |
| `backend/app/config.py` | `13-53,172-203` | Settings defaults and env export |

---

## Closing note

The instinct behind this codebase is right: put the rules where the
experts can read them, make the engine dumb and auditable, and never let
a model widen a permission. The `__init_subclass__` guard in
`mcp/base.py`, the physical `:memory:` sandbox in `db.py`, and the
scratch-store isolation in `run_all_tests.sh` are all better than what I
would expect at this stage.

The problems are not architectural. They are the ordinary consequence of
a demo growing into a product faster than its plumbing: a key left in a
settings file, a debug endpoint nobody re-read, and a test suite that
grew where it was easy to grow rather than where the risk was.

Fix the six P0 items and this is a B+ system with a credible compliance
story. Leave them and the first security review a customer runs will find
the password hashes in an afternoon.

---

*Review performed against the working tree at
`oppal-complete-v14.15/oppal_stage`, 21 September 2026. Test suite run and
passing (1,627 checks). The `/api/preview` finding was reproduced by
execution, not inferred from reading. No code was modified.*
