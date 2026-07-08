# Events Layer — Scoping Note

**Status: documented Phase 2 dependency, not a Phase 1 deliverable.** Nothing in this document is built; it exists so Phase 2 planning starts from an honest picture of what's available, rather than discovering the gaps mid-sprint.

## Why there's no single clean API for local event data

The events that move housing inventory at the metro level — a large employer relocating or laying off staff, a zoning change that unlocks new construction, a city housing initiative, a major employer announcing a new campus — don't share a publisher, a schema, or a update cadence:

- They're **locally generated**: a WARN Act filing lives with a state labor department, a zoning change lives in a city planning department's agenda PDF, an employer announcement lives in a press release or local news story. There is no federal agency whose job is to aggregate "things that affect local housing supply/demand" the way FRED aggregates macro series or Census aggregates demographics.
- They're **heterogeneous in structure**: a mass layoff notice has a date, employer, and headcount; a zoning change has a parcel, a vote outcome, and an effective date; a news story has none of that structure natively — it has to be extracted.
- They're **inconsistently digitized**: some states publish WARN notices as a queryable database, others as a PDF list updated monthly, and coverage/format changes without notice.

This is a genuinely different problem from Task 1/2's pulls (FRED, Zillow, Census, NOAA), which all have stable, documented, machine-readable endpoints. An events layer means either aggregating many inconsistent local sources, or accepting a noisier proxy (news text) and extracting structure from it.

## Candidate sources, ranked by ease of access

### 1. FEMA disaster declarations API — easiest
- Endpoint: `https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries` (OpenFEMA, confirmed present in the OpenFEMA dataset catalog during this pipeline's research — see [pull_intrinsic.py](pull_intrinsic.py) docstring for why the *National Risk Index* dataset, by contrast, isn't in that catalog and has no stable download).
- Free, no API key required, clean JSON/CSV, queryable by state/county/date/incident type.
- **Coverage gap**: natural disasters only (hurricanes, floods, wildfires, severe storms declared for federal assistance). Says nothing about employer moves, zoning, or local housing policy — it's an events layer in name only for those categories, though it usefully complements the NOAA hazard signal already in `intrinsic_static.csv` with a *time-stamped* disaster-declaration series rather than a static 10-year count.

### 2. WARN Act mass layoff notices — moderate, real effort
- Every state runs its own WARN Act (Worker Adjustment and Retraining Notification) system, since enforcement is state-level even though the underlying law is federal.
- No unified API. Some states (e.g. California's EDD) publish a downloadable list; others require scraping a web page or PDF; formats and update frequency vary by state with no standardization.
- Directly relevant to housing inventory (a large layoff can precede a wave of relocations/listings), but building this means 15+ separate state-specific scrapers (one per MSA's home state, more if an MSA spans multiple states, e.g. Washington DC spans DC/VA/MD/WV) with no shared maintenance story — each state can change its page/format independently and silently.

### 3. Google News RSS per MSA — stretch option, noisiest
- Trivial to query (`news.google.com/rss/search?q=...`), free, no key.
- Returns headlines/snippets, not structured events — would need an LLM extraction step to classify each headline as (employer move / zoning change / housing initiative / irrelevant) and pull out entities and dates, with no ground truth to validate extraction accuracy against.
- High recall, low precision, and the extraction step itself becomes a modeling problem (prompt design, hallucination risk, cost per MSA per week) rather than a data pull.

## Recommendation

**Start Phase 2 with FEMA disaster declarations only as MVP event coverage.** It's the only source with a real API, no auth, and stable structure — matching the reliability bar Task 1/2's pulls set. Defer WARN Act scraping and Google News/LLM extraction to a later iteration, once there's a specific research question that natural-disaster events alone can't answer.

This keeps Phase 2's first event feature honest about what it covers (climate/disaster shocks) rather than implying broader "local events" coverage the MVP doesn't actually have.
