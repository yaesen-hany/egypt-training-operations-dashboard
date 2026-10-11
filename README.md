# Insight Dashboard Builder

**Samsung Innovation Campus — Chapter 8: Streamlit, Data Visualization & AI-Assisted Dashboard Building**
** Units 41–42**

Build an interactive Streamlit dashboard using charts, maps, filters, and an AI-assisted iteration log.

---

## Scenario

- A program team wants to monitor training activity across cities and channels.
- The dashboard must help users explore performance, trends, geographic coverage, and key insights.
- Use Streamlit to turn raw tabular data into a clear, interactive product.

## Core Dataset Fields

| Field | Description |
|---|---|
| `date` | Date of the training activity |
| `city`, `governorate` | Location |
| `latitude`, `longitude` | Coordinates for mapping |
| `track` | e.g. AI, Data, Coding |
| `channel` | e.g. Online, Offline, Hybrid |
| `participants`, `completed` | Enrollment and completion counts |
| `satisfaction_score` | Participant satisfaction |
| `cost`, `revenue` | Financial values |

**Submission:** one Streamlit `.py` file .

---

## 1. Dataset & Data Preparation

Use or create a dataset with **at least 30 rows**. Each row represents one training activity or learning event.

- Load data from a CSV, or create a DataFrame inside the script.
- Convert the date column using `pd.to_datetime()`.
- Create at least one derived metric, such as `completion_rate` or `profit`.

**Sample rows**

```
date        | city       | track  | channel | participants | completed | satisfaction | cost  | revenue
2026-09-01  | Cairo      | AI     | Online  | 35           | 28        | 4.5          | 9000  | 16000
2026-09-03  | Giza       | Data   | Offline | 42           | 37        | 4.2          | 12000 | 21000
2026-09-05  | Alexandria | Coding | Hybrid  | 30           | 21        | 3.9          | 8000  | 14500
2026-09-07  | Aswan      | AI     | Offline | 24           | 19        | 4.7          | 7000  | 12500
```

**Required inspection**

- Show `df.head()`, `df.shape`, and a descriptive summary.
- Display the full dataset using `st.dataframe()`.
- Use `st.metric()` for at least three KPIs.

---

## 2. Dashboard Layout Requirements

### Sidebar Controls
- Date range filter
- Track selector
- City or governorate multiselect
- Channel selector
- Minimum satisfaction or completion-rate slider

### Main Dashboard
- Title, caption, and short markdown explanation
- At least 3 KPI metrics with deltas
- Use columns to organize KPIs and charts
- Use tabs for **Overview**, **Trends**, **Geography**, and **Data Explorer**

### Interactivity
- All charts must react to the selected filters
- At least one `st.data_editor()` or editable table
- One `st.download_button()` or exported filtered table
- Use `session_state` to remember at least one user preference

> The app should still run correctly when filters return a small subset of data.

---

## 3. Visualization Requirements

### Built-in Streamlit Charts
- `st.line_chart()` for a time trend
- `st.bar_chart()` for category comparison
- `st.area_chart()` for cumulative or multi-series activity

### Altair
- Create at least one declarative chart.
- Use `pd.melt()` if needed to reshape metrics.
- Map x, y, and color channels clearly.
- Render using `st.altair_chart()`.

### Plotly
- Create one interactive Plotly chart (pie/donut, stacked bar, or interactive line).
- Render using `st.plotly_chart()`.

### Map
- Use `st.map()` **or** a Folium map.
- Show city locations using latitude/longitude.
- For Folium: include popups or `CircleMarker`s and render through `st.components.v1.html()`.

---

**Required final prompt structure**

```
<system_directive>
<role>
You are an expert Senior Streamlit Architect and Enterprise Data Visualization Engineer. You write clean, robust, idiomatic Python 3.10+ code that strictly adheres to defensive programming standards, seamless failover logic, and clean visual design.
</role>

<context>
Project: Nationwide Technical Training Operations Intelligence Dashboard (Egypt - Q4 2026).
Coursework: Samsung Innovation Campus (SIC) — Chapter 8 (Units 41–42): Data Visualization & AI-Assisted Dashboard Building.
Target File: app.py (Single, complete, production-ready executable script).
</context>

<primary_objective>
Construct an end-to-end, zero-crash, highly interactive operations dashboard using Streamlit to monitor technical training cohorts across 8 Egyptian metropolitan centers. The application must turn raw operational records into reactive, multi-dimensional analytics, featuring custom charts (Streamlit native, Altair, Plotly), cartographic mapping (Folium with native fallback), and a resilient dual-engine AI operations assistant.
</primary_objective>

<data_contract>
1. Data Ingestion & Synthesis:
   - Check if a local file named 'training_activity.csv' exists; if present, load it and parse the date column using pd.to_datetime().
   - If absent, deterministically synthesize a 60-row DataFrame spanning 2026-09-01 through 2026-12-31 using a fixed random seed (np.random.default_rng(42)).
2. Canonical Data Domains:
   - 8 Egyptian Centers: Cairo, Giza, Alexandria, Aswan, Mansoura (Dakahlia), Assiut, Tanta (Gharbia), Luxor (with accurate paired latitudes and longitudes).
   - Training Tracks: ['AI', 'Data', 'Coding'].
   - Delivery Channels: ['Online', 'Offline', 'Hybrid'].
3. Required Schema Fields:
   - date, city, governorate, latitude, longitude, track, channel, participants, completed, satisfaction_score, cost, revenue.
4. Mathematical & Analytical Derivations:
   - completion_rate = completed / participants (safeguarded against zero division).
   - profit = revenue - cost.
   - profit_margin_pct = (profit / revenue) * 100 (safeguarded against zero revenue).
   - cost_per_participant = cost / participants (safeguarded against zero participants).
   - month = Period string in 'YYYY-MM' format.
</data_contract>

<sidebar_controls>
- Reset Button: "↺ Clear filters and show all data" using an on_click callback to instantly restore all session state filters to their full global scope.
- Date Range Selector: st.date_input with defensive tuple length verification.
- Multiselect Filters: Dedicated multiselects for Training Track, Target City, and Delivery Channel (defaulting to all options selected).
- Satisfaction Rating Filter: An integer-based slider (min_value=1, max_value=5, default=3, step=1, format="%d ★") eliminating decimal drift, handle offset glitches, and RTL browser alignment issues.
- Primary Metric Selector: st.segmented_control (with graceful fallback to st.selectbox) allowing dynamic switching between ['profit', 'revenue', 'participants'].
- Filter Behavior: If any multiselect is cleared, return an empty DataFrame safely without throwing exceptions; display reactive session count toasts upon filter updates.
</sidebar_controls>

<layout_and_feature_specifications>
1. Header & Executive KPI Strip:
   - Render page configuration (st.set_page_config) as the absolute first Streamlit statement.
   - Header banner with dynamic vector/procedural illustration via PIL (with fallback if PIL is missing).
   - 4 bordered st.metric cards: Total Enrollment, Average Completion Rate, Net Profit Generated, and Operating Margin. Each card must calculate deltas compared against per-session baseline averages.
   - Celebratory Trigger: Execute st.balloons() when the active filtered completion rate meets or exceeds 90% (guarded against multiple triggers using session state signatures).

2. Tab 1: Executive Operations Summary (Overview):
   - Dynamic st.bar_chart displaying the selected primary metric aggregated by city.
   - Dark-themed Plotly Donut chart displaying student intake by track.
   - 3 operational benchmark highlight cards (Top City by Headcount, Most Profitable Track, Average Satisfaction Rating).
   - st.bar_chart showing participant intake across delivery channels.
   - Plotly grouped bar chart comparing Revenue vs. Cost across cities.

3. Tab 2: Temporal Trends & Pipeline:
   - st.line_chart tracking monthly intake across Q4 2026.
   - Multi-series line chart tracking monthly intake broken down by track.
   - st.area_chart showing cumulative nationwide enrollment over time.
   - Declarative Altair chart (melted Enrolled vs. Completed students) featuring interactive mouseover hover-dimming.

4. Tab 3: Geographic Footprint:
   - Interactive Folium map with CircleMarkers scaled proportionally to student enrollment, with tooltips and popups.
   - Provide silent, graceful fallback to native st.map if folium or streamlit_folium is not installed.
   - Regional performance ledger rendered via st.dataframe with custom column formatting (percentages, currency).

5. Tab 4: Data Explorer & Inspection:
   - Dimensions summary strip (Filtered Rows, Schema Dimensions, Memory Footprint in KB).
   - 4 independent persistent toggle buttons controlling:
     * Preview (first 5 rows via df.head()).
     * Descriptive Statistics (df.describe()).
     * Complete Filtered Dataset (st.dataframe).
     * Interactive Sandbox (st.data_editor with fixed row bounds).
   - st.download_button exporting filtered data as a UTF-8 CSV payload.

6. Tab 5: Operations Intelligence Assistant (Dual-Engine System):
   - Dynamic Context Injection: Aggregate live filtered metrics (total enrollment, completion %, revenue, profit, leading city, active tracks) into the system prompt.
   - Dual-Engine Query Mechanism:
     * Primary (Online): Query Google Gemini models using the modern `google-genai` SDK, testing modern models progressively (gemini-3.5-flash, fallback to gemini-2.5-flash-lite).
     * Secondary (Offline Local Fallback): If the API key is empty, offline, or returns 503/429 errors, silently route questions to an embedded deterministic knowledge base without showing errors or stack traces.
   - Credential Cleanliness:
     * Do NOT render API key text inputs, expanders, or configuration forms in the UI.
     * Safely resolve credentials: check internal script constant -> environment variable -> st.secrets (wrapped in a strict try-except block to eliminate StreamlitSecretNotFoundError).
</layout_and_feature_specifications>

<invariants_and_guardrails>
- [CRITICAL] Zero CSS/HTML Injections: Pure native Streamlit components only. Absolutely no unsafe HTML styling hacks.
- [CRITICAL] 100% English UI: All titles, labels, chart axes, captions, tooltips, and assistant dialogues must strictly remain in English.
- [CRITICAL] Mathematical Safeguards: All ratio, mean, and division operations must be guarded against empty slices or zero denominators.
- [CRITICAL] Defensive Import Routine: Include an automated dependency checking routine at the top of the file to auto-install missing packages if needed.
- [CRITICAL] Execution Integrity: Deliver the complete, fully implemented, runnable code without placeholders, abbreviations, or `# TODO` comments.
</invariants_and_guardrails>

<thinking_process>
Before generating the final script, internally verify:
1. Is st.set_page_config strictly the first Streamlit command executed?
2. Are all calculations (margins, completion rates) protected from division by zero when an empty filter is applied?
3. Does the satisfaction slider use whole integer numbers (1 to 5) to prevent fractional desynchronization and RTL inverted text glitches?
4. Is st.secrets wrapped inside a safe try-except block to eliminate StreamlitSecretNotFoundError when secrets.toml is absent?
5. Does the assistant silently fall back to local records when no API key is provided?
</thinking_process>

<output_format>
Output the entire, executable `app.py` inside a single Python code block.
</output_format>
</system_directive>
```

> AI can help generate code, but the final app must be tested, cleaned, and explained by the student.

---

## 4. Deliverables

- [ ] `app.py` file
- [ ] Dataset file or generated dataset code

## Submission Checklist

### Technical
- [ ] Dataset loads successfully
- [ ] Date converted with `pd.to_datetime()`
- [ ] Sidebar filters work together
- [ ] 3+ KPI metrics included
- [ ] `st.line_chart` / `st.bar_chart` / `st.area_chart` used
- [ ] Altair chart included
- [ ] Plotly chart included
- [ ] Map included using `st.map` or Folium

### Product
- [ ] `st.dataframe` or `st.data_editor` included
- [ ] Tabs and columns used properly
- [ ] Session State or cache used meaningfully
- [ ] No broken chart when filter output is small
- [ ] AI prompt log is attached
- [ ] README includes run command
- [ ] Screenshots are attached
- [ ] Final reflection answered

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
